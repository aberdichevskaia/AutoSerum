import math, zlib, numpy as np
from typing import Dict, List, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # AutoSerum/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.verify_memorization import Ngram8Index

try:
    from src.config import REWARD as REWARD_CFG
except Exception:
    REWARD_CFG = {
        "mode": "proxy",
        "w_hits": 3.0,
        "w_proxy": 0.5,
        "hits_log_scale": True,
    }

_EPS = 1e-6


# -------------------------------
# Utilities
# -------------------------------
def _safe_log(x: float, eps: float = _EPS) -> float:
    return math.log(max(x, eps))

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _z_per_token(z_bytes: int, ntok: int) -> float:
    return float(z_bytes) / float(max(ntok, 1))

def _nan_to_num(x: float) -> float:
    if x != x or x in (float("inf"), float("-inf")):
        return 0.0
    return x


# -------------------------------
# Perplexity / mean log p
# -------------------------------
@torch.no_grad()
def calc_perplexity(text: str, model: PreTrainedModel, tok: PreTrainedTokenizer, device: torch.device) -> float:
    """exp(mean NLL) with explicit attention_mask."""
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
    attn = torch.ones_like(ids)
    ids = ids.to(device); attn = attn.to(device)
    out = model(input_ids=ids, attention_mask=attn, labels=ids)
    loss = out.loss if hasattr(out, "loss") else out[0]
    ppl = float(torch.exp(loss).item())
    return max(ppl, _EPS)

@torch.no_grad()
def mean_logp(text: str, model: PreTrainedModel, tok: PreTrainedTokenizer, device: torch.device) -> float:
    """Mean log p = - mean NLL."""
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
    attn = torch.ones_like(ids)
    ids = ids.to(device); attn = attn.to(device)
    out = model(input_ids=ids, attention_mask=attn, labels=ids)
    loss = out.loss if hasattr(out, "loss") else out[0]
    return float(-loss.item())


# -------------------------------
# Proxy (bounded)
# -------------------------------
def membership_proxy(ppl_xl: float, z_bytes: int, ntok: int) -> float:
    """
    Proxy-signal of "suspicion"
    """
    if ppl_xl <= 0 or ntok <= 0:
        return -10.0
    neg_log_ppl = -_safe_log(ppl_xl)
    zpt = _z_per_token(z_bytes, ntok)
    shaped = neg_log_ppl + 0.05 * zpt + 0.01 * max(ntok - 64, 0)
    return _clip(shaped, -3.0, 3.0)


# -------------------------------
# Multi-scale hits (white-box index)
# -------------------------------
def _windows_schedule(k: int) -> List[int]:
    cands = set()
    for w in (max(8, k // 2), k, 2 * k):
        if w >= 8:
            cands.add(int(w))
    return sorted(cands)

def _count_hits_contains(idx: Ngram8Index, ids_p1: np.ndarray, k: int) -> int:
    n = 0
    L = len(ids_p1)
    for j in range(0, L - k + 1):
        win = ids_p1[j:j + k]
        if 0 in win:
            continue
        if idx.contains_window(win, k=k):
            n += 1
    return n

def _hits_multiscale(idx: Ngram8Index, ids_0_based: List[int], base_k: int) -> Dict[str, int]:
    ids_p1 = np.asarray([t + 1 for t in ids_0_based], dtype=np.uint32)  # AUX shift (+1)
    out = {}
    for k in _windows_schedule(base_k):
        if hasattr(idx, "count_hits"):
            h = int(getattr(idx, "count_hits")(ids_p1, k=k))
        else:
            h = _count_hits_contains(idx, ids_p1, k=k)
        out[str(k)] = h
    return out


# -------------------------------
# Metrics computation
# -------------------------------
def _compute_common_metrics(
    text: str,
    tok: PreTrainedTokenizer,
    ppl_model_xl: PreTrainedModel,
    device: torch.device,
    idx: Optional[Ngram8Index],
    window_k: int,
) -> Dict[str, float]:
    ids = tok.encode(text, add_special_tokens=False)
    ntok = len(ids)
    z_bytes = len(zlib.compress(text.encode("utf-8")))
    zpt = _z_per_token(z_bytes, ntok)
    ppl_xl = calc_perplexity(text, ppl_model_xl, tok, device)
    proxy = membership_proxy(ppl_xl, z_bytes, ntok)

    # hits
    hits_by_window: Dict[str, int] = {}
    weighted_hits = 0.0
    if idx is not None and ntok >= 8:
        hits_by_window = _hits_multiscale(idx, ids, base_k=max(8, window_k))
        for k_str, h in hits_by_window.items():
            k = int(k_str)
            if k < window_k:  w = 0.5
            elif k == window_k: w = 1.0
            else:              w = 1.5
            weighted_hits += w * float(h)

    hits_log_scale = bool(REWARD_CFG.get("hits_log_scale", True))
    hits_term = math.log1p(weighted_hits) if hits_log_scale else float(weighted_hits)

    return {
        "ntok": int(ntok),
        "z": int(z_bytes),
        "zpt": float(zpt),
        "ppl_xl": float(ppl_xl),
        "proxy": float(proxy),
        "hits_total": float(weighted_hits),
        "hits_term": float(hits_term),
        "hits_by_window": hits_by_window,
    }


# -------------------------------
# Unified reward (modes: naive, proxy, gap)
# -------------------------------
def reward_mem(
    text: str,
    task_tok: PreTrainedTokenizer,
    ppl_model_xl: PreTrainedModel,
    device: torch.device,
    idx: Optional[Ngram8Index] = None,
    window_k: int = 8,

    ppl_model_small: Optional[PreTrainedModel] = None,

    mode: Optional[str] = None,
    w_hits: Optional[float] = None,
    w_proxy: Optional[float] = None,
) -> Dict[str, float]:
    """
    Modes:
      - "naive":  reward = w_hits * hits_term
      - "proxy":  reward = w_hits * hits_term + w_proxy * proxy
      - "gap":    reward = (mean_logp(main) - mean_logp(ref)) + w_hits * hits_term
    """
    m = (mode or REWARD_CFG.get("mode", "proxy")).lower()
    W_H = w_hits  if w_hits  is not None else float(REWARD_CFG.get("w_hits", 3.0))
    W_P = w_proxy if w_proxy is not None else float(REWARD_CFG.get("w_proxy", 0.5))

    met = _compute_common_metrics(text, task_tok, ppl_model_xl, device, idx, window_k)

    if m == "naive":
        R = W_H * met["hits_term"]

    elif m == "proxy":
        R = W_H * met["hits_term"] + W_P * met["proxy"]

    elif m == "gap":
        gap = 0.0
        if ppl_model_small is not None:
            mlp_main = mean_logp(text, ppl_model_xl, task_tok, device)
            mlp_ref  = mean_logp(text, ppl_model_small, task_tok, device)
            gap = _nan_to_num(mlp_main - mlp_ref)
        R = gap + W_H * met["hits_term"]
        met["gap"] = float(gap)

    else:
        # fallback to proxy
        R = W_H * met["hits_term"] + W_P * met["proxy"]

    out = {
        "reward": float(R),
        "mode": m,
        "w_hits": float(W_H),
        "w_proxy": float(W_P),
        **met,
    }
    return out
