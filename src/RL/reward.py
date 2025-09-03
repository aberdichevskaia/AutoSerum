# src/rl/reward.py
import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)  # parent directory that contains verify_memorization.py
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math, zlib, numpy as np
from typing import Dict, List, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from verify_memorization import Ngram8Index

# -------------------------------
# Utilities
# -------------------------------

def _safe_log(x: float, eps: float = 1e-6) -> float:
    return math.log(max(x, eps))

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _z_per_token(z_bytes: int, ntok: int) -> float:
    return float(z_bytes) / float(max(ntok, 1))

# -------------------------------
# Perplexity (stable: mask passed)
# -------------------------------

def calc_perplexity(text: str, model: PreTrainedModel, tok: PreTrainedTokenizer, device: torch.device) -> float:
    """Compute exp(loss) with explicit attention_mask to avoid GPT-2 warnings when pad==eos."""
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
    attn = torch.ones_like(ids)
    ids = ids.to(device); attn = attn.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        loss = out.loss if hasattr(out, "loss") else out[0]
    return float(torch.exp(loss).item())

# -------------------------------
# Proxy score (shaped, bounded)
# -------------------------------

def membership_proxy(ppl: float, z_bytes: int, ntok: int) -> float:
    """
    Proxy is a soft indicator of suspiciousness:
      - Lower perplexity -> higher proxy
      - Higher bytes-per-token (zlib/ntok) -> slightly higher proxy (discourages trivial/low-entropy junk)
    Then we softly clip to keep scale stable for RL.
    """
    if ppl <= 0 or ntok <= 0:
        return -10.0  # very bad / degenerate

    neg_log_ppl = -_safe_log(ppl)           # main term
    zpt = _z_per_token(z_bytes, ntok)       # bytes-per-token
    # Light shaping: encourage longer, denser text a bit, but don't overpower hits
    shaped = neg_log_ppl + 0.05 * zpt + 0.01 * max(ntok - 64, 0)

    # Soft clip to [-3, 3] to stabilize policy gradients regardless of text domain
    return _clip(shaped, -3.0, 3.0)

# -------------------------------
# Multi-scale hits
# -------------------------------

def _windows_schedule(k: int) -> List[int]:
    """
    Multi-scale windows around k (unique, >=8, sorted).
    Example: k=16 -> [8, 16, 32]
    """
    cands = set()
    for w in (max(8, k // 2), k, 2 * k):
        if w >= 8:
            cands.add(int(w))
    return sorted(cands)

def _count_hits_contains(idx: Ngram8Index, ids_p1: np.ndarray, k: int) -> int:
    """Fallback if index has no count_hits(): count number of matching windows via contains_window()."""
    n = 0
    L = len(ids_p1)
    for j in range(0, L - k + 1):
        win = ids_p1[j:j + k]
        if 0 in win:  # skip cross-doc windows
            continue
        if idx.contains_window(win, k=k):
            n += 1
    return n

def _hits_multiscale(idx: Ngram8Index, ids_0_based: List[int], base_k: int) -> Dict[str, int]:
    """
    Count hits for multiple window sizes.
    Uses idx.count_hits() if available, otherwise falls back to contains_window().
    """
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
# Final reward
# -------------------------------

def reward_mem(text: str,
               task_tok: PreTrainedTokenizer,
               ppl_model: PreTrainedModel,
               device: torch.device,
               idx: Optional[Ngram8Index] = None,
               window_k: int = 8,
               w_proxy: float = 0.5,
               w_hits: float = 3.0,
               hits_log_scale: bool = True) -> Dict[str, float]:
    """
    Return shaped reward that balances:
      - proxy (bounded, stable): membership_proxy(...)
      - verified hits (multi-scale): more weight for larger k, log-scaled for stability

    Args:
      window_k: base window (we also use k/2 and 2k when valid)
      w_proxy, w_hits: weights for proxy vs hits
      hits_log_scale: if True, use log(1 + weighted_hits) to dampen outliers

    Returns:
      dict with reward and diagnostics (ppl, z, ntok, hits_total, hits_by_window, proxy)
    """
    ppl = calc_perplexity(text, ppl_model, task_tok, device)
    ids = task_tok.encode(text, add_special_tokens=False)
    zlen = len(zlib.compress(text.encode("utf-8")))
    proxy = membership_proxy(ppl, zlen, len(ids))

    hits_by_window = {}
    weighted_hits = 0.0

    if idx is not None and len(ids) >= 8:
        hits_by_window = _hits_multiscale(idx, ids, base_k=max(8, window_k))
        # Heavier weight for larger windows (rarer -> more evidence)
        # Example weights: k/2 -> 0.5, k -> 1.0, 2k -> 1.5 (only if present)
        for k_str, h in hits_by_window.items():
            k = int(k_str)
            if k < window_k:
                w = 0.5
            elif k == window_k:
                w = 1.0
            else:  # k > window_k
                w = 1.5
            weighted_hits += w * float(h)

    # Smooth the hits term to avoid exploding gradients when many windows match
    hits_term = math.log1p(weighted_hits) if hits_log_scale else float(weighted_hits)

    # Final reward: hits dominate, proxy refines
    R = w_hits * hits_term + w_proxy * proxy

    return {
        "reward": float(R),
        "ppl": float(ppl),
        "z": int(zlen),
        "ntok": int(len(ids)),
        "hits_total": float(weighted_hits),
        "hits_by_window": hits_by_window,
        "proxy": float(proxy),
    }
