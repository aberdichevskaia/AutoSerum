import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)  # parent directory that contains verify_memorization.py
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import math, zlib, numpy as np
from typing import Dict
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from verify_memorization import Ngram8Index

def calc_perplexity(text: str, model: PreTrainedModel, tok: PreTrainedTokenizer, device: torch.device) -> float:
    # Always pass attention_mask to avoid GPT-2 warning when pad==eos
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt")
    attn = torch.ones_like(ids)
    ids = ids.to(device); attn = attn.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        loss = out.loss if hasattr(out, "loss") else out[0]
    return float(torch.exp(loss).item())

def membership_score(ppl_xl: float, zlen: int, ntok: int) -> float:
    # Simple monotonic proxy; lower is "more suspicious"
    if ppl_xl <= 0 or ntok <= 0:
        return float("-inf")
    return -math.log(ppl_xl) + 0.001 * zlen + 0.02 * max(ntok - 80, 0)

def reward_mem(text: str,
               task_tok: PreTrainedTokenizer,
               ppl_model: PreTrainedModel,
               device: torch.device,
               idx: Ngram8Index = None,
               window_k: int = 8,
               w_proxy: float = 1.0,
               w_hits: float = 3.0) -> Dict[str, float]:
    """Return shaped reward: proxy + verified hits (binary hit)."""
    ppl = calc_perplexity(text, ppl_model, task_tok, device)
    zlen = len(zlib.compress(text.encode("utf-8")))
    ids = task_tok.encode(text, add_special_tokens=False)
    proxy = membership_score(ppl, zlen, len(ids))

    hit = False
    if idx is not None and len(ids) >= window_k:
        # +1 shift (0 is a sentinel in AUX)
        ids_p1 = np.asarray([i + 1 for i in ids], dtype=np.uint32)
        # slide a k-window and query the index
        for j in range(0, len(ids_p1) - window_k + 1):
            win = ids_p1[j:j + window_k]
            if 0 in win:
                continue
            if idx.contains_window(win, k=window_k):
                hit = True
                break

    hits = 1 if hit else 0
    R = w_proxy * proxy + w_hits * hits
    return {
        "reward": float(R),
        "ppl": float(ppl),
        "z": int(zlen),
        "ntok": int(len(ids)),
        "hits": int(hits),
        "proxy": float(proxy),
    }
