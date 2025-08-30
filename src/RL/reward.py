import math, zlib, numpy as np
from typing import Dict, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..verify_memorization import Ngram8Index

def calc_perplexity(text: str, model: PreTrainedModel, tok: PreTrainedTokenizer, device: torch.device) -> float:
    ids = tok.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids, labels=ids)
        loss = out.loss
    return float(torch.exp(loss).item())

def membership_score(ppl_xl: float, zlen: int, ntok: int) -> float:
    """Simple monotonic proxy used in your earlier pipeline."""
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
    """Return shaped reward: proxy + verified hits."""
    ppl = calc_perplexity(text, ppl_model, task_tok, device)
    zlen = len(zlib.compress(text.encode("utf-8")))
    ids = task_tok.encode(text, add_special_tokens=False)
    proxy = membership_score(ppl, zlen, len(ids))

    hits = 0
    if idx is not None and len(ids) >= window_k:
        # Shift by +1 like you used in index builder to avoid zeros, if applicable.
        ids_p1 = [i + 1 for i in ids]
        hits = idx.count_hits(ids_p1, k=window_k)

    # Piecewise shaping: add a bonus per verified hit.
    R = w_proxy * proxy + w_hits * hits
    return {"reward": R, "ppl": ppl, "z": zlen, "ntok": len(ids), "hits": hits, "proxy": proxy}
