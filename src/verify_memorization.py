import os
import sqlite3
import numpy as np
import zlib
from transformers import GPT2TokenizerFast


class Ngram8Index:
    """Lightweight 8-gram inverted index: hash(8 tokens) -> candidate positions; then exact 50-token check."""
    def __init__(self, auxidx_dir: str, n: int = 8):
        self.tokens = np.memmap(os.path.join(auxidx_dir, "tokens.uint32"), dtype=np.uint32, mode="r")
        self.con = sqlite3.connect(os.path.join(auxidx_dir, "ng8.sqlite"), check_same_thread=False)
        self.cur = self.con.cursor()
        self.N = n

    @staticmethod
    def _hash(span):
        h = 1469598103934665603
        for x in span:
            h ^= int(x)
            h *= 1099511628211
            h &= (1 << 64) - 1
        return h

    @staticmethod
    def _to_i64(u):
        return int(u if u < (1 << 63) else u - (1 << 64))

    def contains_window(self, seq: np.ndarray, k: int = 50) -> bool:
        if len(seq) < k or 0 in seq:
            return False
        h = self._hash(seq[:self.N])
        hs = self._to_i64(h)
        for (pos,) in self.cur.execute("SELECT pos FROM ng WHERE h=?", (hs,)):
            if np.array_equal(self.tokens[pos:pos + k], seq[:k]):
                return True
        return False


def membership_score(ppl_xl: float, zbytes: int, ntok: int) -> float:
    """Simple membership proxy: log(PPL) normalized by zlib bytes per token."""
    zpt = zbytes / max(ntok, 1)
    return np.log(max(ppl_xl, 1e-6)) / max(zpt, 1e-6)


def prepare_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok
