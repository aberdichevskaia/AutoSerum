import os, sys, json, argparse, hashlib, statistics, math
from typing import Dict, List, Tuple
import numpy as np

# Local imports
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = FILE_DIR
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.verify_memorization import Ngram8Index, prepare_tokenizer

def load_jsonl(path: str) -> List[dict]:
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    return recs

def window_hash(tokens: List[int], j: int, k: int) -> str:
    # Hash of the k-token substring (token-level, not bytes)
    span = tokens[j:j+k]
    m = hashlib.sha1()
    m.update((",".join(map(str, span))).encode("utf-8"))
    return m.hexdigest()

def recompute_hits_if_needed(samples: List[dict], idx: Ngram8Index, k: int, tok) -> Tuple[int, set]:
    """If flagged.jsonl is absent, recompute hits by scanning windows."""
    hits = 0
    uniq = set()
    for rec in samples:
        ids = tok.encode(rec["text"], add_special_tokens=False)
        if len(ids) < k:
            continue
        ids_p1 = np.asarray([t + 1 for t in ids], dtype=np.uint32)
        hit = False
        for j in range(0, len(ids_p1) - k + 1):
            win = ids_p1[j:j + k]
            if 0 in win:
                continue
            if idx.contains_window(win, k=k):
                hit = True
                uniq.add(window_hash(ids, j, k))
                break
        if hit:
            hits += 1
    return hits, uniq

def summarize_run(label: str, run_dir: str, idx: Ngram8Index, k: int, tok, out_csv: List[str]) -> Dict:
    samples = load_jsonl(os.path.join(run_dir, "samples.jsonl"))
    flagged = load_jsonl(os.path.join(run_dir, "flagged.jsonl"))

    total = len(samples)
    if total == 0:
        print(f"[EVAL] {label}: no samples in {run_dir}")
        return {}

    # Totals
    tot_tokens = sum(int(rec.get("ntok", 0)) for rec in samples)
    ppl_vals = [float(rec.get("ppl_xl", float("nan"))) for rec in samples]
    score_vals = [float(rec.get("score", float("nan"))) for rec in samples]

    # Hits
    if flagged:
        hits = len(flagged)
        uniq = set()
        # Deduplicate by window content if needed (fallback to text hash)
        for rec in flagged:
            ids = tok.encode(rec["text"], add_special_tokens=False)
            if len(ids) >= k:
                # approximate: hash first matching k-span (we don't know j)
                uniq.add(window_hash(ids, 0, k))  # weak proxy
            else:
                uniq.add(hashlib.sha1(rec["text"].encode("utf-8")).hexdigest())
    else:
        hits, uniq = recompute_hits_if_needed(samples, idx, k, tok)

    hit_rate = hits / total
    hits_per_10k_tok = hits / (max(1, tot_tokens) / 10000.0)

    def safe_stat(arr, fn, default=float("nan")):
        arr2 = [x for x in arr if not math.isnan(x)]
        return fn(arr2) if arr2 else default

    ppl_med = safe_stat(ppl_vals, statistics.median)
    score_med = safe_stat(score_vals, statistics.median)

    print(f"[EVAL] {label}: total={total} hits={hits} hit_rate={hit_rate:.3%} "
          f"uniq_hits={len(uniq)} hits/10kTok={hits_per_10k_tok:.3f} "
          f"ppl_med={ppl_med:.2f} score_med={score_med:.3f}")

    out_csv.append(",".join(map(str, [
        label, run_dir, total, hits, f"{hit_rate:.5f}", len(uniq),
        f"{hits_per_10k_tok:.5f}", f"{ppl_med:.4f}", f"{score_med:.4f}"
    ])))
    return {
        "label": label, "total": total, "hits": hits, "hit_rate": hit_rate,
        "uniq_hits": len(uniq), "hits_per_10k_tok": hits_per_10k_tok,
        "ppl_med": ppl_med, "score_med": score_med
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auxidx", required=True, help="AUX index directory")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="List of label:path entries, e.g. baseline:/... rl:/... random:/...")
    ap.add_argument("--outcsv", default="eval_summary.csv")
    args = ap.parse_args()

    tok = prepare_tokenizer()
    idx = Ngram8Index(args.auxidx, n=8)

    csv_lines = ["label,run_dir,total,hits,hit_rate,uniq_hits,hits_per_10k_tok,ppl_med,score_med"]
    for item in args.runs:
        if ":" not in item:
            print(f"[EVAL] skip malformed run spec: {item}")
            continue
        label, path = item.split(":", 1)
        summarize_run(label, path, idx, args.window, tok, csv_lines)

    with open(args.outcsv, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"[EVAL] summary -> {os.path.abspath(args.outcsv)}")

if __name__ == "__main__":
    main()
