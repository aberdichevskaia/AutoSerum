import os
import argparse
import sqlite3
import numpy as np
import sys

# make src/ importable
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import PATHS, INDEX


def rhash(span):
    # FNV-1a 64-bit (unsigned math)
    h = 1469598103934665603
    for x in span:
        h ^= int(x)
        h *= 1099511628211
        h &= (1 << 64) - 1
    return h

def to_i64(u):
    # map uint64 -> int64 for SQLite INTEGER
    return int(u if u < (1 << 63) else u - (1 << 64))


def main():
    ap = argparse.ArgumentParser()
    # CLI overrides only; defaults come from config.py
    ap.add_argument("--auxidx", default=None, help="Directory with tokens.uint32 / doc_offsets.uint64")
    ap.add_argument("--ngram", type=int, default=None)
    ap.add_argument("--downsample", type=int, default=None, help="Index every k-th position (>=1)")
    ap.add_argument("--progress_steps", type=int, default=None)
    args = ap.parse_args()

    auxidx = args.auxidx if args.auxidx is not None else PATHS["auxidx_dir"]
    ngram = args.ngram if args.ngram is not None else INDEX["ngram"]
    downsample = args.downsample if args.downsample is not None else INDEX["downsample"]
    progress_steps = args.progress_steps if args.progress_steps is not None else INDEX["progress_steps"]

    if not os.path.isdir(auxidx):
        raise FileNotFoundError(f"auxidx dir not found: {auxidx}")

    tok_path = os.path.join(auxidx, "tokens.uint32")
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"tokens file not found: {tok_path}")

    # name DB by ngram (ng8.sqlite, ng16.sqlite, ...); keep ng8.sqlite when ngram==8
    db_name = f"ng{int(ngram)}.sqlite"
    db_path = os.path.join(auxidx, db_name)

    tokens = np.memmap(tok_path, dtype=np.uint32, mode="r")
    M = int(tokens.shape[0])
    N = int(max(1, ngram))
    step = int(max(1, downsample))

    # progress accounting
    total_iters = max((M - N + 1) // step, 1)
    step_quota = max(total_iters // max(1, int(progress_steps)), 1)
    next_tick = step_quota
    tick_idx = 1

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA page_size=65536;")
    cur.execute("DROP TABLE IF EXISTS ng")
    cur.execute("CREATE TABLE ng (h INTEGER, pos INTEGER)")
    con.commit()

    print(f"[IDX] start: dir={auxidx}, M={M}, N={N}, downsample={step}, db={db_name}", flush=True)
    buf, B = [], 200_000
    iters = 0

    for i in range(0, M - N + 1, step):
        span = tokens[i:i + N]
        if 0 in span:
            continue
        h = rhash(span)
        buf.append((to_i64(h), int(i)))
        iters += 1

        if len(buf) >= B:
            cur.executemany("INSERT INTO ng VALUES (?,?)", buf)
            con.commit()
            buf.clear()

        if iters >= next_tick and tick_idx <= progress_steps:
            pct = int(100 * min(iters, total_iters) / total_iters)
            print(f"[IDX] progress {tick_idx}/{progress_steps} (~{pct}%) — {iters}/{total_iters} inserts", flush=True)
            tick_idx += 1
            next_tick += step_quota

    if buf:
        cur.executemany("INSERT INTO ng VALUES (?,?)", buf)
        con.commit()

    # build index AFTER inserts (faster)
    print("[IDX] creating index on h...", flush=True)
    cur.execute("CREATE INDEX ng_idx ON ng(h)")
    con.commit()
    con.close()
    print(f"[IDX] done -> {db_path}", flush=True)


if __name__ == "__main__":
    main()
