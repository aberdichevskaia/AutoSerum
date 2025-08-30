import os
import argparse
import sqlite3
import numpy as np


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
    ap.add_argument("--auxidx", required=True, help="Dir with tokens.uint32")
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=1, help="Index every k-th position")
    ap.add_argument("--progress_steps", type=int, default=10)
    args = ap.parse_args()

    tok_path = os.path.join(args.auxidx, "tokens.uint32")
    db_path = os.path.join(args.auxidx, "ng8.sqlite")

    tokens = np.memmap(tok_path, dtype=np.uint32, mode="r")
    M = tokens.shape[0]
    N = args.ngram
    step = max(args.downsample, 1)

    # progress accounting based on loop iterations
    total_iters = max((M - N + 1) // step, 1)
    step_quota = max(total_iters // args.progress_steps, 1)
    next_tick = step_quota
    tick_idx = 1

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA page_size=65536;")
    cur.execute("DROP TABLE IF EXISTS ng")  # optional if recreating
    cur.execute("CREATE TABLE ng (h INTEGER, pos INTEGER)")
    con.commit()

    print(f"[IDX] start: M={M}, N={N}, downsample={step}", flush=True)
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

        if iters >= next_tick and tick_idx <= args.progress_steps:
            pct = int(100 * min(iters, total_iters) / total_iters)
            print(f"[IDX] progress {tick_idx}/{args.progress_steps} (~{pct}%) — {iters}/{total_iters} inserts", flush=True)
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
