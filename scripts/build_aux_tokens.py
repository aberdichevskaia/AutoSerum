import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for aux index (tokens/offsets)")
    ap.add_argument("--dataset", default="togethercomputer/RedPajama-Data-1T-Sample")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_tokens", type=int, default=50_000_000, help="Stop after this many tokens (+sentinels)")
    ap.add_argument("--progress_steps", type=int, default=10, help="How many progress ticks to print")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok_path = os.path.join(args.out, "tokens.uint32")
    off_path = os.path.join(args.out, "doc_offsets.uint64")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        trust_remote_code=True, 
    )


    total = 0
    offsets = [0]
    goal = max(args.max_tokens, 1)
    step_quota = max(goal // args.progress_steps, 1)
    next_tick = step_quota
    tick_idx = 1

    print(f"[AUX] start: max_tokens={args.max_tokens}, dataset={args.dataset}", flush=True)
    with open(tok_path, "wb") as f:
        for ex in ds:
            text = ex.get("text") or ""
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            # +1 to keep 0 reserved as a document sentinel
            arr = np.asarray([i + 1 for i in ids] + [0], dtype=np.uint32)
            arr.tofile(f)
            total += arr.size
            offsets.append(total)

            if total >= next_tick and tick_idx <= args.progress_steps:
                pct = int(100 * min(total, goal) / goal)
                print(f"[AUX] progress {tick_idx}/{args.progress_steps} (~{pct}%) — {total} tokens", flush=True)
                tick_idx += 1
                next_tick += step_quota

            if total >= args.max_tokens:
                break

    np.asarray(offsets, dtype=np.uint64).tofile(off_path)
    print(f"[AUX] done. total tokens (with sentinels): {total}", flush=True)
    print(f"[AUX] tokens -> {tok_path}", flush=True)
    print(f"[AUX] offsets -> {off_path}", flush=True)


if __name__ == "__main__":
    main()
