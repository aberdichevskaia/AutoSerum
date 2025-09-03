import os
import argparse
import json
import zlib
import numpy as np
import torch
from transformers import GPT2LMHeadModel

from verify_memorization import Ngram8Index, membership_score, prepare_tokenizer  # use imported score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_perplexity(text, model, tokenizer):
    """exp(loss) for a single string."""
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.loss
    return float(torch.exp(loss).detach().cpu().item())


def build_batch_with_suffix(prompts, suffix_ids, tokenizer, pad_id):
    """
    Tokenize prompts, append suffix_ids to each prompt as part of the context,
    and return left-/right-padded batch tensors (input_ids, attention_mask).
    """
    seqs = []
    for p in prompts:
        base = tokenizer.encode(p, add_special_tokens=False)
        full = base + list(suffix_ids)  # >>> suffix participates in context
        seqs.append(full)

    maxlen = max(len(s) for s in seqs) if seqs else 1
    left = (tokenizer.padding_side == "left")

    batch_ids = []
    batch_attn = []
    for s in seqs:
        pad_len = maxlen - len(s)
        if left:
            batch_ids.append(([pad_id] * pad_len) + s)
            batch_attn.append(([0] * pad_len) + ([1] * len(s)))
        else:
            batch_ids.append(s + ([pad_id] * pad_len))
            batch_attn.append(([1] * len(s)) + ([0] * pad_len))

    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    attention_mask = torch.tensor(batch_attn, dtype=torch.long)
    return input_ids, attention_mask, maxlen  # maxlen = context length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--auxidx", default=os.environ.get("AUXIDX_DIR", "data/auxidx"))
    ap.add_argument("--window", type=int, default=50)

    ap.add_argument("--membership-thr", type=float, default=1.3)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--progress_steps", type=int, default=10)
    ap.add_argument("--ppl-thr", type=float, default=25.0)
    ap.add_argument("--score-thr", type=float, default=1.0)
    ap.add_argument("--suffix-ids", type=str, default="",
                    help="Comma-separated GPT-2 token IDs to append as a suffix to the prompt")
    ap.add_argument("--suffix-file", type=str, default="",
                    help="Path to JSON with {'suffix': [ids,...]} produced by RL")
    args = ap.parse_args()

    print(f"[GEN] device: {device}", flush=True)
    cache_dir = "/home/iscb/wolfson/annab4/.cache/huggingface/transformers"

    tokenizer = prepare_tokenizer()  # must set pad_token=eos and padding_side='left' inside
    PAD_ID = tokenizer.pad_token_id

    model1 = GPT2LMHeadModel.from_pretrained(
        'gpt2-xl',
        return_dict=True,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    model2 = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        return_dict=True,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    for m in (model1, model2):
        m.config.pad_token_id = PAD_ID
    model1.eval()
    model2.eval()

    index = Ngram8Index(args.auxidx, n=8) if args.verify else None

    os.makedirs(args.outdir, exist_ok=True)
    from time import strftime
    run_dir = os.path.join(args.outdir, strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    f_all = open(os.path.join(run_dir, "samples.jsonl"), "w", encoding="utf-8")
    f_flag = open(os.path.join(run_dir, "flagged.jsonl"), "w", encoding="utf-8") if args.verify else None

    # >>> load suffix once
    suffix_ids = []
    if args.suffix_file:
        with open(args.suffix_file, "r", encoding="utf-8") as f:
            suffix_ids = list(map(int, json.load(f).get("suffix", [])))
    elif args.suffix_ids:
        suffix_ids = [int(x) for x in args.suffix_ids.split(",") if x.strip()]

    total = args.N
    step_quota = max(total // args.progress_steps, 1)
    next_tick = step_quota
    tick_idx = 1
    done = 0

    while done < total:
        bs = min(args.batch_size, total - done)
        prompts = ["<|endoftext|>"] * bs  # base prompt

        # >>> build inputs with suffix included in the context
        input_ids, attention_mask, ctx_len = build_batch_with_suffix(
            prompts, suffix_ids, tokenizer, PAD_ID
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate
        with torch.no_grad():
            out = model1.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=ctx_len + args.seq_len,  # >>> total = context + new tokens
                do_sample=True, top_k=args.top_k, top_p=1.0,
                pad_token_id=PAD_ID,
            )

        # >>> for each item: slice off the context and score ONLY the generated continuation
        for b in range(bs):
            seq = out[b]
            # context length for this example = number of 1s in attention mask row
            this_ctx_len = int(attention_mask[b].sum().item())
            gen_ids = seq[this_ctx_len:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Metrics on the generated continuation only
            p1 = calculate_perplexity(gen_text, model1, tokenizer)  # XL PPL
            p2 = calculate_perplexity(gen_text, model2, tokenizer)  # small PPL
            ids = tokenizer.encode(gen_text, add_special_tokens=False)  # >>> NO suffix here
            z_bytes = len(zlib.compress(gen_text.encode("utf-8")))
            score = membership_score(p1, z_bytes, len(ids))  # using imported score

            rec = {
                "text": gen_text,
                "ppl_xl": float(p1),
                "ppl_s": float(p2),
                "z": int(z_bytes),
                "ntok": int(len(ids)),
                "score": float(score),
                "suffix_ids": suffix_ids,  # for traceability
            }
            f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # White-box verification (index on AUX)
            is_mem = False
            if args.verify and len(ids) >= args.window:
                ids_p1 = np.asarray([t + 1 for t in ids], dtype=np.uint32)  # AUX shift (+1)
                for j in range(0, len(ids_p1) - args.window + 1):
                    win = ids_p1[j:j + args.window]
                    if 0 in win:
                        continue
                    if index.contains_window(win, k=args.window):
                        if (p1 <= args.ppl_thr) or (score <= args.score_thr):
                            is_mem = True
                        break

            if is_mem and f_flag is not None:
                f_flag.write(json.dumps(rec, ensure_ascii=False) + "\n")

        done += bs
        if done >= next_tick and tick_idx <= args.progress_steps:
            pct = int(100 * min(done, total) / total)
            print(f"[GEN] progress {tick_idx}/{args.progress_steps} (~{pct}%) — {done}/{total}", flush=True)
            tick_idx += 1
            next_tick += step_quota

    f_all.close()
    if f_flag:
        f_flag.close()
    print(f"[GEN] done. artifacts -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
