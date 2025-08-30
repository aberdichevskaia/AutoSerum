import os
import argparse
import json
import zlib
import numpy as np
import torch
from transformers import GPT2LMHeadModel
from verify_memorization import Ngram8Index, membership_score, prepare_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_perplexity(text, model, tokenizer):
    """exp(loss) for a single string."""
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.loss
    return float(torch.exp(loss).detach().cpu().item())


def membership_score(ppl_xl: float, z_bytes: int, ntok: int) -> float:
    """Lower is more suspicious: low XL perplexity relative to byte entropy per token."""
    import math
    zpt = (z_bytes / max(ntok, 1)) if ntok else 1.0
    return math.log(max(ppl_xl, 1.000001)) / max(zpt, 1e-6)


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

    args = ap.parse_args()

    print(f"[GEN] device: {device}", flush=True)
    cache_dir = "/home/iscb/wolfson/annab4/.cache/huggingface/transformers" #os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None

    tokenizer = prepare_tokenizer()
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
        m.eval()

    index = Ngram8Index(args.auxidx, n=8) if args.verify else None

    os.makedirs(args.outdir, exist_ok=True)
    from time import strftime
    run_dir = os.path.join(args.outdir, strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    f_all = open(os.path.join(run_dir, "samples.jsonl"), "w")
    f_flag = open(os.path.join(run_dir, "flagged.jsonl"), "w") if args.verify else None

    total = args.N
    step_quota = max(total // args.progress_steps, 1)
    next_tick = step_quota
    tick_idx = 1
    done = 0

    while done < total:
        bs = min(args.batch_size, total - done)
        prompts = ["<|endoftext|>"] * bs
        input_len = 1
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}

        with torch.no_grad():
            out = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + args.seq_len,
                do_sample=True, top_k=args.top_k, top_p=1.0,
                pad_token_id=PAD_ID,
            )
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)

        for text in texts:
            # 1) Metrics
            p1 = float(calculate_perplexity(text, model1, tokenizer))  # XL perplexity
            p2 = float(calculate_perplexity(text, model2, tokenizer))  # small perplexity
            ids = tokenizer.encode(text, add_special_tokens=False)
            z_bytes = len(zlib.compress(text.encode("utf-8")))
            score = membership_score(p1, z_bytes, len(ids))  # lower = more suspicious

            # 2) Record sample
            rec = {
                "text": text,
                "ppl_xl": p1,
                "ppl_s": p2,
                "z": int(z_bytes),
                "ntok": int(len(ids)),
                "score": float(score),
            }
            f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # 3) Optional membership verification (white-box)
            is_mem = False
            if args.verify and len(ids) >= args.window:
                # AUX tokens are +1-shifted (0 is a document sentinel), so shift our ids by +1
                ids_p1 = np.asarray([t + 1 for t in ids], dtype=np.uint32)

                # Slide a window and check index hits; confirm with exact token match inside the index helper
                for j in range(0, len(ids_p1) - args.window + 1):
                    win = ids_p1[j:j + args.window]
                    if 0 in win:  # skip windows crossing document boundaries
                        continue
                    if index.contains_window(win, k=args.window):
                        # Flag if either condition indicates likely memorization:
                        #   (a) very low XL perplexity, or (b) low membership score
                        if (p1 <= args.ppl_thr) or (score <= args.score_thr):
                            is_mem = True
                        break

            if is_mem:
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
