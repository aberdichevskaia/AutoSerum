# src/sample_gpt2.py
import os
import argparse
import json
import zlib
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# resolving local pathes
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # AutoSerum/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, GEN
from src.extraction.verify_memorization import Ngram8Index, membership_score

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
        full = base + list(suffix_ids)  # suffix participates in context
        seqs.append(full)

    maxlen = max(len(s) for s in seqs) if seqs else 1
    left = (tokenizer.padding_side == "left")

    batch_ids, batch_attn = [], []
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


def parse_args():
    ap = argparse.ArgumentParser()

    # generation / scoring
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--prompt", type=str, default=None)

    # models (now overridable; defaults come from GEN)
    ap.add_argument("--main-lm", type=str, default=None, help="HF id for main/gen model (e.g., gpt2-xl)")
    ap.add_argument("--ref-lm", type=str, default=None, help="HF id for reference/small model (e.g., gpt2)")

    # verify / aux index
    ap.add_argument("--verify", dest="verify", action="store_true")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.set_defaults(verify=None)
    ap.add_argument("--auxidx", type=str, default=None)
    ap.add_argument("--window", type=int, default=None)

    # thresholds / progress
    ap.add_argument("--membership-thr", type=float, default=None)
    ap.add_argument("--progress-steps", type=int, default=None)
    ap.add_argument("--ppl-thr", type=float, default=None)
    ap.add_argument("--score-thr", type=float, default=None)

    # outputs / paths
    ap.add_argument("--runs-dir", type=str, default=None, help="Base runs directory (default from PATHS.runs_dir)")
    ap.add_argument("--out-subdir", type=str, default=None, help="Subdir under runs for this job (default GEN.out_subdir)")
    ap.add_argument("--outdir", type=str, default=None, help="(Deprecated) full base dir for this run; overrides --runs-dir/--out-subdir")
    ap.add_argument("--hf-cache", type=str, default=None, help="HF cache dir (default from PATHS.hf_home)")

    # suffix controls
    ap.add_argument("--suffix-ids", type=str, default="", help="Comma-separated GPT-2 token IDs to append as a suffix to the prompt")
    ap.add_argument("--suffix-file", type=str, default="", help="Path to JSON with {'suffix': [ids,...]} produced by RL")

    return ap.parse_args()


def main():
    # ---- defaults from config.py ----
    paths = dict(PATHS)
    gen = dict(GEN)

    # sensible fallbacks if not present in GEN
    gen.setdefault("main_lm", "gpt2-xl")
    gen.setdefault("ref_lm", "gpt2")
    gen.setdefault("top_p", 1.0)
    gen.setdefault("prompt", "<|endoftext|>")
    gen.setdefault("progress_steps", 10) 

    # ---- CLI overrides ----
    args = parse_args()
    # numeric / bools
    if args.N is not None: gen["N"] = args.N
    if args.batch_size is not None: gen["batch_size"] = args.batch_size
    if args.seq_len is not None: gen["seq_len"] = args.seq_len
    if args.top_k is not None: gen["top_k"] = args.top_k
    if args.top_p is not None: gen["top_p"] = args.top_p
    if args.verify is not None: gen["verify"] = bool(args.verify)
    if args.window is not None: gen["aux_window"] = args.window
    if args.membership_thr is not None: gen["membership_thr"] = args.membership_thr
    if args.progress_steps is not None: gen["progress_steps"] = args.progress_steps
    if args.ppl_thr is not None: gen["ppl_thr"] = args.ppl_thr
    if args.score_thr is not None: gen["score_thr"] = args.score_thr
    # strings
    if args.main_lm is not None: gen["main_lm"] = args.main_lm
    if args.ref_lm is not None: gen["ref_lm"] = args.ref_lm
    if args.prompt is not None: gen["prompt"] = args.prompt
    if args.auxidx is not None: paths["auxidx_dir"] = args.auxidx
    if args.runs_dir is not None: paths["runs_dir"] = args.runs_dir
    if args.out_subdir is not None: gen["out_subdir"] = args.out_subdir
    if args.hf_cache is not None: paths["hf_home"] = args.hf_cache

    # ---- resolved parameters ----
    total = int(gen["N"])
    batch_size = int(gen["batch_size"])
    seq_len = int(gen["seq_len"])
    top_k = int(gen["top_k"])
    top_p = float(gen["top_p"])
    prompt_text = str(gen["prompt"])
    verify = bool(gen["verify"])
    auxidx_dir = str(paths["auxidx_dir"])
    window_primary = int(gen["aux_window"])
    ppl_thr = float(gen["ppl_thr"])
    score_thr = float(gen["score_thr"])
    progress_steps = int(gen["progress_steps"])
    hf_cache_dir = str(paths["hf_home"])
    main_lm_id = str(gen["main_lm"])
    ref_lm_id = str(gen["ref_lm"])

    # ---- tokenizer ----
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=hf_cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    PAD_ID = tokenizer.pad_token_id

    print(f"[GEN] device: {device} | main_lm={main_lm_id} | ref_lm={ref_lm_id}", flush=True)

    # ---- models ----
    model1 = GPT2LMHeadModel.from_pretrained(
        main_lm_id,
        return_dict=True,
        cache_dir=hf_cache_dir,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
        low_cpu_mem_usage=True
    ).to(device)

    model2 = GPT2LMHeadModel.from_pretrained(
        ref_lm_id,
        return_dict=True,
        cache_dir=hf_cache_dir,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
        low_cpu_mem_usage=True
    ).to(device)

    for m in (model1, model2):
        m.config.pad_token_id = PAD_ID
    model1.eval(); model2.eval()

    index = Ngram8Index(auxidx_dir, n=8) if verify else None

    # ---- outputs ----
    from time import strftime
    base_out = paths["runs_dir"]
    out_subdir = gen["out_subdir"]
    run_dir = os.path.join(base_out, out_subdir, strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    f_all = open(os.path.join(run_dir, "samples.jsonl"), "w", encoding="utf-8")
    f_flag = open(os.path.join(run_dir, "flagged.jsonl"), "w", encoding="utf-8") if verify else None

    # ---- suffix loading ----
    suffix_ids = []
    if args.suffix_file:
        with open(args.suffix_file, "r", encoding="utf-8") as f:
            suffix_ids = list(map(int, json.load(f).get("suffix", [])))
    elif args.suffix_ids:
        suffix_ids = [int(x) for x in args.suffix_ids.split(",") if x.strip()]

    # ---- progress accounting ----
    total_needed = total
    step_quota = max(total_needed // progress_steps, 1)
    next_tick = step_quota
    tick_idx = 1
    done = 0

    # report windows (always include the primary window)
    WIN_SET = sorted({8, 16, 32, int(window_primary)})

    # ---- main loop ----
    while done < total_needed:
        bs = min(batch_size, total_needed - done)
        prompts = [prompt_text] * bs  # base prompt (from config/CLI)

        # build inputs with suffix included in the context
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
                max_length=ctx_len + seq_len,  # total = context + new tokens
                do_sample=True, top_k=top_k, top_p=top_p,
                pad_token_id=PAD_ID,
            )

        # for each item: slice off the context and score ONLY the generated continuation
        for b in range(bs):
            seq = out[b]
            # context length for this example = number of 1s in attention mask row
            this_ctx_len = int(attention_mask[b].sum().item())
            gen_ids = seq[this_ctx_len:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Metrics on the generated continuation only
            p1 = calculate_perplexity(gen_text, model1, tokenizer)  # main model PPL
            p2 = calculate_perplexity(gen_text, model2, tokenizer)  # ref model PPL
            ids = tokenizer.encode(gen_text, add_special_tokens=False)  # NO suffix here
            z_bytes = len(zlib.compress(gen_text.encode("utf-8")))
            score = membership_score(p1, z_bytes, len(ids))  # using imported score

            # Default hits stats (if no verification)
            hits_total = 0
            hits_by_window = {}

            # White-box verification (index on AUX)
            is_mem = False
            if verify and index is not None and len(ids) >= min(WIN_SET):
                # AUX shift (+1), skip windows that include 0 (doc boundary)
                ids_p1 = np.asarray([t + 1 for t in ids], dtype=np.uint32)

                # Count matches for multiple window sizes
                for k in WIN_SET:
                    if len(ids_p1) < k:
                        continue
                    cnt_k = 0
                    for j in range(0, len(ids_p1) - k + 1):
                        win = ids_p1[j:j + k]
                        if 0 in win:
                            continue
                        if index.contains_window(win, k=k):
                            cnt_k += 1
                    hits_by_window[str(k)] = int(cnt_k)

                # Primary window stats / decision
                hits_total = int(hits_by_window.get(str(int(window_primary)), 0))
                if hits_total > 0 and ((p1 <= ppl_thr) or (score <= score_thr)):
                    is_mem = True

            rec = {
                "text": gen_text,
                "ppl_main": float(p1),
                "ppl_ref": float(p2),
                "z": int(z_bytes),
                "ntok": int(len(ids)),
                "score": float(score),
                "suffix_ids": suffix_ids,  # for traceability
                "hits_total": int(hits_total),
                "hits_by_window": hits_by_window,  # e.g., {"8": n8, "16": n16, ...}
            }
            f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if is_mem and f_flag is not None:
                f_flag.write(json.dumps(rec, ensure_ascii=False) + "\n")

        done += bs
        if done >= next_tick and tick_idx <= progress_steps:
            pct = int(100 * min(done, total_needed) / total_needed)
            print(f"[GEN] progress {tick_idx}/{progress_steps} (~{pct}%) — {done}/{total_needed}", flush=True)
            tick_idx += 1
            next_tick += step_quota

    f_all.close()
    if f_flag:
        f_flag.close()
    print(f"[GEN] done. artifacts -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
