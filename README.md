# AutoSerum — Minimal pipeline for memorization extraction from GPT-2

AutoSerum is a small, modular project for **discovering memorized text** in autoregressive language models (starting with GPT-2 and GPT-2-XL). It includes:

* A **toy AUX dataset** builder (from public web corpora) and an **n-gram index** for fast white-box substring lookups.
* A **generation + filtering** script that samples from a model, ranks candidates by a heuristic “membership score”, and (optionally) **verifies** them via the index.
* A lightweight **RL loop** that learns **prompt suffixes** which increase the odds of extracting memorized continuations.

This repo is intentionally student-project scale: you can run end-to-end on a single GPU + a few GB of AUX data, then scale up later if useful.

---

## Repository layout

```
AutoSerum/
├─ README.md
├─ scripts/
│  ├─ build_aux_tokens.py
│  ├─ build_ngram8_index.py
│  ├─ build_small_corpus.py
├─ src/
│  ├─ sample_gpt2.py
│  ├─ verify_memorization.py
│  └─ RL/
│     ├─ train_mem_prompt.py
│     ├─ policy.py
│     ├─ env_mem.py
│     └─ reward.py
├─ data/
├─ runs/
└─ slurm/
```

---

## Quick start

### 1) Environment (tested)

* Python ≥ 3.10 (tested with 3.12)
* CUDA GPU recommended (≥12 GB for `gpt2-xl`; 24 GB is comfortable)
* Packages (working combo):

  * `torch` 2.8.x (CUDA build matching your driver)
  * `transformers` 4.55.x
  * `datasets` 2.21.x
  * `zstandard`
  * `numpy`, `tqdm`

Example:

```bash
conda create -n carlini python=3.12 -y
conda activate carlini
pip install torch --index-url https://download.pytorch.org/whl/cu124    # choose the correct CUDA build
pip install transformers==4.55.4 datasets==2.21.0 zstandard numpy tqdm
```

### 2) Hugging Face cache & (optional) token

```bash
export HF_HOME=/path/to/.cache/huggingface
# export HF_TOKEN=hf_********************************   # if you need gated repos
```

### 3) Build a small AUX dataset + index

Tokenize a small subset (e.g., SlimPajama), writing a flat token file:

```bash
python scripts/build_aux_tokens.py \
  --dataset cerebras/SlimPajama-627B \
  --split train \
  --max-docs 20000 \
  --outdir /path/to/autoserum/auxidx
# → /path/to/autoserum/auxidx/tokens.uint32
```

Build an **n-gram (n=8) SQLite** index:

```bash
python scripts/build_ngram8_index.py \
  --tokens /path/to/autoserum/auxidx/tokens.uint32 \
  --out /path/to/autoserum/auxidx/ng8.sqlite
```

Set an env var for convenience:

```bash
export AUXIDX_DIR=/path/to/autoserum/auxidx
```

> The paper’s full pipeline uses \~9 TB and weeks of compute. Start tiny to validate; later increase `--max-docs`.

---

## Generation + (optional) verification

### What it does

* Samples from **GPT-2-XL** (or your model), optionally with a **prompt suffix**, and computes:

  * **Perplexity** under XL and small GPT-2,
  * **Zlib byte length**,
  * A **membership score** (heuristic).
* If `--verify` is set, slides a **window** over output tokens and checks whether any window appears in the **AUX n-gram index**.

### Run

```bash
python src/sample_gpt2.py \
  --N 1000 \
  --batch-size 1 \
  --seq-len 256 \
  --top-k 40 \
  --verify \
  --auxidx "$AUXIDX_DIR" \
  --window 32 \
  --ppl-thr 15 \
  --score-thr 1.0 \
  --membership-thr 1.3 \
  --progress_steps 10 \
  --outdir runs/gen
```

**Outputs** (`runs/gen/<timestamp>`):

* `samples.jsonl` — all samples with metrics.
* `flagged.jsonl` — samples that pass heuristics **and** had an index hit in a sliding window.

**Notes**

* Tokenizer uses **left padding** with `pad_token=eos`.
* Use `torch_dtype=float16` and `low_cpu_mem_usage=True` for `gpt2-xl`.
* Learned suffix can be passed via `--suffix-file path/to/best.json`.

---

## RL: learn a prompt suffix that helps extraction

The RL loop learns **K tokens** to append to a base prefix, aiming to increase a reward correlated with memorization.

### Reward (shaped)

* **Proxy**: function of XL perplexity, zlib byte length, token count.
* **Verification**: number of **index hits** of n-gram windows in the generated continuation.
* **Final**: `R = w_proxy * proxy + w_hits * hits` (see `src/RL/reward.py`).

### Train

```bash
python src/RL/train_mem_prompt.py
```

Configuration lives in the `CFG` dataclass (models, K, batch, iters…). Adjust in code or extend to CLI args.

**Outputs** (`runs/memrl`):

* `train_log.jsonl` — per-iteration mean/max reward, loss, baseline, debug.
* `best.json` — best suffix token IDs and diagnostics.

### A/B test the learned suffix

```bash
# learned
python src/sample_gpt2.py ... \
  --suffix-file runs/memrl/best.json \
  --outdir runs/ab/learned

# random with same length
python src/sample_gpt2.py ... \
  --suffix-ids 128,42,17,199 \
  --outdir runs/ab/random
```

Compare `flagged.jsonl` counts and aggregates in `samples.jsonl`.

---

## Slurm example

```bash
#!/bin/bash
#SBATCH --job-name=sample_and_verify
#SBATCH --output=output_sample_and_verify.txt
#SBATCH --error=error_sample_and_verify.txt
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=24:00:00

set -euo pipefail
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate carlini

export HF_HOME=/path/to/.cache/huggingface
export AUXIDX_DIR=/path/to/autoserum/auxidx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /path/to/AutoSerum/src/sample_gpt2.py \
  --N 1000 \
  --batch-size 1 \
  --seq-len 256 \
  --top-k 40 \
  --verify \
  --auxidx "$AUXIDX_DIR" \
  --window 32 \
  --ppl-thr 15 \
  --score-thr 1.0 \
  --membership-thr 1.3 \
  --progress_steps 10
```

---

## How the pieces fit

1. `build_aux_tokens.py` — stream dataset → GPT-2 tokenize → write `tokens.uint32`.
2. `build_ngram8_index.py` — slide 8-gram window → insert into SQLite with index.
3. `sample_gpt2.py` — generate → compute metrics/score → optionally verify via index → write JSONL.
4. **RL** — learn suffix that increases proxy + hits; save `best.json`.

---

## Configuration & paths

* `HF_HOME` — Hugging Face cache root (prefer over deprecated `TRANSFORMERS_CACHE`).
* `AUXIDX_DIR` — directory containing `tokens.uint32` and `ng8.sqlite`.
* `TS_CORPUS` — optional local text file for RL slicing (fallback text is built-in).
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — helps with CUDA fragmentation.

---

## Troubleshooting

* **Right-padding warning** (GPT-2): ensure `padding_side='left'` and `pad_token=eos`.
* **OOM (`gpt2-xl`)**: use fp16, lower batch/seq-len, free the GPU.
* **`Can't initialize NVML`**: benign on some clusters; PyTorch still runs.
* **Streaming/zstd**: `pip install zstandard`; keep `datasets` + `fsspec` compatible.
* **`Not a directory: .../ng8.sqlite/tokens.uint32`**: pass the **directory** path to `--auxidx`, not the sqlite file.
* **Slurm `unexpected EOF`**: unmatched quotes or a comment after a line with `\`. Run `bash -n` and/or `dos2unix`.

---

## Scaling up

* Increase `--max-docs` in `build_aux_tokens.py` and rebuild the index.
* Consider sharding the index for very large corpora (this project uses a lighter SQLite n-gram approximation).

---

## Limitations

* AUX dataset is a **proxy** for true pretraining data → results are **lower bounds**.
* Index matches **exact** GPT-2 token windows; no paraphrase fuzziness.
* RL is a **minimal REINFORCE**; you can add entropy bonus, temperature schedules, larger slicing corpora, etc.

---

## Citations (background)

* Carlini et al., “Extracting Training Data from Large Language Models” (USENIX Security ’21).
* “Scalable Extraction of Training Data from (Production) Language Models.”
* SlimPajama / RedPajama / The Pile / RefinedWeb / Dolma corpora.

---

## Repro tips

* Log versions: `python -V`, `torch.__version__`, `transformers.__version__`, `datasets.__version__`.
* Each run writes to `runs/<timestamp>`.
* Seed: `CFG.seed` in RL (and optionally `torch.cuda.manual_seed_all`).

---

## Acknowledgments

This project builds on public datasets and prior memorization work. The implementation is minimal to keep experimentation feasible on university hardware.
