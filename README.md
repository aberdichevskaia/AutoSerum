# AutoSerum — Minimal pipeline for memorization extraction from LLMs

AutoSerum is a small, modular project for **discovering memorized text** in autoregressive LMs (starting with GPT-2 / GPT-2-XL). It includes:

* An **AUX dataset builder** (from public web corpora) and an **n-gram index** for fast white-box substring checks.
* A **generation + filtering** step that samples from a model, scores candidates with a heuristic **membership score**, and optionally **verifies** them via the index.
* A lightweight **RL loop** that learns **prompt suffixes** to increase the odds of extracting memorized continuations.

The repo is intentionally student-scale: you can run end-to-end on a single GPU plus a few GB of AUX data, then scale up if useful.

---

## Repository layout

```
AutoSerum/
├─ README.md
├─ data/
├─ runs/
├─ src/
│  ├─ config.py                     # single source of truth: paths & defaults
│  ├─ _bootstrap.py                 # optional: stable imports when running files directly
│  ├─ datasets_building/
│  │  ├─ build_aux_tokens.py        # stream → tokenize (GPT-2) → tokens.uint32, doc_offsets.uint64
│  │  └─ build_ngram8_index.py      # build n=8 SQLite index over token windows
│  ├─ extraction/
│  │  ├─ sample_gpt2.py             # generation + scoring + (opt.) verification
│  │  └─ verify_memorization.py     # Ngram8Index, membership_score, tokenizer helper
│  └─ RL/
│     ├─ train_mem_prompt.py        # RL to learn a K-token suffix
│     ├─ policy.py
│     ├─ env_mem.py
│     └─ reward.py
└─ slurm/ (optional)
```

> If you don’t use `python -m`, keep `_bootstrap.py` and add the small header shown in **Running** to make imports stable.

---

## Environment

* Python ≥ 3.10 (tested with 3.12)
* CUDA GPU recommended (≥12 GB for `gpt2-xl`; 24 GB is comfortable)
* Packages (working combo):

  * `torch` 2.8.x (CUDA build matching your driver)
  * `transformers` 4.55.x
  * `datasets` 2.21.x
  * `zstandard`
  * `numpy`, `tqdm`

Example (adjust CUDA wheel channel/version as needed):

```bash
conda create -n auto_serum python=3.12 --override-channels -c conda-forge -c defaults -y
conda activate auto_serum

# for generation 
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 # adjust CUDA as needed  
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 
pip install transformers==4.55.4 tqdm==4.67.1 numpy==2.0.2 tokenizers==0.21.4 huggingface_hub==0.34.4 safetensors==0.6.2

# for dataset building
pip install --no-cache-dir "datasets==2.21.0"
pip install --no-cache-dir zstandard
pip install --no-cache-dir "fsspec[http]==2024.6.1" "aiohttp>=3.8,<4"
```

---

## Configuration (single source of truth)

All defaults live in `src/config.py`:

* `PATHS`: `hf_home`, `auxidx_dir`, `runs_dir`, `corpus`
* `BUILD_AUX`: dataset, split, `max_tokens`, `progress_steps`
* `INDEX`: `ngram`, `downsample`, `progress_steps`
* `TRAIN`: RL/training parameters (models, K, batch, iters, etc.)
* `GEN`: generation/verification parameters (e.g., `main_lm`, `ref_lm`, `prompt`, `top_k`, `top_p`, `out_subdir`, thresholds)

You can override **any** of these via CLI flags at run time.
Hugging Face cache is passed as `cache_dir=PATHS["hf_home"]`.

Optionally, if you need gated repos, run:
```bash
# export HF_TOKEN=hf_********************************  
```
---

## Building the AUX dataset + index

### 1) Tokens

Default (uses `PATHS` / `BUILD_AUX`):

```bash
python -u src/datasets_building/build_aux_tokens.py
```

Optional overrides:

```bash
python -u -m src.datasets_building.build_aux_tokens \
  --out /path/to/auxidx \
  --dataset cerebras/SlimPajama-627B \
  --max_tokens 10000000
```

Outputs:

* `auxidx/tokens.uint32`
* `auxidx/doc_offsets.uint64`

### 2) N-gram (n=8) index

```bash
python -u src/datasets_building/build_ngram8_index.py
```

Output:

* `auxidx/ng8.sqlite`

---

## Generation + (optional) verification

What it does:

* Samples with **GEN.main_lm** (default `gpt2-xl`), optional **suffix**.
* Computes PPL under main and ref models, zlib size, membership score.
* If `verify=True`, slides a token window and checks presence in the AUX index.

Default run (everything from `config.py`):

```bash
python -u src/extraction/sample_gpt2.py
```

With a learned suffix:

```bash
python -u src/extraction/sample_gpt2.py \
  --suffix-file /path/to/runs/memrl/best_suffix.json
```

Selective overrides:

```bash
python -u src/extraction/sample_gpt2.py \
  --verify --window 32 --main-lm gpt2-xl --ref-lm gpt2 --N 200
```

Outputs in `runs/<GEN.out_subdir>/<timestamp>/`:

* `samples.jsonl` — all samples with metrics
* `flagged.jsonl` — samples passing thresholds **and** verified by index hits

Implementation notes:

* Tokenizer uses **left padding** with `pad_token=eos`.
* Models use `torch_dtype=float16` on CUDA and `low_cpu_mem_usage=True`.

---

## RL: learn a prompt suffix

The RL loop learns **K tokens** to append to a base prefix aiming to increase a reward correlated with memorization.

Reward (see `src/RL/reward.py`):

* **Proxy**: shaped function of PPL, zlib bytes, ntokens.
* **Verification**: multi-scale index hits (heavier weight for larger windows).
* Final reward combines both (bounded proxy + log-scaled hits).

Train:

```bash
python -u src/RL/train_mem_prompt.py
```

Override examples:

```bash
python -u src/RL/train_mem_prompt.py --iters 100 --batch-size 4
```

Outputs in `runs/<TRAIN.out_subdir>/`:

* `train_log.jsonl`
* `best.json` + `best_suffix.json`
* periodic `suffix_iter_*.json`

A/B testing the suffix:

```bash
# learned
python -u src/extraction/sample_gpt2.py --suffix-file runs/memrl/best_suffix.json

# random (same length)
python -u src/extraction/sample_gpt2.py --suffix-ids 128,42,17,199
```

---

## SLURM examples

Minimal sampling + verification (everything from `config.py`, only suffix is specified):

```bash
#!/bin/bash
#SBATCH --job-name=sample_and_verify
#SBATCH --output=output_sample_and_verify.txt
#SBATCH --error=error_sample_and_verify.txt
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500000
#SBATCH --time=1440
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --mail-user=you@domain

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate auto_serum

python -u src/extraction/sample_gpt2.py \
  --suffix-file /path/to/runs/memrl/best_suffix.json
```
---

## How the pieces fit

1. `build_aux_tokens.py` — stream dataset → GPT-2 tokenize → write `tokens.uint32` (+ `doc_offsets.uint64`).
2. `build_ngram8_index.py` — slide 8-gram window → insert `(hash, pos)` into SQLite with an index.
3. `sample_gpt2.py` — generate → compute metrics/score → (opt.) verify via index → write JSONL.
4. `RL` — learn suffix that increases proxy + hits; save the best suffix JSON.

---

## Limitations

* AUX is a **proxy** for true pretraining data → results are **lower bounds**.
* Index matches **exact** GPT-2 token windows (no paraphrase fuzziness).
* RL is a minimal REINFORCE; you can extend it (entropy schedules, better baselines, etc.).

---

## References (selection)

* Carlini et al., *Extracting Training Data from Large Language Models*, USENIX Security 2021.
* Somekh et al., *Scalable Extraction of Training Data from (Production) Language Models*, 2023.
* D. Shin et al., *RLPROMPT: Optimizing Discrete Text Prompts with Reinforcement Learning*, 2020.
* **SlimPajama** dataset.

---

## Acknowledgments

This project builds on public datasets and prior memorization work. The implementation is minimal to keep experimentation feasible on university hardware.
