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
* CUDA GPU recommended (≥ 12 GB for `gpt2-xl`; 24 GB is comfortable)
* Packages (working set):

  * `torch` 2.x (CUDA build for your driver)
  * `transformers` 4.55.x
  * `datasets` 2.21.x
  * `numpy`, `tqdm`, `safetensors`, `huggingface_hub`, `tokenizers`
  * streaming/compression: `zstandard`, `fsspec[http]`, `aiohttp`

Example (adjust CUDA wheel channel/version as needed):

```bash
conda create -n auto_serum python=3.12 -y
conda activate auto_serum

pip install --upgrade pip setuptools wheel
# pick the correct CUDA URL/version for your system:
# pip install --index-url https://download.pytorch.org/whl/cu124
# pip install torch torchvision torchaudio

pip install "transformers==4.55.*" "datasets==2.21.*" tqdm numpy tokenizers \
            huggingface_hub safetensors zstandard "fsspec[http]" "aiohttp>=3.8,<4"
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
We **don’t** rely on environment variables; Hugging Face cache is passed as `cache_dir=PATHS["hf_home"]`.

---

## Building the AUX dataset + index

### 1) Tokens

Default (uses `PATHS` / `BUILD_AUX`):

```bash
python -u -m src.datasets_building.build_aux_tokens
# or:
# python -u src/datasets_building/build_aux_tokens.py
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
python -u -m src.datasets_building.build_ngram8_index
# or: python -u src/datasets_building/build_ngram8_index.py
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
python -u -m src.extraction.sample_gpt2
```

With a learned suffix:

```bash
python -u -m src.extraction.sample_gpt2 \
  --suffix-file /path/to/runs/memrl/best_suffix.json
```

Selective overrides:

```bash
python -u -m src.extraction.sample_gpt2 \
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
python -u -m src.RL.train_mem_prompt
```

Override examples:

```bash
python -u -m src.RL.train_mem_prompt --iters 100 --batch-size 4
```

Outputs in `runs/<TRAIN.out_subdir>/`:

* `train_log.jsonl`
* `best.json` + `best_suffix.json`
* periodic `suffix_iter_*.json`

A/B testing the suffix:

```bash
# learned
python -u -m src.extraction.sample_gpt2 --suffix-file runs/memrl/best_suffix.json

# random (same length)
python -u -m src.extraction.sample_gpt2 --suffix-ids 128,42,17,199
```

---

## Running (imports that always work)

If you **don’t** use packages / `python -m`, keep `_bootstrap.py` and add this header at the **top** of each entry-point you run directly (e.g., `sample_gpt2.py`, `train_mem_prompt.py`, dataset builders):

```python
# --- bootstrap (stable imports when running as a file) ---
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # add .../src
from _bootstrap import *  # noqa: F401,F403
# ---------------------------------------------------------
```

Then use absolute imports everywhere inside the repo, e.g.:

```python
from src.config import PATHS, GEN
from src.extraction.verify_memorization import Ngram8Index
from src.RL.policy import SuffixPolicy
```

**Alternative**: run as modules (no bootstrap needed):

```bash
python -m src.extraction.sample_gpt2
python -m src.RL.train_mem_prompt
python -m src.datasets_building.build_aux_tokens
python -m src.datasets_building.build_ngram8_index
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

python -u -m src.extraction.sample_gpt2 \
  --suffix-file /path/to/runs/memrl/best_suffix.json
```

Train RL (defaults from `config.py`):

```bash
python -u -m src.RL.train_mem_prompt
```

Build AUX tokens / index:

```bash
python -u -m src.datasets_building.build_aux_tokens
python -u -m src.datasets_building.build_ngram8_index
```

---

## How the pieces fit

1. `build_aux_tokens.py` — stream dataset → GPT-2 tokenize → write `tokens.uint32` (+ `doc_offsets.uint64`).
2. `build_ngram8_index.py` — slide 8-gram window → insert `(hash, pos)` into SQLite with an index.
3. `sample_gpt2.py` — generate → compute metrics/score → (opt.) verify via index → write JSONL.
4. `RL` — learn suffix that increases proxy + hits; save the best suffix JSON.

---

## Troubleshooting

* **Padding warnings** (GPT-2): use `padding_side='left'` and `pad_token=eos`.
* **OOM on `gpt2-xl`**: use fp16 (CUDA), reduce batch / seq-len, free GPU memory.
* **Index path confusion**: pass the **directory** (`auxidx_dir`) containing both `tokens.uint32` and `ng8.sqlite`.
* **Streaming issues**: ensure `zstandard` and compatible `datasets`/`fsspec`/`aiohttp`.
* **NaNs in Categorical**: we clamp/sanitize logits in RL; if it persists, lower LR or clamp tighter.

---

## Scaling up

* Increase `BUILD_AUX.max_tokens` and rebuild the index.
* For very large corpora, consider sharding the index (this project uses a simple SQLite-based n-gram map).

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

## Repro tips

* Log versions: `python -V`, `torch.__version__`, `transformers.__version__`, `datasets.__version__`.
* Each run writes under `runs/<subdir>/<timestamp>`.
* Seeds: `TRAIN.seed` (and optionally `torch.cuda.manual_seed_all`).

---

## Acknowledgments

This project builds on public datasets and prior memorization work. The implementation is minimal to keep experimentation feasible on university hardware.
