#!/bin/bash
#SBATCH --job-name=sample_and_verify_no_suffix_main_distillgpt2
#SBATCH --output=output_sample_and_verify_no_suffix_main_distillgpt2.txt
#SBATCH --error=error_sample_and_verify_no_suffix_main_distillgpt2.txt
#SBATCH --partition=gpu-mol
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500000
#SBATCH --time=1440
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --mail-user=annab4@mail.tau.ac.il

set -euo pipefail
source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate auto_serum

PY=/home/iscb/wolfson/annab4/AutoSerum/src/extraction/sample_gpt2.py
AUX=/home/iscb/wolfson/annab4/autoserum/auxidx_big
HF=/home/iscb/wolfson/annab4/.cache/huggingface

COMMON="--N 2000 --verify --auxidx $AUX --hf-cache $HF"

python -u "$PY" \
  $COMMON \
  --main-lm "gpt2" \
  --ref-lm "distilgpt2" \
  --run-name "gpt2_no_suffix"

