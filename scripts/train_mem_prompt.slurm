#!/bin/bash
#SBATCH --job-name=train_mem_prompt_naive
#SBATCH --output=output_train_mem_prompt_naive.txt
#SBATCH --error=error_train_mem_prompt_naive.txt
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

python -u /home/iscb/wolfson/annab4/AutoSerum/src/RL/train_mem_prompt.py \
  --auxidx-dir /home/iscb/wolfson/annab4/autoserum/auxidx_big \
  --reward_mode naive
