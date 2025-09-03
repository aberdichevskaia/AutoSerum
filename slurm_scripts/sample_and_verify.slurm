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
#SBATCH --mail-user=annab4@mail.tau.ac.il  

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate carlini

# export HF_TOKEN=hf_xxxx
export HF_HOME=/home/iscb/wolfson/annab4/.cache/huggingface
export AUXIDX_DIR=/home/iscb/wolfson/annab4/autoserum/auxidx

# random prefix
#TODO pathes are not pretty 
python /home/iscb/wolfson/annab4/AutoSerum/src/sample_gpt2.py \
  --N 1000 \
  --batch-size 1 \
  --seq-len 256 \
  --top-k 40 \
  --verify \
  --auxidx "$AUXIDX_DIR" \
  --window 32 \
  --suffix-file "/home/iscb/wolfson/annab4/slurm_outputs/runs/memrl/best_suffix.json" \
  --ppl-thr 15 \
  --score-thr 1.0 \
  --membership-thr 1.3 \
  --progress_steps 10
