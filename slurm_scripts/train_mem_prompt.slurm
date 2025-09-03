#!/bin/bash
#SBATCH --job-name=train_mem_prompt
#SBATCH --output=output_train_mem_prompt.txt  
#SBATCH --error=error_train_mem_prompt.txt
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=16
#SBATCH --mem=500000                      
#SBATCH --time=1440  
#SBATCH --mail-type=ALL,TIME_LIMIT_80 
#SBATCH --mail-user=annab4@mail.tau.ac.il  
export HF_HOME=/home/iscb/wolfson/annab4/.cache/huggingface

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate carlini

export HF_HOME=/home/iscb/wolfson/annab4/.cache/huggingface

# Path to the database 
export AUXIDX_DIR=/home/iscb/wolfson/annab4/autoserum/auxidx

# Path to the saving dir
export TS_OUTDIR=/home/iscb/wolfson/annab4/AutoSerum/runs/memrl

# Optional small local corpus to slice prefixes from:
export TS_CORPUS=/home/iscb/wolfson/annab4/AutoSerum/data/slices.txt

python /home/iscb/wolfson/annab4/AutoSerum/src/RL/train_mem_prompt.py
