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
conda activate auto_serum

python /home/iscb/wolfson/annab4/AutoSerum/src/extraction/sample_gpt2.py \
  --N 2000 \
  --suffix-file "/home/iscb/wolfson/annab4/slurm_outputs/runs/memrl/best_suffix.json"
