#!/bin/bash
#SBATCH --job-name=build_aux_tokens
#SBATCH --output=output_build_aux_tokens.txt  
#SBATCH --error=error_build_aux_tokens.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=8
#SBATCH --mem=640000                      
#SBATCH --time=1440  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il   

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate auto_serum

python /home/iscb/wolfson/annab4/AutoSerum/src/datasets_building/build_aux_tokens.py \
  #--out /home/iscb/wolfson/annab4/autoserum/auxidx \
  --out /home/iscb/wolfson/annab4/autoserum/auxidx_big \
  --dataset cerebras/SlimPajama-627B \
  --max_tokens 2_000_000_000 \
  --progress_steps 20


