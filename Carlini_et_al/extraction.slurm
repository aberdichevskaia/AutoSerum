#!/bin/bash
#SBATCH --job-name=extraction_carlini
#SBATCH --output=output_extraction_carlini.txt  
#SBATCH --error=error_extraction_carlini.txt
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

python /home/iscb/wolfson/annab4/AutoSerum/Carlini_et_al/extraction.py --N 1000 --batch-size 10