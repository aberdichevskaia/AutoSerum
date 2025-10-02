#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --output=output_build_index.txt
#SBATCH --error=error_build_index.txt
#SBATCH --partition=cpu-killable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=640000
#SBATCH --time=1440
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annab4@mail.tau.ac.il

source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate auto_serum

python -u /home/iscb/wolfson/annab4/AutoSerum/src/datasets_building/build_ngram8_index.py \
    --auxidx /home/iscb/wolfson/annab4/autoserum/auxidx_big \
    --downsample 16 \
    --progress_steps 20
