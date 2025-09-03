python /home/iscb/wolfson/annab4/AutoSerum/scripts/eval_extraction.py \
  --auxidx /home/iscb/wolfson/annab4/autoserum/auxidx \
  --window 32 \
  --runs \
    baseline:/home/iscb/wolfson/annab4/slurm_outputs/runs/2025-08-30_21-22-41 \
    rl:/home/iscb/wolfson/annab4/slurm_outputs/runs/2025-09-03_02-04-02 \
    random:/home/iscb/wolfson/annab4/slurm_outputs/runs/2025-09-03_13-57-49 \
  --outcsv /home/iscb/wolfson/annab4/slurm_outputs/runs/eval_summary.csv
