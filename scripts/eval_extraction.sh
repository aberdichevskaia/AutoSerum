source /home/iscb/wolfson/annab4/miniconda3/etc/profile.d/conda.sh
conda activate auto_serum

python /home/iscb/wolfson/annab4/AutoSerum/src/extraction/eval_extraction.py \
  --auxidx /home/iscb/wolfson/annab4/autoserum/auxidx_big \
  --window 50 \
  --runs \
    gpt2xl_none:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2xl_no_suffix \
    gpt2xl_random:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2xl_true_random_suffix \
    gpt2xl_proxy:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2xl_proxy_suffix \
    gpt2xl_naive:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2xl_naive_suffix \
    gpt2xl_gap:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2xl_gap_suffix \
    gpt2_proxy:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2_proxy_suffix \
    gpt2_random:/home/iscb/wolfson/annab4/AutoSerum/runs/gen/gpt2_true_random_suffix \
  --outcsv /home/iscb/wolfson/annab4/AutoSerum/runs/eval_summary.csv
