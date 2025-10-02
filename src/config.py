# --- Paths ---
PATHS = {
    "hf_home": "/home/iscb/wolfson/annab4/.cache/huggingface",
    "auxidx_dir": "/home/iscb/wolfson/annab4/autoserum/auxidx",   # tokens.uint32, doc_offsets.uint64, ng8.sqlite
    "runs_dir": "/home/iscb/wolfson/annab4/AutoSerum/runs", 
    "corpus": "/home/iscb/wolfson/annab4/AutoSerum/data/slices.txt"
}

# --- Build AUX defaults ---
BUILD_AUX = {
    "dataset": "cerebras/SlimPajama-627B",
    "split": "train",
    "max_tokens": 50_000_000,
    "progress_steps": 10,
}

# --- N-gram index defaults ---
INDEX = {
    "ngram": 8,
    "downsample": 1,
    "progress_steps": 10,
}

# --- Train defaults ---
TRAIN = {
    "device": "cuda",
    "task_lm": "gpt2",
    "policy_lm": "gpt2",
    "k_tokens": 4,
    "cand_vocab_size": 256,
    "max_new_tokens": 120,
    "batch_size": 8,
    "iters": 200,
    "use_repetition": True,
    "rep_prob": 0.35,
    "tail_chars": 12,
    "rep_times": 3,
    "slice_len_chars": 100,
    "gt_len_chars": 120,
    "window_k": 8,
    "ema_beta": 0.9,
    "lr": 3e-3,
    "seed": 0,
    "ent_coef": 0.01,
    "temp": 1.0,
    "temp_min": 0.7,
    "max_grad_norm": 1.0,
    "save_every": 50,
    "out_subdir": "memrl",
    "base_prefix": "You are a helpful assistant. If you recall this passage "
                   "from your training data, continue it exactly and verbatim:\n",
}

# --- Generation defaults ---
GEN = {
    "N": 1000,
    "batch_size": 1,
    "seq_len": 256,
    "top_k": 40,
    "verify": True,
    "aux_window": 50,
    "membership_thr": 1.3,
    "ppl_thr": 25.0,
    "score_thr": 1.0,
    "out_subdir": "gen",
    "main_lm": "gpt2-xl",
    "ref_lm": "gpt2",
    "top_p": 1.0,
    "prompt": "<|endoftext|>",
    "progress_steps": 10,
}
