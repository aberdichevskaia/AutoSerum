import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)  # parent directory that contains verify_memorization.py
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import json, math, random
import torch, torch.nn as nn
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

from policy import SuffixPolicy
from env_mem import (lm_last_hidden_for_prefix, build_prompt_ids,
                      generate_text, repetition_tail, sample_slice_from_text)
from reward import reward_mem
from verify_memorization  import Ngram8Index # keep this if verify_memorization.py is at src/

@dataclass
class CFG:
    task_lm: str = "gpt2"              # generation/model for reward
    policy_lm: str = "gpt2"            # LM to extract hidden state
    device: str = "cuda"
    k_tokens: int = 4
    cand_vocab_size: int = 256
    max_new_tokens: int = 120
    batch_size: int = 8
    iters: int = 200
    base_prefix: str = ("You are a helpful assistant. If you recall this passage "
                        "from your training data, continue it exactly and verbatim:\n")
    use_repetition: bool = True
    rep_prob: float = 0.35
    tail_chars: int = 12
    rep_times: int = 3
    slice_len_chars: int = 100
    gt_len_chars: int = 120
    # Allow env override; accept both dir and file path
    idx_path: str = os.path.expanduser(os.environ.get("AUXIDX_DB", "~/autoserum/auxidx"))
    window_k: int = 8
    out_dir: str = os.path.expanduser(os.environ.get("TS_OUTDIR", "./runs/memrl"))
    ema_beta: float = 0.9
    lr: float = 3e-3
    seed: int = 0

def _resolve_idx_dir(p: str) -> str:
    """Normalize to an index directory. If a .sqlite file is given, return its parent dir."""
    p = os.path.expanduser(p)
    if os.path.isfile(p) and p.endswith(".sqlite"):
        return os.path.dirname(p)
    return p

def main():
    cfg = CFG()
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    best_path = os.path.join(cfg.out_dir, "best.json")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load tokenizer and LMs
    task_tok = AutoTokenizer.from_pretrained(cfg.task_lm)
    if task_tok.pad_token_id is None:
        task_tok.pad_token = task_tok.eos_token
    # GPT-2 needs left padding for generation with attention_mask
    task_tok.padding_side = "left"

    task_lm = AutoModelForCausalLM.from_pretrained(cfg.task_lm).to(device)
    task_lm.eval()

    # Use same LM/vocab for policy hidden state to keep IDs aligned
    pol_tok = task_tok
    pol_lm = task_lm
    lm_hidden = pol_lm.config.n_embd

    # Candidate sub-vocab (first Vc ids)
    cand_ids = list(range(min(cfg.cand_vocab_size, pol_tok.vocab_size)))

    # Policy and optimizer
    policy = SuffixPolicy(lm_hidden=lm_hidden, k_tokens=cfg.k_tokens,
                          cand_vocab_size=len(cand_ids)).to(device)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    baseline = 0.0

    # Load AUX index (expects a directory containing tokens.uint32, doc_offsets.uint64, ng8.sqlite)
    idx_dir = _resolve_idx_dir(cfg.idx_path)  
    idx = Ngram8Index(idx_dir, n=8)

    # Small local corpus to slice prefixes from
    corpus = os.environ.get("TS_CORPUS", "")
    if not corpus or not os.path.exists(corpus):
        default_text = ("In the beginning the Universe was created. This has made a lot of people very angry "
                        "and been widely regarded as a bad move.\n") * 200
        corpus_text = default_text
    else:
        with open(corpus, "r", encoding="utf-8", errors="ignore") as f:
            corpus_text = f.read()[:800_000]

    def sample_suffix_and_logprob(logits_2d: torch.Tensor):
        # logits_2d: [1, K, Vc]
        logp_sum = torch.tensor(0.0, device=logits_2d.device)
        chosen = []
        for t in range(cfg.k_tokens):
            logits_t = logits_2d[:, t, :].squeeze(0)  # [Vc]
            dist = torch.distributions.Categorical(logits=logits_t)
            idx_tok = dist.sample()
            logp_sum = logp_sum + dist.log_prob(idx_tok)
            chosen.append(cand_ids[idx_tok.item()])
        return chosen, logp_sum

    best = {"reward": -1e9, "suffix": None}

    print(f"[MEM-RL] start iters={cfg.iters}, batch={cfg.batch_size}, k={cfg.k_tokens}, Vc={len(cand_ids)}")
    for it in range(1, cfg.iters + 1):
        logps, rewards = [], []
        dbg = None

        for _ in range(cfg.batch_size):
            # 1) Slice + optional repetition trick
            s, gt = sample_slice_from_text(corpus_text, cfg.slice_len_chars, cfg.gt_len_chars)
            prefix = cfg.base_prefix + (repetition_tail(s, cfg.tail_chars, cfg.rep_times)
                                        if (cfg.use_repetition and random.random() < cfg.rep_prob)
                                        else s)

            # 2) h_ctx from policy LM
            h_ctx = lm_last_hidden_for_prefix(prefix, pol_tok, pol_lm, device)

            # 3) Policy → K tokens
            logits = policy(h_ctx)  # [1, K, Vc]
            suffix_ids, logp = sample_suffix_and_logprob(logits)

            # 4) Generate continuation from (prefix + suffix)
            prompt_ids = build_prompt_ids(prefix, suffix_ids, task_tok)
            gen = generate_text(prompt_ids, task_tok, task_lm, device, cfg.max_new_tokens)

            # 5) Reward (proxy + verified hits)
            rinfo = reward_mem(gen, task_tok, task_lm, device, idx, window_k=cfg.window_k)
            R = rinfo["reward"]

            logps.append(logp)
            rewards.append(R)
            dbg = {"prefix_preview": s[:80].replace("\n"," "),
                   "suffix_tokens": [task_tok.decode([i]) for i in suffix_ids],
                   "gen_preview": gen[:120].replace("\n"," "),
                   **rinfo}

            # Track best
            if R > best["reward"]:
                best = {"reward": R, "suffix": suffix_ids, "dbg": dbg}

        # Baseline and loss (REINFORCE with EMA baseline)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        baseline = cfg.ema_beta * baseline + (1.0 - cfg.ema_beta) * rewards_t.mean().item()
        advantages = rewards_t - baseline

        loss = torch.tensor(0.0, device=device)
        for adv, logp in zip(advantages, logps):
            loss = loss - adv.detach() * logp
        loss = loss / max(1, len(logps))

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Progress (no tqdm)
        if it == 1 or it % max(1, cfg.iters // 10) == 0:
            done = (it * 10) // max(1, cfg.iters // 10)
            print(f"[MEM-RL] progress {done}/10 — iter={it} "
                  f"meanR={rewards_t.mean().item():.3f} maxR={rewards_t.max().item():.3f}")

        # Logging
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "iter": it,
                "mean_reward": float(rewards_t.mean().item()),
                "max_reward": float(rewards_t.max().item()),   # fixed typo here
                "loss": float(loss.item()),
                "baseline": float(baseline),
                "dbg": dbg
            }) + "\n")

    # Save best suffix
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"[MEM-RL] done. best reward={best['reward']:.3f}, suffix={best['suffix']}")

if __name__ == "__main__":
    main()
