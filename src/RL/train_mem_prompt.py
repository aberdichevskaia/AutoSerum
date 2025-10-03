# src/RL/train_mem_prompt.py
import os
import sys
import json
import random
import argparse
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# resolving local pathes
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # AutoSerum/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import PATHS, TRAIN
from src.RL.policy import SuffixPolicy
from src.RL.env_mem import (
    lm_last_hidden_for_prefix,
    build_prompt_ids,
    generate_text,
    repetition_tail,
    sample_slice_from_text,
)
from src.extraction.verify_memorization import Ngram8Index
from src.RL.reward import reward_mem

# --------------------------
# Config container
# --------------------------
@dataclass
class CFG:
    # Models / device
    task_lm: str
    policy_lm: str
    device: str

    # Policy head (suffix)
    k_tokens: int
    cand_vocab_size: int

    # Generation
    max_new_tokens: int
    batch_size: int
    iters: int

    # Prefix construction
    base_prefix: str
    use_repetition: bool
    rep_prob: float
    tail_chars: int
    rep_times: int
    slice_len_chars: int
    gt_len_chars: int

    # AUX index
    idx_path: str
    window_k: int

    # IO / training stability
    out_dir: str
    ema_beta: float
    lr: float
    seed: int

    # Exploration / regularization
    ent_coef: float
    temp: float
    temp_min: float
    max_grad_norm: float

    # Checkpoints
    save_every: int


def _resolve_idx_dir(p: str) -> str:
    """Normalize to an index directory. If a .sqlite file is given, return its parent dir."""
    p = os.path.expanduser(p)
    if os.path.isfile(p) and p.endswith(".sqlite"):
        return os.path.dirname(p)
    return p


def anneal_temp(it: int, iters: int, t0: float, tmin: float) -> float:
    """Linear anneal from t0 to tmin across training."""
    iters = max(1, iters)
    alpha = min(max(it / iters, 0.0), 1.0)
    return max(tmin, t0 + (tmin - t0) * alpha)


def build_argparser() -> argparse.Namespace:
    """CLI overrides for key paths and training params (env vars are not used)."""
    ap = argparse.ArgumentParser()
    # Paths
    ap.add_argument("--auxidx-dir", type=str, default=None, help="Path to AUX index dir (tokens/offsets/sqlite)")
    ap.add_argument("--runs-dir", type=str, default=None, help="Base directory for runs")
    ap.add_argument("--out-subdir", type=str, default=None, help="Subdirectory under runs for this job")
    ap.add_argument("--corpus", type=str, default=None, help="Local corpus file to sample prefixes from")
    ap.add_argument("--hf-cache", type=str, default=None, help="HuggingFace cache dir (passed to cache_dir)")

    # Training overrides (optional)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--task-lm", type=str, default=None)
    ap.add_argument("--policy-lm", type=str, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--k-tokens", type=int, default=None)
    ap.add_argument("--cand-vocab-size", type=int, default=None)
    ap.add_argument("--window-k", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)

    return ap.parse_args()


def make_cfg_from_sources(paths_cfg: dict, train_cfg: dict, args: argparse.Namespace) -> tuple[CFG, str, str]:
    """Compose final CFG and return (cfg, hf_cache_dir, corpus_path)."""
    # Start from config.py values (single source of truth)
    paths = dict(paths_cfg)
    train = dict(train_cfg)

    # Apply CLI overrides
    if args.auxidx_dir is not None:
        paths["auxidx_dir"] = args.auxidx_dir
    if args.runs_dir is not None:
        paths["runs_dir"] = args.runs_dir
    if args.out_subdir is not None:
        train["out_subdir"] = args.out_subdir
    if args.corpus is not None:
        paths["corpus"] = args.corpus
    if args.hf_cache is not None:
        paths["hf_home"] = args.hf_cache  # used as cache_dir in HF loaders

    if args.device is not None:
        train["device"] = args.device
    if args.task_lm is not None:
        train["task_lm"] = args.task_lm
    if args.policy_lm is not None:
        train["policy_lm"] = args.policy_lm
    if args.iters is not None:
        train["iters"] = args.iters
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        train["max_new_tokens"] = args.max_new_tokens
    if args.k_tokens is not None:
        train["k_tokens"] = args.k_tokens
    if args.cand_vocab_size is not None:
        train["cand_vocab_size"] = args.cand_vocab_size
    if args.window_k is not None:
        train["window_k"] = args.window_k
    if args.lr is not None:
        train["lr"] = args.lr
    if args.seed is not None:
        train["seed"] = args.seed
    if args.save_every is not None:
        train["save_every"] = args.save_every
        
    # Base prefix
    base_prefix = train.get("base_prefix")

    # Compose out_dir from runs_dir + out_subdir
    runs_dir = paths["runs_dir"]
    out_subdir = train.get("out_subdir", "memrl")
    out_dir = os.path.join(runs_dir, out_subdir)

    cfg = CFG(
        # Models / device
        task_lm=str(train["task_lm"]),
        policy_lm=str(train["policy_lm"]),
        device=str(train["device"]),

        # Policy head
        k_tokens=int(train["k_tokens"]),
        cand_vocab_size=int(train["cand_vocab_size"]),

        # Generation
        max_new_tokens=int(train["max_new_tokens"]),
        batch_size=int(train["batch_size"]),
        iters=int(train["iters"]),

        # Prefix
        base_prefix=base_prefix,
        use_repetition=bool(train["use_repetition"]),
        rep_prob=float(train["rep_prob"]),
        tail_chars=int(train["tail_chars"]),
        rep_times=int(train["rep_times"]),
        slice_len_chars=int(train["slice_len_chars"]),
        gt_len_chars=int(train["gt_len_chars"]),

        # AUX
        idx_path=_resolve_idx_dir(str(paths["auxidx_dir"])),
        window_k=int(train["window_k"]),

        # IO / stability
        out_dir=str(out_dir),
        ema_beta=float(train["ema_beta"]),
        lr=float(train["lr"]),
        seed=int(train["seed"]),

        # Exploration / regularization
        ent_coef=float(train["ent_coef"]),
        temp=float(train["temp"]),
        temp_min=float(train["temp_min"]),
        max_grad_norm=float(train["max_grad_norm"]),

        # Checkpoints
        save_every=int(train["save_every"]),
    )

    hf_cache_dir = str(paths["hf_home"])
    corpus_path = str(paths["corpus"])
    return cfg, hf_cache_dir, corpus_path


def main():
    # Args → merge with config.py to build final CFG
    args = build_argparser()
    cfg, hf_cache_dir, corpus_path = make_cfg_from_sources(PATHS, TRAIN, args)

    # Repro
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # IO setup
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    best_path = os.path.join(cfg.out_dir, "best.json")

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Tokenizer + LMs (pass cache_dir explicitly; no env vars)
    task_tok = AutoTokenizer.from_pretrained(cfg.task_lm, cache_dir=hf_cache_dir)
    if task_tok.pad_token_id is None:
        task_tok.pad_token = task_tok.eos_token
    task_tok.padding_side = "left"

    task_lm = AutoModelForCausalLM.from_pretrained(cfg.task_lm, cache_dir=hf_cache_dir).to(device)
    task_lm.eval()

    # Use same LM/vocab for policy hidden state (keeps token IDs aligned)
    pol_tok = task_tok
    pol_lm = task_lm
    lm_hidden = pol_lm.config.n_embd

    # Candidate sub-vocab (first Vc ids)
    
    cand_ids = list(range(min(cfg.cand_vocab_size, pol_tok.vocab_size)))
    
        # ---- Expert suffixes (library) ----
    EXPERT_SUFFIX_TEXTS = [
        "\n",           # plain newline (document-style continuation)
        "\n\n",         # blank line (common in web data)
        ":\n",          # colon+newline (list/quote continuation)
        '"""\n',        # triple-quote fence
        "```\n",        # code fence
        " ###\n",       # markdown-ish section cue
        " —\n",         # em-dash continuation
        " Continue verbatim:\n",             # explicit verbatim cue
        " Do not paraphrase. Continue:\n",   # explicit instruction
        (" la" * 16) + "\n",     # tiny repeating pattern
        (" --" * 16) + "\n",     # repeated dash fence
    ]
    # Build fixed-length K token sequences mapped to candidate vocab
    expert_suffixes = []


    for s in EXPERT_SUFFIX_TEXTS:
        ids = pol_tok.encode(s, add_special_tokens=False)
        # pad/trim to K
        if not ids:
            ids = [pol_tok.eos_token_id]
        if len(ids) < cfg.k_tokens:
            ids = ids + [ids[-1]] * (cfg.k_tokens - len(ids))
        ids = ids[:cfg.k_tokens]
        # map to candidate vocab (use fallback if OOV)
        mapped = []
        for t in ids:
            if t in cand_ids:
                mapped.append(t)
            else:
                mapped.append(cand_ids[0])  # fallback
        expert_suffixes.append(mapped)

    p_expert = 0.30   # 30% of rollouts use expert suffixes (tweak as you like)
    # -----------------------------------

    # Policy + optimizer
    policy = SuffixPolicy(lm_hidden=lm_hidden, k_tokens=cfg.k_tokens, cand_vocab_size=len(cand_ids)).to(device)
    policy = policy.to(dtype=torch.float32)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    baseline = 0.0

    # AUX index
    idx = Ngram8Index(cfg.idx_path, n=8)

    # Local corpus (for simple prefix slices)
    if not corpus_path or not os.path.exists(corpus_path):
        default_text = (
            "In the beginning the Universe was created. This has made a lot of people very angry "
            "and been widely regarded as a bad move.\n"
        ) * 200
        corpus_text = default_text
    else:
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            corpus_text = f.read()[:800_000]

    def sample_suffix_and_logprob(logits_2d: torch.Tensor, temp: float):
        """
        logits_2d: [1, K, Vc] — returns (chosen_ids, logp_sum, entropy_sum,mode).
        We apply temperature and accumulate entropy for an exploration bonus.
        If we choose an expert rollout, we DON'T sample; we score that fixed action under the policy.
        """
        assert temp > 0.0
        # Defensive cleaning to avoid NaNs/Infs in Categorical
        logits_2d = logits_2d.to(dtype=torch.float32)
        logits_2d = torch.nan_to_num(logits_2d, nan=0.0, posinf=50.0, neginf=-50.0)
        logits_2d = torch.clamp(logits_2d, -50.0, 50.0)
        K = logits_2d.shape[1]  #defined once, before branche conditioning
        logp_sum = torch.tensor(0.0, device=logits_2d.device)
        ent_sum = torch.tensor(0.0, device=logits_2d.device)
        chosen = []

        use_expert = (len(expert_suffixes) > 0) and (random.random() < p_expert)

        if use_expert:
            # pick an expert sequence
            expert = random.choice(expert_suffixes)  # list of K token IDs (full-vocab IDs)
            for t in range(K):
                logits_t = (logits_2d[:, t, :].squeeze(0) / temp)
                # find index j in candidate vocab for this expert token
                tok = expert[t]
                try:
                    j = cand_ids.index(tok)
                except ValueError:
                    j = 0
                dist = torch.distributions.Categorical(logits=logits_t)
                logp_sum = logp_sum + dist.log_prob(torch.tensor(j, device=logits_t.device))
                ent_sum  = ent_sum + dist.entropy()
                chosen.append(tok)
            mode = "expert"
        else:
            for t in range(K):
                logits_t = logits_2d[:, t, :].squeeze(0) / temp  # [Vc]
                if not torch.isfinite(logits_t).all():
                    logits_t = torch.zeros_like(logits_t)
                dist = torch.distributions.Categorical(logits=logits_t)
                idx_tok = dist.sample()
                logp_sum = logp_sum + dist.log_prob(idx_tok)
                ent_sum = ent_sum + dist.entropy()
                chosen.append(cand_ids[idx_tok.item()])
            mode = "rl"

        return chosen, logp_sum, ent_sum, mode

    best = {"reward": -1e9, "suffix": None}

    print(f"[MEM-RL] start iters={cfg.iters}, batch={cfg.batch_size}, k={cfg.k_tokens}, Vc={len(cand_ids)}")
    for it in range(1, cfg.iters + 1):
        curr_temp = anneal_temp(it, cfg.iters, cfg.temp, cfg.temp_min)

        logps, rewards, entropies = [], [], []
        dbg = None

        for _ in range(cfg.batch_size):
            # 1) Make prefix (slice + optional repetition trick)
            s, gt = sample_slice_from_text(corpus_text, cfg.slice_len_chars, cfg.gt_len_chars)
            prefix = cfg.base_prefix + (
                repetition_tail(s, cfg.tail_chars, cfg.rep_times)
                if (cfg.use_repetition and random.random() < cfg.rep_prob)
                else s
            )

            # 2) Last hidden state for prefix (policy LM)
            h_ctx = lm_last_hidden_for_prefix(prefix, pol_tok, pol_lm, device)
            h_ctx = h_ctx.detach().clone().contiguous()

            # 3) Policy → K tokens
            logits = policy(h_ctx)  # [1, K, Vc]
            suffix_ids, logp, ent, mode  = sample_suffix_and_logprob(logits, temp=curr_temp)

            # 4) Generate continuation from (prefix + suffix)
            prompt_ids = build_prompt_ids(prefix, suffix_ids, task_tok)
            gen = generate_text(prompt_ids, task_tok, task_lm, device, cfg.max_new_tokens)

            # 5) Reward (proxy + verified hits)
            rinfo = reward_mem(gen, task_tok, task_lm, device, idx, window_k=cfg.window_k)
            R = rinfo["reward"]

            logps.append(logp)
            entropies.append(ent)
            rewards.append(R)
            dbg = {
                "prefix_preview": s[:80].replace("\n", " "),
                "suffix_tokens": [task_tok.decode([i]) for i in suffix_ids],
                "gen_preview": gen[:120].replace("\n", " "),
                **rinfo,
                "rollout_mode": mode,  # "expert" or "rl"
            }

            # Track best
            if R > best["reward"]:
                best = {"reward": R, "suffix": suffix_ids, "dbg": dbg}

        # Baseline + loss (REINFORCE with EMA baseline + entropy bonus)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        entropies_t = torch.stack(entropies) if len(entropies) > 0 else torch.tensor(0.0, device=device)

        baseline = cfg.ema_beta * baseline + (1.0 - cfg.ema_beta) * rewards_t.mean().item()
        advantages = rewards_t - baseline

        loss_pg = torch.tensor(0.0, device=device)
        for adv, logp in zip(advantages, logps):
            loss_pg = loss_pg - adv.detach() * logp
        loss_pg = loss_pg / max(1, len(logps))

        # maximize entropy => subtract in loss
        loss_ent = -cfg.ent_coef * (entropies_t.mean() if entropies_t.ndim > 0 else entropies_t)
        loss = loss_pg + loss_ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        opt.step()

        # Progress (no tqdm)
        if it == 1 or it % max(1, cfg.iters // 10) == 0:
            done = (it * 10) // max(1, cfg.iters // 10)
            mean_H = float(entropies_t.mean().item() if entropies_t.ndim > 0 else entropies_t.item())
            print(
                f"[MEM-RL] progress {done}/10 — iter={it} "
                f"meanR={rewards_t.mean().item():.3f} maxR={rewards_t.max().item():.3f} "
                f"T={curr_temp:.2f} H={mean_H:.2f}"
            )

        # Logging
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "iter": it,
                        "mean_reward": float(rewards_t.mean().item()),
                        "max_reward": float(rewards_t.max().item()),
                        "loss": float(loss.item()),
                        "loss_pg": float(loss_pg.item()),
                        "loss_ent": float(loss_ent.item()),
                        "baseline": float(baseline),
                        "temp": float(curr_temp),
                        "mean_entropy": float(
                            entropies_t.mean().item() if entropies_t.ndim > 0 else entropies_t.item()
                        ),
                        "dbg": dbg,
                    }
                )
                + "\n"
            )

        # Periodic checkpoints for quick A/B with sample_gpt2 --suffix-file
        if it % max(1, cfg.save_every) == 0 and best["suffix"] is not None:
            ckpt_best = os.path.join(cfg.out_dir, f"best_ckpt_iter_{it}.json")
            with open(ckpt_best, "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)
            with open(os.path.join(cfg.out_dir, f"suffix_iter_{it}.json"), "w", encoding="utf-8") as f:
                json.dump({"suffix": best["suffix"]}, f, ensure_ascii=False)

    # Save final best
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    if best["suffix"] is not None:
        with open(os.path.join(cfg.out_dir, "best_suffix.json"), "w", encoding="utf-8") as f:
            json.dump({"suffix": best["suffix"]}, f, ensure_ascii=False, indent=2)

    print(f"[MEM-RL] done. best reward={best['reward']:.3f}, suffix={best['suffix']}")


if __name__ == "__main__":
    main()
