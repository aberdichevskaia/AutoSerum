import os, sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)  # parent directory that contains verify_memorization.py
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json, random
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from policy import SuffixPolicy
from env_mem import (
    lm_last_hidden_for_prefix,
    build_prompt_ids,
    generate_text,
    repetition_tail,
    sample_slice_from_text,
)
from reward import reward_mem
from verify_memorization import Ngram8Index  # expects index dir with tokens.uint32, doc_offsets.uint64, ng8.sqlite


@dataclass
class CFG:
    # Models / device
    task_lm: str = "gpt2"   # generation + reward PPL model
    policy_lm: str = "gpt2" # hidden-state extractor (keep same vocab)
    device: str = "cuda"

    # Policy head (suffix)
    k_tokens: int = 4
    cand_vocab_size: int = 256

    # Generation
    max_new_tokens: int = 120
    batch_size: int = 8
    iters: int = 200

    # Prefix construction (simple, reproducible)
    base_prefix: str = (
        "You are a helpful assistant. If you recall this passage "
        "from your training data, continue it exactly and verbatim:\n"
    )
    use_repetition: bool = True
    rep_prob: float = 0.35
    tail_chars: int = 12
    rep_times: int = 3
    slice_len_chars: int = 100
    gt_len_chars: int = 120

    # AUX index
    idx_path: str = os.path.expanduser(os.environ.get("AUXIDX_DB", "~/autoserum/auxidx"))
    window_k: int = 8

    # IO / training stability
    out_dir: str = os.path.expanduser(os.environ.get("TS_OUTDIR", "./runs/memrl"))
    ema_beta: float = 0.9
    lr: float = 3e-3
    seed: int = 0

    # Exploration / regularization
    ent_coef: float = 0.01       # entropy bonus weight
    temp: float = 1.0            # initial policy temperature
    temp_min: float = 0.7        # min temperature (linear anneal)
    max_grad_norm: float = 1.0   # gradient clipping

    # Checkpoints
    save_every: int = 50         # save best checkpoint every N iters


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
    # towards tmin as it increases
    return max(tmin, t0 + (tmin - t0) * alpha)


def main():
    cfg = CFG()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    best_path = os.path.join(cfg.out_dir, "best.json")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Tokenizer + LMs
    task_tok = AutoTokenizer.from_pretrained(cfg.task_lm)
    if task_tok.pad_token_id is None:
        task_tok.pad_token = task_tok.eos_token
    task_tok.padding_side = "left"

    task_lm = AutoModelForCausalLM.from_pretrained(cfg.task_lm).to(device)
    task_lm.eval()

    # Use same LM/vocab for policy hidden state (keeps token IDs aligned)
    pol_tok = task_tok
    pol_lm = task_lm
    lm_hidden = pol_lm.config.n_embd

    # Candidate sub-vocab (first Vc ids)
    cand_ids = list(range(min(cfg.cand_vocab_size, pol_tok.vocab_size)))

    # Policy + optimizer
    policy = SuffixPolicy(lm_hidden=lm_hidden, k_tokens=cfg.k_tokens, cand_vocab_size=len(cand_ids)).to(device)
    policy = policy.to(dtype=torch.float32)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    baseline = 0.0

    # AUX index
    idx_dir = _resolve_idx_dir(cfg.idx_path)
    idx = Ngram8Index(idx_dir, n=8)

    # Local corpus (for simple prefix slices)
    corpus = os.environ.get("TS_CORPUS", "")
    if not corpus or not os.path.exists(corpus):
        default_text = (
            "In the beginning the Universe was created. This has made a lot of people very angry "
            "and been widely regarded as a bad move.\n"
        ) * 200
        corpus_text = default_text
    else:
        with open(corpus, "r", encoding="utf-8", errors="ignore") as f:
            corpus_text = f.read()[:800_000]

    def sample_suffix_and_logprob(logits_2d: torch.Tensor, temp: float):
        """
        logits_2d: [1, K, Vc] — returns (chosen_ids, logp_sum, entropy_sum).
        We apply temperature and accumulate entropy for an exploration bonus.
        """
        assert temp > 0.0
        # Clean logits defensively
        logits_2d = logits_2d.to(dtype=torch.float32)
        logits_2d = torch.nan_to_num(logits_2d, nan=0.0, posinf=50.0, neginf=-50.0)
        logits_2d = torch.clamp(logits_2d, -50.0, 50.0)

        logp_sum = torch.tensor(0.0, device=logits_2d.device)
        ent_sum = torch.tensor(0.0, device=logits_2d.device)
        chosen = []

        K = logits_2d.shape[1]
        for t in range(K):
            logits_t = logits_2d[:, t, :].squeeze(0) / temp  # [Vc]
            # Additional guard
            if not torch.isfinite(logits_t).all():
                logits_t = torch.zeros_like(logits_t)
            dist = torch.distributions.Categorical(logits=logits_t)
            idx_tok = dist.sample()
            logp_sum = logp_sum + dist.log_prob(idx_tok)
            ent_sum = ent_sum + dist.entropy()
            chosen.append(cand_ids[idx_tok.item()])

        return chosen, logp_sum, ent_sum


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
            suffix_ids, logp, ent = sample_suffix_and_logprob(logits, temp=curr_temp)

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
