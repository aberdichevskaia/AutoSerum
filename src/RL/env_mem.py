import random, torch
from typing import List, Tuple, Dict
from transformers import PreTrainedTokenizer, PreTrainedModel

def lm_last_hidden_for_prefix(prefix_text: str,
                              tok: PreTrainedTokenizer,
                              lm: PreTrainedModel,
                              device: torch.device) -> torch.Tensor:
    ids = tok.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = lm(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_last = out.hidden_states[-1][0, -1, :].detach()
    return h_last

def build_prompt_ids(prefix_text: str, suffix_token_ids: List[int],
                     tok: PreTrainedTokenizer) -> torch.Tensor:
    ids = tok.encode(prefix_text, add_special_tokens=False) + suffix_token_ids
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def generate_text(prompt_ids: torch.Tensor,
                  tok: PreTrainedTokenizer,
                  lm: PreTrainedModel,
                  device: torch.device,
                  max_new_tokens: int = 120) -> str:
    inp = prompt_ids.unsqueeze(0).to(device)
    attn = torch.ones_like(inp)
    out = lm.generate(
        input_ids=inp,
        attention_mask=attn,
        do_sample=True, temperature=0.9, top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out[0][prompt_ids.shape[0]:].tolist()
    return tok.decode(gen_ids, clean_up_tokenization_spaces=True)

def repetition_tail(prefix: str, tail_chars: int = 12, repeat: int = 3) -> str:
    tail = prefix[-tail_chars:]
    return prefix + (" " + tail) * repeat

def sample_slice_from_text(corpus: str, slice_len_chars: int, gt_len_chars: int) -> Tuple[str, str]:
    max_r = max(0, len(corpus) - (slice_len_chars + gt_len_chars) - 1)
    r = random.randint(0, max_r)
    s = corpus[r: r + slice_len_chars]
    gt = corpus[r + slice_len_chars: r + slice_len_chars + gt_len_chars]
    return s, gt
