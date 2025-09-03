import torch, torch.nn as nn

class SuffixPolicy(nn.Module):
    """
    π(a | h_ctx): takes LM last hidden state h_ctx ∈ R^{H} and outputs logits [K, Vc].
    """
    def __init__(self, lm_hidden: int, k_tokens: int, cand_vocab_size: int, hidden: int = 256):
        super().__init__()
        self.k = k_tokens
        self.vc = cand_vocab_size
        self.proj = nn.Sequential(
            nn.Linear(lm_hidden, hidden),
            nn.Tanh(),
            # Removed LayerNorm to avoid NaNs from bad inputs
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, k_tokens * cand_vocab_size),
        )

    def forward(self, h_ctx: torch.Tensor) -> torch.Tensor:
        """h_ctx shape: [H] or [B, H] → logits [B, K, Vc]"""
        if h_ctx.dim() == 1:
            h_ctx = h_ctx.unsqueeze(0)
        h_ctx = h_ctx.to(dtype=torch.float32)
        z = self.proj(h_ctx)
        logits = self.head(z).view(-1, self.k, self.vc)
        # Safety clamp to avoid extreme magnitudes
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        return logits
