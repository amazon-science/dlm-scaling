import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self
from lit_gpt.config import Config
from lit_gpt.model import Block, CausalSelfAttention
from xformers.ops import SwiGLU

RoPECache = Tuple[torch.Tensor, torch.Tensor]


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


class BD3LM(nn.Module):
    """Block Discrete Denoising Diffusion Language Model.

    Single tower over [clean_x, noisy_x_t] (length 2L) with a 4-quadrant attention mask:
      TL (clean->clean): block-causal            (k_q >= k_k)
      TR (clean->noisy): fully masked
      BL (noisy->clean): strictly-previous blocks (k_q >  k_k) — no leakage from own clean block
      BR (noisy->noisy): block-diagonal          (k_q == k_k)
    Both halves share positions 0..L-1. Returns logits on the noisy half only.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.mtp_window = config.mtp_window

        # +1 row so mask id (== padded_vocab_size) embeds directly.
        self.wte = nn.Embedding(config.padded_vocab_size + 1, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
        ))
        self.rope_cache: Optional[RoPECache] = None
        self.attn_mask: Optional[torch.Tensor] = None

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, CausalSelfAttention)) or \
               (name == "w3.weight" and isinstance(module, SwiGLU)):
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer)

    def forward(self, clean_idx: torch.Tensor, noisy_idx: torch.Tensor) -> torch.Tensor:
        # clean_idx, noisy_idx: (B, L); returns (B, L, V) — noisy half only.
        B, n = clean_idx.size()
        assert self.config.block_size >= n, \
            f"Cannot forward sequence of length {n}, block_size is only {self.config.block_size}"
        assert n % self.mtp_window == 0, \
            f"sequence length {n} must be divisible by mtp_window {self.mtp_window}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(clean_idx)
        cos_cache, sin_cache = self.rope_cache
        # Both halves refer to the same underlying positions 0..n-1.
        cos = torch.cat([cos_cache[:n], cos_cache[:n]], dim=0)
        sin = torch.cat([sin_cache[:n], sin_cache[:n]], dim=0)

        if self.attn_mask is None or self.attn_mask.shape[-1] != 2 * n \
           or self.attn_mask.device != clean_idx.device:
            self.attn_mask = self._build_mask(n, self.mtp_window, clean_idx.device)

        x = torch.cat([self.wte(clean_idx), self.wte(noisy_idx)], dim=1)  # (B, 2n, C)
        for block in self.transformer.h:
            x, _ = block(x, (cos, sin), 2 * n, self.attn_mask)
        x = self.transformer.ln_f(x)
        return self.lm_head(x[:, n:, :])  # (B, n, V) — noisy half only

    def _build_mask(self, n: int, b: int, device: torch.device) -> torch.Tensor:
        L = 2 * n
        block_idx = torch.arange(n, device=device) // b            # (n,)
        kq = block_idx.unsqueeze(1)                                # (n, 1) query block
        kk = block_idx.unsqueeze(0)                                # (1, n) key block
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        mask[:n, :n] = kq >= kk    # TL: clean->clean, block-causal
        # TR left as False        # clean never attends to noisy
        mask[n:, :n] = kq >  kk    # BL: noisy->clean, STRICTLY previous blocks
        mask[n:, n:] = kq == kk    # BR: noisy->noisy, block-diagonal
        return mask.unsqueeze(0).unsqueeze(0)                      # (1, 1, 2n, 2n)

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size + 1,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))
