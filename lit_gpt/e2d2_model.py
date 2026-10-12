import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self
from lit_gpt.config import Config
from lit_gpt.model import Block, CausalSelfAttention
from lit_gpt.fused_rotary_embedding import apply_rotary_emb_func
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


class CrossAttention(nn.Module):
    """Standard cross-attention: Q from decoder, K/V from encoder memory.

    Dedicated projections (no weight sharing with self-attention). GQA layout matches
    lit_gpt.model.CausalSelfAttention: n_head query heads, n_query_groups K/V groups,
    head_size per head. Uses the same SDPA-with-bool-mask path (True = attend).
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        hs = config.head_size
        self.q_proj  = nn.Linear(config.n_embd, config.n_head * hs, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_query_groups * hs, bias=config.bias)
        self.proj    = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config

    def forward(
        self,
        x_dec: torch.Tensor,
        mem: torch.Tensor,
        q_rope: RoPECache,
        kv_rope: RoPECache,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T_dec, C = x_dec.size()
        T_mem = mem.size(1)
        q = self.q_proj(x_dec).view(B, T_dec, self.config.n_head, self.config.head_size)
        kv = self.kv_proj(mem).view(B, T_mem, 2 * self.config.n_query_groups, self.config.head_size)
        k, v = kv.split(self.config.n_query_groups, dim=-2)

        cos_q, sin_q = q_rope
        cos_k, sin_k = kv_rope
        q = apply_rotary_emb_func(q, cos_q, sin_q, False, True)
        k = apply_rotary_emb_func(k, cos_k, sin_k, False, True)

        scale = 1.0 / math.sqrt(self.config.head_size)
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        if q_.size() != k_.size():
            k_ = k_.repeat_interleave(q_.shape[1] // k_.shape[1], dim=1)
            v_ = v_.repeat_interleave(q_.shape[1] // v_.shape[1], dim=1)

        # Handle degenerate rows (all-False, e.g. cross-mask block 0 which has an empty
        # strict prefix). SDPA on an all-False row produces NaN. We flip a single key True
        # on those rows so softmax has a valid entry, then zero the corresponding output rows
        # so their residual contribution is 0 -- equivalent to skipping cross-attn for them.
        row_has_key = mask.any(dim=-1, keepdim=True)      # (..., T_dec, 1); True if row has any attended key
        empty_rows = ~row_has_key                          # (..., T_dec, 1)
        # Broadcast to (..., T_dec, T_mem) and place True at key 0 for empty rows.
        pad = torch.zeros_like(mask)
        pad[..., :, 0:1] = empty_rows
        safe_mask = mask | pad

        y = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=safe_mask, dropout_p=0.0, scale=scale, is_causal=False
        )                                                  # (B, n_head_q, T_dec, hs)
        # Gate rows: (..., T_dec, 1) row_has_key -> match y's (B, n_head_q, T_dec, hs) layout.
        gate = row_has_key.to(y.dtype)                     # broadcastable to y
        y = (y * gate).transpose(1, 2)                     # (B, T_dec, n_head_q, hs)
        return self.proj(y.reshape(B, T_dec, C))


class CrossDecoderBlock(nn.Module):
    """Standard seq2seq decoder layer: self-attn -> cross-attn -> MLP, each a pre-norm sublayer."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_sa  = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.sa       = CausalSelfAttention(config)
        self.norm_ca  = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.ca       = CrossAttention(config)
        self.norm_mlp = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp      = config.mlp_class(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        dec_rope: RoPECache,
        enc_rope: RoPECache,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        h, _ = self.sa(self.norm_sa(x), dec_rope, x.size(1), self_mask)
        x = x + h
        x = x + self.ca(self.norm_ca(x), mem, dec_rope, enc_rope, cross_mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class E2D2Model(nn.Module):
    """E2D2 with standard cross-attention decoder.

    The encoder is block-causal (_build_encoder_mask below). The decoder replaces the
    concat-form self-attention over [x_enc, x_dec] with a standard seq2seq decoder layer:
    within-block self-attn (b×b) -> cross-attn over encoder memory (banded prefix) -> MLP.
    The encoder memory (post encoder_ln_f) is computed once and reused across all decoder
    layers (last_hidden_state semantics; no shared_kv_cache variant).
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        n_dec_layers = config.n_dec_layers if config.n_dec_layers is not None else config.n_layer // 2
        n_enc_layers = config.n_layer - n_dec_layers
        # Both towers must be non-empty. Without this, n_dec >= n_layer silently yields an
        # empty encoder_blocks, so `mem` degenerates to encoder_ln_f(wte(enc_idx)) -- an
        # embedding lookup and a norm, with no attention over the clean prefix at all.
        # Training still runs and the loss stays finite, so the failure is silent; the
        # decoder-depth sweeps of Section 4.2 walk right up to this boundary
        # (n_dec = L-2, and L-1 at the edge). n_enc = 1 is the pathological end of the
        # memory-tax axis rather than a correctness bug, so it is not asserted here.
        assert n_enc_layers >= 1 and n_dec_layers >= 1, (
            f"n_dec_layers={n_dec_layers} with n_layer={config.n_layer} leaves "
            f"n_enc_layers={n_enc_layers}; both towers need at least one layer "
            f"(1 <= n_dec_layers <= n_layer - 1)"
        )
        self.n_dec_layers = n_dec_layers
        self.n_enc_layers = n_enc_layers
        self.mtp_window = config.mtp_window

        # +1 to support mask token id = padded_vocab_size
        self.wte = nn.Embedding(config.padded_vocab_size + 1, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        self.encoder_blocks = nn.ModuleList(Block(config) for _ in range(n_enc_layers))
        self.encoder_ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)

        self.decoder_blocks = nn.ModuleList(CrossDecoderBlock(config) for _ in range(n_dec_layers))
        self.decoder_ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)

        self.rope_cache: Optional[RoPECache] = None
        self.encoder_mask: Optional[torch.Tensor] = None
        self.self_mask: Optional[torch.Tensor] = None
        self.cross_mask: Optional[torch.Tensor] = None

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, (CausalSelfAttention, CrossAttention))) or \
               (name == "w3.weight" and isinstance(module, SwiGLU)):
                if module in self.decoder_blocks.modules():
                    nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / self.n_dec_layers)
                else:
                    nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / self.n_enc_layers)

    def forward(self, enc_idx: torch.Tensor, dec_idx: torch.Tensor) -> torch.Tensor:
        # enc_idx: (B, n) clean tokens; dec_idx: (B, n) noisy tokens. One forward is shared
        # by training, validate() and duel_validate_sweep() in pretrain/train_e2d2.py.
        B, n = enc_idx.size()
        b = self.mtp_window
        assert self.config.block_size >= n
        assert n % b == 0, f"sequence length {n} must be divisible by mtp_window {b}"

        if self.rope_cache is None:
            self.rope_cache = self._build_rope_cache(enc_idx)
        cos_cache, sin_cache = self.rope_cache
        # Both halves cover positions 0..n-1.
        dec_rope = enc_rope = (cos_cache[:n], sin_cache[:n])

        if self.encoder_mask is None or self.encoder_mask.shape[-1] != n \
           or self.encoder_mask.device != enc_idx.device:
            self.encoder_mask = self._build_encoder_mask(n, b, enc_idx.device)
        if self.self_mask is None or self.self_mask.shape[-1] != n \
           or self.self_mask.device != enc_idx.device:
            self.self_mask = self._build_self_mask(n, b, enc_idx.device)
        if self.cross_mask is None or self.cross_mask.shape[-1] != n \
           or self.cross_mask.device != enc_idx.device:
            self.cross_mask = self._build_cross_mask(n, b, enc_idx.device)

        # --- Encoder (block-causal) ---
        x_enc = self.wte(enc_idx)
        for blk in self.encoder_blocks:
            x_enc, _ = blk(x_enc, enc_rope, n, self.encoder_mask)
        mem = self.encoder_ln_f(x_enc)                     # normalized ONCE, on the memory path

        # --- Decoder (self-attn -> cross-attn -> MLP per layer) ---
        x_dec = self.wte(dec_idx)
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec, mem, dec_rope, enc_rope, self.self_mask, self.cross_mask)
        x_dec = self.decoder_ln_f(x_dec)
        dec_logits = self.lm_head(x_dec)                   # (B, n, V)
        return dec_logits

    def _build_encoder_mask(self, n: int, b: int, device: torch.device) -> torch.Tensor:
        # Block k attends to all tokens in blocks 0..k.
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        for k in range(n // b):
            mask[k * b:(k + 1) * b, :(k + 1) * b] = True
        rem = n % b
        if rem:
            mask[n - rem:, :n] = True
        return mask.unsqueeze(0).unsqueeze(0)              # (1, 1, n, n)

    def _build_self_mask(self, n: int, b: int, device: torch.device) -> torch.Tensor:
        # Block-diagonal b×b (full within-block bidirectional). Matches BR of e2d2_model.py:141.
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        for k in range(n // b):
            s = k * b
            e = s + b
            mask[s:e, s:e] = True
        rem = n % b
        if rem:
            mask[n - rem:, n - rem:] = True
        return mask.unsqueeze(0).unsqueeze(0)              # (1, 1, n, n)

    def _build_cross_mask(self, n: int, b: int, device: torch.device) -> torch.Tensor:
        # Decoder block k attends encoder positions 0..k*b-1 (strict prefix; matches BL of e2d2_model.py:140).
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        for k in range(n // b):
            s = k * b
            e = s + b
            mask[s:e, :s] = True
        rem = n % b
        if rem:
            n_full = (n // b) * b
            mask[n_full:, :n_full] = True
        return mask.unsqueeze(0).unsqueeze(0)              # (1, 1, n, n)

    def _build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
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
