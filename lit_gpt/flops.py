"""Single source of truth for FLOP accounting across all pretraining scripts.

Convention (identical for every method; see Appendix B.1 of the paper):
  * training FLOPs = 3 x forward  (backward ~= 2x forward; activation checkpointing OFF)
  * matmul: 2 FLOPs per parameter per token; embedding lookups are free
  * attention: 4d per ALGORITHMICALLY NECESSARY (query, key) pair (QK^T 2d + AV 2d),
    i.e. a mask-aware / theoretical-best kernel. softmax ignored.
  * output heads: 2*d*V per scored position. model_para_config does NOT contain any
    head, so heads are always added explicitly.
  * ignored on purpose (identical across methods, so no relative bias): RMSNorm, rotary,
    softmax, residual adds (all O(n*d)); all evaluation cost (validate / DUEL / lm-eval).

The cross-attention projections are charged as one lump, which is valid because E2D2's
decoder length equals its memory length (both n): q_proj/out proj consume decoder states
and kv_proj consumes the encoder memory (e2d2_model.py:55).

This module intentionally does NOT import torch: it only reads duck-typed config attributes
(n_layer / n_head / n_embd / n_query_groups / head_size / padded_vocab_size / block_size),
so it can be imported for unit testing in an environment without torch. (Note that
`import lit_gpt.flops` still pulls torch in through lit_gpt/__init__.py -- load this file
directly with importlib.util.spec_from_file_location to bypass that.)
"""

from typing import Optional

# ---------------------------------------------------------------- primitives
#
# NOTE: the scalar literals below are INTEGERS (2, 4, 6), not floats. Attention and head
# terms are then exact Python ints, which is what the previous inline blocks produced.
# Writing 2.0 / 4.0 / 6.0 instead coerces them to float and shifts the forward total by
# one ULP on some configs (verified: 7 of 348 model x b x n_dec x ar combos), breaking the
# bit-identity regression against the previous inline accounting. Only params_per_layer()
# should introduce a float.

def matmul_flops(params: float, n_tokens: int) -> float:
    """2 FLOPs per parameter per token."""
    return 2 * params * n_tokens


def attn_flops(n_layers: int, d: int, pairs: int) -> float:
    """4d per (query, key) pair, summed over n_layers identical layers."""
    return n_layers * 4 * d * pairs


def head_flops(d: int, vocab: int, n_positions: int) -> float:
    """2*d*V per position pushed through an output head."""
    return 2 * d * vocab * n_positions


def params_per_layer(total_params_millions: float, n_layer: int) -> float:
    """model_para_config is in millions and holds non-embedding transformer params only."""
    return total_params_millions * 1e6 / n_layer


def cross_attn_kv_params(config) -> float:
    """kv_proj of one CrossAttention sublayer -- consumes the MEMORY (e2d2_model.py:40,55)."""
    return config.n_embd * 2 * config.n_query_groups * config.head_size


def cross_attn_qo_params(config) -> float:
    """q_proj + out proj of one CrossAttention sublayer -- consume DECODER states (:39,41,54,88)."""
    return (
        config.n_embd * config.n_head * config.head_size            # q_proj
        + config.n_embd * config.n_embd                             # out proj
    )


def cross_attn_params(config) -> float:
    """All three projections. Valid as a single lump ONLY when decoder length == memory length,
    which is the case for E2D2 -- see the module docstring."""
    return cross_attn_kv_params(config) + cross_attn_qo_params(config)


# ------------------------------------------------------- (q,k) pair counters
# Every counter takes the sequence geometry and returns the number of
# algorithmically-necessary (query, key) pairs for ONE layer.

def pairs_causal(n: int) -> int:
    """Token-level causal: query i sees [0, i].  n(n+1)/2"""
    return n * (n + 1) // 2


def pairs_full(n: int) -> int:
    """Bidirectional / full matrix.  n^2"""
    return n * n


def pairs_block_causal(n: int, b: int) -> int:
    """Block k sees blocks 0..k in full.  b^2 K(K+1)/2 = n^2/2 + nb/2"""
    assert n % b == 0, f"block_size {n} must be divisible by b={b}"
    K = n // b
    return b * b * K * (K + 1) // 2


def pairs_block_strict_prefix(n: int, b: int) -> int:
    """Block k sees blocks 0..k-1 (strict).  b^2 K(K-1)/2 = n^2/2 - nb/2"""
    assert n % b == 0, f"block_size {n} must be divisible by b={b}"
    K = n // b
    return b * b * K * (K - 1) // 2


def pairs_block_diagonal(n: int, b: int) -> int:
    """Non-overlapping block-diagonal b x b.  K b^2 = n b"""
    assert n % b == 0, f"block_size {n} must be divisible by b={b}"
    return (n // b) * b * b


# ------------------------------------------------------------ per-method APIs
# Each returns (parts: dict[str, float], fwd: float, old_6nd: float).
# old_6nd is the OLD formula's per-sample TRAINING cost (it already contains the 6),
# kept only for the startup cross-reference print and the paper appendix.

def flops_per_sample_ar(config, total_params_millions: float):
    """Causal decoder-only ARM (lit_gpt.model.GPT, flash_attn causal=True)."""
    d, V, n, L = config.n_embd, config.padded_vocab_size, config.block_size, config.n_layer
    P = total_params_millions * 1e6
    parts = {
        "matmul":     matmul_flops(P, n),
        "self_attn":  attn_flops(L, d, pairs_causal(n)),
        "lm_head":    head_flops(d, V, n),
    }
    return parts, sum(parts.values()), 6 * P * n


def flops_per_sample_mdm(config, total_params_millions: float):
    """Bidirectional masked diffusion (lit_gpt.diffmodel.TransEncoder, causal=False)."""
    d, V, n, L = config.n_embd, config.padded_vocab_size, config.block_size, config.n_layer
    P = total_params_millions * 1e6
    parts = {
        "matmul":     matmul_flops(P, n),
        "self_attn":  attn_flops(L, d, pairs_full(n)),   # bidirectional: full matrix
        "lm_head":    head_flops(d, V, n),
    }
    return parts, sum(parts.values()), 6 * P * n


def flops_per_sample_bd3lm(config, total_params_millions: float, b: int):
    """Single tower over [clean, noisy] of length 2n; logits on the noisy half only."""
    d, V, n, L = config.n_embd, config.padded_vocab_size, config.block_size, config.n_layer
    P = total_params_millions * 1e6
    # 4 quadrants of bd3lm_model.py:92-102. TR is fully masked -> 0.
    pairs = (
        pairs_block_causal(n, b)            # TL clean->clean, block-causal
        + pairs_block_strict_prefix(n, b)   # BL noisy->clean, strictly previous blocks
        + pairs_block_diagonal(n, b)        # BR noisy->noisy, block-diagonal
    )                                       # == n^2 + n*b
    parts = {
        "matmul_2n":  matmul_flops(P, 2 * n),
        "attn":       attn_flops(L, d, pairs),
        "lm_head":    head_flops(d, V, n),  # noisy half only (bd3lm_model.py:90)
    }
    return parts, sum(parts.values()), 6 * P * 2 * n


def flops_per_sample_e2d2(config, total_params_millions: float, b: int, n_dec_layers: int):
    """Encoder is block-causal (e2d2_model.py), decoder blocks are non-overlapping."""
    d, V, n, L = config.n_embd, config.padded_vocab_size, config.block_size, config.n_layer
    n_dec = n_dec_layers
    n_enc = L - n_dec
    p_layer = params_per_layer(total_params_millions, L)
    p_enc = p_layer * n_enc
    p_dec = p_layer * n_dec + n_dec * cross_attn_params(config)
    parts = {
        "enc_matmul":   matmul_flops(p_enc, n),
        "enc_attn":     attn_flops(n_enc, d, pairs_block_causal(n, b)),
        "dec_matmul":   matmul_flops(p_dec, n),
        "dec_self":     attn_flops(n_dec, d, pairs_block_diagonal(n, b)),
        "cross_attn":   attn_flops(n_dec, d, pairs_block_strict_prefix(n, b)),
        "lm_head":      head_flops(d, V, n),
    }
    return parts, sum(parts.values()), 6 * (p_enc + p_dec) * n


# ------------------------------------------------------------------- helpers

def max_step_from_flops(flops_e18: float, flops_per_sample_train: float,
                        global_batch_size: int, nodes_num: int) -> int:
    """--flops is in units of 1e18 REAL FLOPs (semantics unchanged from the 6ND era)."""
    return int(flops_e18 * 1e18 / (flops_per_sample_train * global_batch_size) / nodes_num)


def max_step_from_tokens(tokens_e9: float, block_size: int,
                         global_batch_size: int, nodes_num: int) -> int:
    """--tokens is in units of 1e9 tokens. Every sample is exactly block_size tokens
    (the dataset is packed), so tokens/step = block_size * global_batch_size * nodes_num --
    the same samples-per-step convention max_step_from_flops uses."""
    return int(tokens_e9 * 1e9 / (block_size * global_batch_size) / nodes_num)


def format_report(method: str, parts: dict, fwd: float, old_6nd: float,
                  max_step: int, max_step_6nd: Optional[int], budget: str = "") -> str:
    """The single startup-print format shared by all four training scripts.

    Key-matching rules must stay aligned with the parts dicts above: attention terms are
    those whose key contains "attn" or ends with "_self"; head terms contain "head". A matmul
    term whose key happens to contain "cross" therefore has to avoid "attn" in its name, or it
    would be counted into the attention fraction.

    max_step_6nd=None means the step count was NOT derived from a FLOP budget (e.g. a
    token budget), so the 6ND step-count cross-reference is dropped; `budget` is an extra
    line describing where max_step came from.
    """
    train = 3.0 * fwd
    attn = sum(v for k, v in parts.items() if "attn" in k or k.endswith("_self"))
    heads = sum(v for k, v in parts.items() if "head" in k)
    lines = [f"[FLOP accounting: {method}] per-sample forward parts (real FLOPs):"]
    width = max(len(k) for k in parts)
    for k, v in parts.items():
        lines.append(f"  {k:<{width}}  {v:.3e}")
    lines += [
        f"  {'forward total':<{width}}  {fwd:.3e}",
        f"  attn frac {attn / fwd:.4f}   head frac {heads / fwd:.4f}",
        f"  train/sample (3x)  {train:.3e}   max_step={max_step}",
    ]
    if max_step_6nd is None:
        lines.append(f"  [old 6ND] /sample  {old_6nd:.3e}   (new/old={train / old_6nd:.4f})")
    else:
        lines.append(
            f"  [old 6ND] /sample  {old_6nd:.3e}   max_step={max_step_6nd}"
            f"   (new/old={train / old_6nd:.4f}, max_step x{max_step / max(max_step_6nd, 1):.4f})"
        )
    if budget:
        lines.append(f"  budget: {budget}")
    return "\n".join(lines)
