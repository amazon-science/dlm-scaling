'''
lm-evaluation-harness entry point for BD3LM (block discrete denoising diffusion LM).

Mirrors evaluate_e2d2.py (block-AR encoder-decoder MDM), evaluate_mdm.py (MDM) and
evaluate_ar.py (ARM) so all four arms can be compared on the same zero-shot tasks.

WHY THE LIKELIHOOD CORE IS IMPORTED FROM evaluate_e2d2 RATHER THAN REIMPLEMENTED. Those six
functions never touch a model: they take a `logits_fn(clean, noisy)` closure plus the block
geometry, so their correctness rests on five properties of whatever that closure wraps --

  P1  signature      two (B, n) int64 tensors -> (B, n, V) logits aligned to the SECOND one
                     (bd3lm_model.py:90 returns lm_head(x[:, n:, :]), the noisy half only)
  P2  alignment      n % mtp_window == 0 is a hard requirement  (bd3lm_model.py:72)
  P3  cross-block    logits in block k depend on `clean` only through blocks < k
  P4  within-block   logits in block k depend on `noisy` only through block k
  P5  pad hazard     mask_id == padded_vocab_size embeds but is NOT a legal logit index

-- and BD3LM satisfies all five. P3/P4 hold because its 4-quadrant mask
(bd3lm_model.py:98-101) is SET-IDENTICAL to E2D2's three masks:

    BD3LM quadrant                        E2D2 equivalent (e2d2_model.py)
    TL clean->clean   kq >= kk            _build_encoder_mask :270  block k sees blocks 0..k
    TR clean->noisy   all masked          separate towers: the encoder never sees dec_idx
    BL noisy->clean   kq >  kk            _build_cross_mask   :293  block k sees [0, k*b)
    BR noisy->noisy   kq == kk            _build_self_mask    :281  block-diagonal b x b

The parameterization differs (one shared tower here, two towers with a cross-attention module
there), which changes the NUMBERS but not the STRUCTURE the estimators depend on.

WHAT P3 BUYS. BL is strictly `>`, not `>=`, so a noisy row in block k reads clean rows in blocks
< k, and those rows -- under the block-causal TL quadrant -- themselves reach at most block k-1.
The argument holds across all layers because the mask is layer-invariant. Therefore

  (a) feeding the FULL clean sequence, target included, as clean_idx does not leak: a scored
      position in block k is conditioned on exactly x^{<k} plus block k's already-revealed
      positions; and
  (b) all n/b blocks are CONDITIONALLY INDEPENDENT given clean_idx, so one position per block can
      be revealed per forward pass and an exact conditional log p(target | prefix) costs
      <= mtp_window passes regardless of sequence length -- versus len(target) for MDM's
      nll_type='chain_rule'/'duel'.

This is the same argument pretrain/train_bd3lm.py:476 duel_validate_sweep rests on; the scorer
here is that loop restricted from "score every position" to "score the target only", plus the
`live` gate that loop does not need (there every block is full; here the blocks of one request
hold unequal numbers of scored positions).

The MC path is the one reuse that is EMPIRICAL rather than structural: mc_target_nll's noise
must match BD3LM's training. pretrain/train_bd3lm.py:181 forward_process and
pretrain/train_e2d2.py:263 forward_process are byte-identical (per-block t, eps=1e-3, the same
is_training branch), and evaluate_e2d2.py:229 block_forward_process is that with
is_training=False -- so they agree today.

Usage:

    python eval/evaluate_bd3lm.py --tasks hellaswag --model bd3lm \
        --model_args model_name=142,ckpt_path=...,mtp_window=4,nll_type='duel'

model_name / mtp_window are inferred from the checkpoint path when it carries train_bd3lm.py's
hp_name (e.g. bd3lm-142M-w4-10.0flop-lowvar); explicit model_args always win.
'''
import contextlib
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
from pathlib import Path

# This file lives in eval/, so the repo root has to be on sys.path for `import lit_gpt` to
# resolve when it is run as `python eval/evaluate_bd3lm.py` from the repo root.
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from lit_gpt.bd3lm_model import BD3LM
from lit_gpt.config import Config

# The likelihood core, unchanged. See the module docstring for why this is sound; the estimators
# are shared so that "the same functional at every b" is true by construction rather than by
# inspection, which is the entire point of the b-axis study. patch_* are re-exported so a caller
# can `from evaluate_bd3lm import patch_bootstrap_stderr` without knowing where it lives.
from evaluate_e2d2 import (  # noqa: F401  (patch_* are re-exported, not used here)
    block_forward_process,
    build_inputs,
    duel_target_nll,
    greedy_block_decode,
    mc_target_nll,
    padded_length,
    patch_bootstrap_stderr,
    patch_extra_acc_norm,
)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------------
# Checkpoint / architecture resolution
# --------------------------------------------------------------------------------------------

# pretrain/train_bd3lm.py:208-214:
#   bd3lm-{N}M-w{b}-{budget}flop[-lowvar][-gbs{n}][-lr{x}]
_HP_RE = re.compile(r'^bd3lm-(\d+)M-w(\d+)-')


def parse_hp_name(ckpt_path):
    """Recover model_name / mtp_window from train_bd3lm.py's hp_name, or {} if unrecognized.

    Checked against the parent directory name (the training out_dir) and the file stem, so both
    `.../bd3lm-142M-w4-10.0flop-lowvar/iter-008954-ckpt.safetensors` and a flattened
    `models/bd3lm-142M-w4-10.0flop-lowvar.safetensors` resolve.

    ⚠️ All four b-arms share ONE checkpoint root and are told apart only by the w{b} tag, so
    the directory name is the ONLY ground truth for b -- which is why __init__ prints what it
    resolved.
    """
    p = Path(str(ckpt_path))
    for cand in (p.parent.name, p.stem, p.name):
        m = _HP_RE.match(cand)
        if m:
            return {
                'model_name': int(m.group(1)),
                'mtp_window': int(m.group(2)),
                'hp_name': cand,
            }
    return {}


# --------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------

@register_model("bd3lm")
class BD3LMEvalHarness(LM):
    def __init__(
            self,
            model_name=None,
            ckpt_path=None,
            mtp_window=None,
            mask_id=None,
            max_length=2048,
            batch_size=32,
            mc_num=1024,
            padding=False,
            nll_type='duel',
            duel_alg='greedy',
            greddy=False,
            cfg=0.,
            device="cuda",
    ):
        super().__init__()
        # Before any checkpoint I/O, so a typo fails in a second rather than after a load.
        assert nll_type in ['duel', 'mc'], nll_type
        assert duel_alg in ['greedy', 'left_to_right'], duel_alg
        assert ckpt_path is not None, 'ckpt_path is required'

        # --- Architecture resolution. BD3LM is SIMPLER than E2D2 -- no n_dec_layers, no
        # encoder/decoder split, and bd3lm_model.py:46-51 has no conditional parameters -- and
        # that makes mtp_window MORE dangerous, not less: it is the only architecture flag not
        # implied by the weights, so there is no strict-load mismatch to catch a wrong value.
        # Config.mtp_window defaults to 1 and train_bd3lm.py --mtp_window defaults to 4, and a
        # wrong b runs clean while scoring a DIFFERENT factorization. Resolve it explicitly.
        inferred = parse_hp_name(ckpt_path)
        model_name = model_name if model_name is not None else inferred.get('model_name')
        mtp_window = mtp_window if mtp_window is not None else inferred.get('mtp_window')

        if model_name is None:
            raise ValueError('model_name could not be inferred from ckpt_path; pass it explicitly')
        if mtp_window is None:
            raise ValueError(
                'mtp_window could not be inferred from ckpt_path; pass it explicitly. There is '
                'no safe default: Config.mtp_window is 1 and train_bd3lm.py --mtp_window is 4, '
                'and a wrong value produces wrong numbers rather than an error.')
        mtp_window = int(mtp_window)

        config = Config.from_name(f'Diff_LLaMA_{model_name}M')
        # BEFORE construction: bd3lm_model.py:43 reads config.mtp_window in __init__.
        config.mtp_window = mtp_window

        self.model = BD3LM(config).to(device)
        self.model.load_state_dict(load_file(str(ckpt_path)))
        self.model.eval()

        if mask_id is None:
            mask_id = config.padded_vocab_size
        # wte has padded_vocab_size + 1 rows so MASK embeds directly (bd3lm_model.py:46) while
        # lm_head emits padded_vocab_size logits (:47). Any other mask_id would index a real token.
        assert mask_id == config.padded_vocab_size, (
            f'mask_id {mask_id} != padded_vocab_size {config.padded_vocab_size}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

        self.mask_id = mask_id
        self.mtp_window = mtp_window
        self.max_length = min(int(max_length), config.block_size)
        self.batch_size = int(batch_size)
        self.mc_num = int(mc_num)
        if nll_type == 'mc':
            # Only the MC path reads batch_size; asserting unconditionally would fail the duel
            # path over an argument it never touches.
            assert self.mc_num % self.batch_size == 0
        self.padding = bool(padding)
        self.nll_type = nll_type
        self.duel_alg = duel_alg
        self.greddy = bool(greddy)
        self.cfg = float(cfg)
        self.device = torch.device(device)

        print(f'[bd3lm] Diff_LLaMA_{model_name}M n_layer={config.n_layer} b={mtp_window} '
              f'| nll_type={nll_type} duel_alg={duel_alg} cfg={self.cfg} '
              f'padding={self.padding} greddy={self.greddy}'
              + (f" | hp_name={inferred['hp_name']}" if 'hp_name' in inferred else ''))

    def _autocast(self):
        if self.device.type == 'cuda':
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def get_logits(self, clean_idx, noisy_idx, prompt_index):
        '''
        prompt_index : 1D bool tensor, length = clean_idx.shape[1], True on prefix positions
        '''
        if self.cfg > 0.:
            assert len(prompt_index) == clean_idx.shape[1]
            # BOTH halves, as evaluate_e2d2.py:421-431 and unlike evaluate_mdm.py:96-98. The
            # prefix reaches a scored position by two routes: earlier CLEAN blocks (BL) and the
            # prefix tokens sharing the scored position's own block in the NOISY half (BR, where
            # build_inputs deliberately leaves them unmasked so they condition). Masking one half
            # would leave the conditioning nearly intact.
            un_clean = clean_idx.clone()
            un_noisy = noisy_idx.clone()
            un_clean[:, prompt_index] = self.mask_id
            un_noisy[:, prompt_index] = self.mask_id
            clean_idx = torch.cat([clean_idx, un_clean])
            noisy_idx = torch.cat([noisy_idx, un_noisy])

        with self._autocast():
            logits = self.model(clean_idx, noisy_idx)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits

    def _logits_fn(self, prompt_index):
        """The seam the shared estimators consume: (clean, noisy) -> logits, CFG and autocast
        folded in. Matches E2D2's logits_fn contract exactly -- see the module docstring, P1."""
        return lambda clean_idx, noisy_idx: self.get_logits(clean_idx, noisy_idx, prompt_index)

    @torch.no_grad()
    def _eval_target_nll_duel(self, prefix, target):
        n = padded_length(len(prefix) + len(target), self.mtp_window, self.padding,
                          self.max_length)
        clean_idx, _, to_score, prompt_index = build_inputs(
            prefix, target, n, self.mask_id, self.device)
        return duel_target_nll(
            self._logits_fn(prompt_index), clean_idx, to_score, self.mtp_window, self.mask_id,
            self.model.config.padded_vocab_size, duel_alg=self.duel_alg).cpu()

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        n = padded_length(len(prefix) + len(target), self.mtp_window, self.padding,
                          self.max_length)
        clean_idx, _, to_score, prompt_index = build_inputs(
            prefix, target, n, self.mask_id, self.device)
        return mc_target_nll(
            self._logits_fn(prompt_index), clean_idx, to_score, self.mtp_window, self.mask_id,
            self.model.config.padded_vocab_size, self.mc_num, self.batch_size)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        """The exact-match bit lambada's `acc` is scored from.

        lambada goes through loglikelihood(), never generate_until(): lm_eval reads its
        perplexity off the first tuple element and its acc off the second, which is this bool.
        So this IS the generation path -- block-autoregressive, and unlike the scorers it cannot
        be parallel across blocks, since block k's clean context must contain what block k-1
        generated. Costs ~len(target) forwards, hence greddy=False by default.
        """
        if not self.greddy:
            return False

        lp, L = len(prefix), len(prefix) + len(target)
        n = padded_length(L, self.mtp_window, self.padding, self.max_length)
        seq = torch.full((1, n), self.mask_id, dtype=torch.long, device=self.device)
        seq[0, :lp] = prefix.to(device=self.device, dtype=torch.long)
        to_gen = torch.zeros((1, n), dtype=torch.bool, device=self.device)
        to_gen[0, lp:L] = True
        prompt_index = torch.zeros(n, dtype=torch.bool, device=self.device)
        prompt_index[:lp] = True

        seq = greedy_block_decode(self._logits_fn(prompt_index), seq, to_gen, self.mtp_window,
                                  self.mask_id)
        correct = seq[0, lp:L].cpu() == target.cpu()
        return bool(torch.all(correct))

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= self.max_length

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'duel':
                    ll = -self._eval_target_nll_duel(prefix, target)
                elif self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    def generate_until(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    set_seed(1234)
    # patch_bootstrap_stderr is not optional for lambada_standard: without it lm_eval 0.4.4
    # deadlocks after every forward has completed.
    patch_bootstrap_stderr()
    patch_extra_acc_norm()
    cli_evaluate()
