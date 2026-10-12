'''
lm-evaluation-harness entry point for E2D2 (block-autoregressive encoder-decoder MDM).

Mirrors evaluate_mdm.py (MDM) / evaluate_ar.py (ARM) so the three arms can be compared on
the same zero-shot tasks. The one structural difference is how the likelihood is obtained.

WHY E2D2 GETS AN EXACT LIKELIHOOD CHEAPLY. Its cross-attention mask is a STRICT block prefix
(lit_gpt/e2d2_model.py:293-304 -- decoder block k attends encoder positions [0, k*b) only) and
its decoder self-attention is block-diagonal (:281-291). So log p(x) = sum_k log p(x^k | x^{<k})
factorizes, and -- crucially -- all n/b blocks are CONDITIONALLY INDEPENDENT given a clean
enc_idx. One position per block can therefore be revealed per forward pass, so an exact
conditional log p(target | prefix) costs `mtp_window` passes regardless of sequence length,
versus len(target) passes for MDM's nll_type='chain_rule'.

This is the same argument duel_validate_sweep rests on (pretrain/train_e2d2.py:667-765); the
scorer below is that loop restricted from "score every position" to "score the target only".

Feeding the FULL clean sequence as enc_idx does not leak the target: a scored position in block
k is conditioned on exactly x^{<k} plus the already-revealed positions of block k.

Usage (see also the eval_mdm.sh / eval/scripts/eval_mdm_1e20.sh command shapes):

    python evaluate_e2d2.py --tasks hellaswag --model e2d2 --batch_size 128 \
        --model_args model_name=666,ckpt_path=...,mtp_window=4,n_dec_layers=14,nll_type='duel'

model_name / n_dec_layers / mtp_window are inferred from the checkpoint path when it carries
train_e2d2.py's hp_name (e.g. e2d2-666M-14dec-w4-100.0flop-lowvar); explicit model_args always
win.
'''
import contextlib
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
# resolve when it is run as `python eval/evaluate_e2d2.py` from the repo root.
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from lit_gpt.config import Config
from lit_gpt.e2d2_model import E2D2Model


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------------
# Checkpoint / architecture resolution
# --------------------------------------------------------------------------------------------

# pretrain/train_e2d2.py's hp_name:
#   e2d2-{N}M-{nd}dec-w{b}-{budget}flop|tok[-lowvar][-gbs{n}][-lr{x}]
_HP_RE = re.compile(r'^e2d2-(\d+)M-(\d+)dec-w(\d+)-')


def parse_hp_name(ckpt_path):
    """Recover the architecture flags from train_e2d2.py's hp_name, or {} if unrecognized.

    Checked against both the parent directory name (the training out_dir) and the file stem,
    so both `.../e2d2-666M-14dec-w4-100.0flop-lowvar/iter-027637-ckpt.safetensors` and a
    flattened `models/e2d2-666M-14dec-w4-100.0flop-lowvar.safetensors` resolve.
    """
    p = Path(str(ckpt_path))
    for cand in (p.parent.name, p.stem, p.name):
        m = _HP_RE.match(cand)
        if m:
            return {
                'model_name': int(m.group(1)),
                'n_dec_layers': int(m.group(2)),
                'mtp_window': int(m.group(3)),
                'hp_name': cand,
            }
    return {}


def n_dec_layers_from_state_dict(sd):
    """Decoder depth implied by the checkpoint's decoder_blocks.{i}.* keys."""
    idx = {int(k.split('.')[1]) for k in sd if k.startswith('decoder_blocks.')}
    if not idx:
        raise ValueError('no decoder_blocks.* keys: is this an E2D2 checkpoint?')
    return max(idx) + 1


# --------------------------------------------------------------------------------------------
# Sequence layout
# --------------------------------------------------------------------------------------------

def padded_length(L, mtp_window, padding, max_length):
    """E2D2Model.forward hard-asserts n % mtp_window == 0 (e2d2_model.py:229), so unlike MDM
    the padding is not optional -- only its size is.

    padding=False pads to the next multiple of b (the cheap default). padding=True pads to a
    fixed max_length, reproducing evaluate_mdm.py:100-104. The two give IDENTICAL scores:
    whole extra MASK blocks cannot influence an earlier block, because the cross mask is a
    strict block prefix and the decoder self mask is block-diagonal.
    """
    b = mtp_window
    if padding:
        assert max_length % b == 0, f'max_length {max_length} not divisible by mtp_window {b}'
        n = max_length
    else:
        n = ((L + b - 1) // b) * b
    assert L <= n, f'sequence length {L} exceeds padded length {n}'
    return n


def build_inputs(prefix, target, n, mask_id, device):
    """(enc_idx, dec_idx, to_score, prompt_index), all length n.

    enc_idx  -- the full CLEAN sequence, MASK on the pad tail. Never mutated.
    dec_idx  -- clean prefix, MASK on the scored span and on the pad tail.
    to_score -- True on the target positions only.

    The MASK pad is harmless: a pad position sits strictly after every real token, so the
    encoder rows it can contaminate (its own block, under the block-causal encoder mask) are
    only ever read by decoder blocks k with k*b > that position -- blocks that are themselves
    entirely pad and never scored. The one real effect is that the last content block's
    decoder self-attention sees a few extra MASK positions, which is in-distribution.
    """
    lp, lt = len(prefix), len(target)
    L = lp + lt
    enc_idx = torch.full((1, n), mask_id, dtype=torch.long, device=device)
    enc_idx[0, :lp] = prefix.to(device=device, dtype=torch.long)
    enc_idx[0, lp:L] = target.to(device=device, dtype=torch.long)

    to_score = torch.zeros((1, n), dtype=torch.bool, device=device)
    to_score[0, lp:L] = True

    prompt_index = torch.zeros(n, dtype=torch.bool, device=device)
    prompt_index[:lp] = True

    dec_idx = enc_idx.clone()
    dec_idx[to_score] = mask_id
    return enc_idx, dec_idx, to_score, prompt_index


# --------------------------------------------------------------------------------------------
# Likelihood estimators. Kept as free functions taking a `logits_fn(enc_idx, dec_idx)` so the
# numeric core is unit-testable on CPU without constructing the whole harness (CFG, autocast
# and batching live in the class).
# --------------------------------------------------------------------------------------------

def duel_target_nll(logits_fn, enc_idx, to_score, mtp_window, mask_id, padded_vocab_size,
                    duel_alg='greedy'):
    """EXACT summed NLL of the scored positions, in <= mtp_window forward passes.

    Two nested factorizations. Across blocks, the strict-prefix cross mask gives
    log p(target | prefix) = sum_k log p(target n block k | x^{<k}, prefix n block k), and
    every block's conditioning set is complete after ONE pass over the clean enc_idx -- which
    is why all blocks advance together. Within a block, revealing one position at a time and
    summing the true-token conditional log-probs is the any-order chain rule, so the sum
    telescopes exactly. The reveal is TEACHER-FORCED (the true token, not the argmax); that is
    what makes it a likelihood of `target` rather than of a self-generated path.

    `duel_alg` matters: masked diffusion is not a consistent joint, so different within-block
    reveal orders give different values. 'greedy' matches train_e2d2.py:214's duel_alg, keeping
    these numbers on the same footing as the metric/duel_nll logged for every training run.
    """
    b = mtp_window
    B, n = enc_idx.shape
    assert B == 1, 'one request at a time (mirrors evaluate_mdm.py)'
    assert n % b == 0
    n_blocks = n // b
    device = enc_idx.device

    # lm_head emits padded_vocab_size logits (0..V-1) but MASK_ID == V (the extra wte row,
    # e2d2_model.py:180-181). duel_validate_sweep gathers with real training data so it never
    # hits this; here enc_idx carries MASK_ID on the pad tail and an ungated gather would be an
    # out-of-bounds index. Pad positions are never selected, so the clamped value is never read.
    true_tok = enc_idx.clamp(max=padded_vocab_size - 1)

    dec_idx = enc_idx.clone()
    dec_idx[to_score] = mask_id
    remaining = to_score.clone()
    block_base = (torch.arange(n_blocks, device=device) * b).view(1, n_blocks)
    pos = torch.arange(n, device=device).view(1, n)

    # Per-request, not always b: blocks hold different numbers of scored positions (a block
    # fully inside the prefix holds none), and an exhausted block is inert for later steps.
    n_steps = int(remaining.view(1, n_blocks, b).sum(-1).max().item())

    ll = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(n_steps):
        # .float() before logsumexp, as train_e2d2.py:721: the logits arrive in bf16 from the
        # autocast region and accumulating lse in bf16 visibly drifts the NLL.
        logits = logits_fn(enc_idx, dec_idx).float()
        lse = torch.logsumexp(logits, dim=-1)
        true_lp = torch.gather(logits, -1, true_tok.unsqueeze(-1)).squeeze(-1) - lse

        if duel_alg == 'greedy':
            score = logits.max(dim=-1).values - lse
        elif duel_alg == 'left_to_right':
            score = (-pos).float()
        else:
            raise NotImplementedError(duel_alg)
        score = score.masked_fill(~remaining, float('-inf'))

        sel = torch.topk(score.view(1, n_blocks, b), 1, dim=-1).indices.view(1, n_blocks)
        sel = sel + block_base
        # A block with nothing left has an all -inf row, so topk returns an ARBITRARY index in
        # it. Dropping those from the ll sum is mandatory -- without it prefix tokens get
        # counted into the target likelihood, silently and with no error. (The scatters below
        # happen to be idempotent on dead picks, but gate them too rather than rely on that.)
        live = remaining.view(1, n_blocks, b).any(-1)
        ll = ll + torch.gather(true_lp, -1, sel)[live].sum()

        sel_live = sel[live].view(1, -1)
        dec_idx.scatter_(-1, sel_live, true_tok.gather(-1, sel_live))
        remaining.scatter_(-1, sel_live, torch.zeros_like(sel_live, dtype=torch.bool))

    return -ll


def block_forward_process(batch, mtp_window, eps=1e-3):
    """(mask_indices, p_mask) with E2D2's PER-BLOCK noise level.

    Verbatim pretrain/train_e2d2.py:263-272 with is_training=False. Skipping
    ensure_one_mask_per_block is load-bearing: it is a training-side device for nonzero
    gradients and would bias the estimator. validate() does the same (train_e2d2.py:633).
    """
    B, l = batch.shape
    n_blocks = l // mtp_window
    t = torch.rand((B, n_blocks), device=batch.device).repeat_interleave(mtp_window, dim=1)
    p_mask = (1 - eps) * t + eps
    mask_indices = torch.rand((B, l), device=batch.device) < p_mask
    return mask_indices, p_mask


def mc_target_nll(logits_fn, enc_idx, to_score, mtp_window, mask_id, padded_vocab_size,
                  mc_num, batch_size, eps=1e-3):
    """Monte-Carlo weighted-ELBO summed NLL -- an UPPER bound on the exact NLL.

    The E2D2 analogue of evaluate_mdm.py:151-175: enc_idx stays clean and full, only the
    scored span of dec_idx is noised, and the per-token CE is reweighted by 1/p_mask. eps and
    the reweighting match validate() (train_e2d2.py:643-644), so this is on the same scale as
    the metric/val_loss logged during training. Exists for comparability with the MDM runs,
    which have no cheap exact estimator; prefer nll_type='duel'.
    """
    assert mc_num % batch_size == 0
    n = enc_idx.shape[1]
    true_tok = enc_idx.clamp(max=padded_vocab_size - 1)
    enc_b = enc_idx.expand(batch_size, n).contiguous()
    true_b = true_tok.expand(batch_size, n).contiguous()
    score_b = to_score.expand(batch_size, n)

    acc = []
    for _ in range(mc_num // batch_size):
        mask_indices, p_mask = block_forward_process(enc_b, mtp_window, eps)
        # Noise the target span only; the prefix conditions and the pad tail is already MASK.
        scored = mask_indices & score_b
        dec_b = enc_b.clone()
        dec_b[scored] = mask_id
        logits = logits_fn(enc_b, dec_b).float()
        loss = F.cross_entropy(logits[scored], true_b[scored], reduction='none')
        acc.append(((loss / p_mask[scored]).sum() / batch_size).cpu())
    return sum(acc) / len(acc)


def greedy_block_decode(logits_fn, seq, to_gen, mtp_window, mask_id):
    """Block-autoregressive greedy decode; returns the completed sequence.

    Unlike duel_target_nll this CANNOT be parallel across blocks: the encoder must see the
    generated prefix, so blocks are decoded in order. Passing `seq` (MASK beyond the committed
    prefix) as enc_idx is safe for the strict-prefix reason -- block k reads encoder positions
    [0, k*b), all of which are already committed by the time it runs. Within a block, the
    single highest-confidence argmax is committed per pass, as evaluate_mdm.py:187-196.

    Cost is ~len(target) forwards, hence greddy=False by default.
    """
    b = mtp_window
    B, n = seq.shape
    assert B == 1
    assert n % b == 0
    seq, to_gen = seq.clone(), to_gen.clone()
    for k in range(n // b):
        s, e = k * b, (k + 1) * b
        for _ in range(int(to_gen[0, s:e].sum().item())):
            logits = logits_fn(seq, seq)[0, s:e].float()
            lsm = torch.log_softmax(logits, dim=-1)
            conf, x0 = lsm.max(dim=-1)
            conf = conf.masked_fill(~to_gen[0, s:e], float('-inf'))
            j = int(torch.argmax(conf).item())
            seq[0, s + j] = x0[j]
            to_gen[0, s + j] = False
    return seq


# --------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------

@register_model("e2d2")
class E2D2EvalHarness(LM):
    def __init__(
            self,
            model_name=None,
            ckpt_path=None,
            mtp_window=None,
            n_dec_layers=None,
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
        assert nll_type in ['duel', 'mc'], nll_type
        assert duel_alg in ['greedy', 'left_to_right'], duel_alg
        assert ckpt_path is not None, 'ckpt_path is required'

        # --- Architecture resolution. E2D2 is NOT fully determined by the config name:
        # mtp_window and n_dec_layers are runtime attributes on Config. Getting mtp_window
        # wrong still runs and silently produces wrong numbers, so resolve it explicitly and
        # report what was used.
        inferred = parse_hp_name(ckpt_path)
        model_name = model_name if model_name is not None else inferred.get('model_name')
        mtp_window = mtp_window if mtp_window is not None else inferred.get('mtp_window')
        n_dec_layers = n_dec_layers if n_dec_layers is not None else inferred.get('n_dec_layers')

        if model_name is None:
            raise ValueError('model_name could not be inferred from ckpt_path; pass it explicitly')
        if mtp_window is None:
            raise ValueError(
                'mtp_window could not be inferred from ckpt_path; pass it explicitly. There is '
                'no safe default: Config.mtp_window is 1 and train_e2d2.py --mtp_window is 2, '
                'and a wrong value produces wrong numbers rather than an error.')
        mtp_window, n_dec_layers = int(mtp_window), (None if n_dec_layers is None else int(n_dec_layers))

        sd = load_file(str(ckpt_path))
        nd_ckpt = n_dec_layers_from_state_dict(sd)
        if n_dec_layers is None:
            n_dec_layers = nd_ckpt
        elif n_dec_layers != nd_ckpt:
            raise ValueError(
                f'n_dec_layers={n_dec_layers} but the checkpoint has {nd_ckpt} decoder blocks')

        config = Config.from_name(f'Diff_LLaMA_{model_name}M')
        config.n_dec_layers = n_dec_layers
        config.mtp_window = mtp_window

        self.model = E2D2Model(config).to(device)
        self.model.load_state_dict(sd)
        self.model.eval()

        if mask_id is None:
            mask_id = config.padded_vocab_size
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
            # Only the MC path uses batch_size; asserting unconditionally would fail the duel
            # path over an argument it never reads.
            assert self.mc_num % self.batch_size == 0
        self.padding = bool(padding)
        self.nll_type = nll_type
        self.duel_alg = duel_alg
        self.greddy = bool(greddy)
        self.cfg = float(cfg)
        self.device = torch.device(device)

        print(f'[e2d2] Diff_LLaMA_{model_name}M n_layer={config.n_layer} '
              f'n_dec={n_dec_layers} n_enc={config.n_layer - n_dec_layers} b={mtp_window} '
              f'| nll_type={nll_type} '
              f'duel_alg={duel_alg} cfg={self.cfg} padding={self.padding} '
              f'greddy={self.greddy}'
              + (f" | hp_name={inferred['hp_name']}" if 'hp_name' in inferred else ''))

    def _autocast(self):
        if self.device.type == 'cuda':
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def get_logits(self, enc_idx, dec_idx, prompt_index):
        '''
        prompt_index : 1D bool tensor, length = enc_idx.shape[1], True on prefix positions
        '''
        if self.cfg > 0.:
            assert len(prompt_index) == enc_idx.shape[1]
            # The prefix reaches a scored position ONLY through the encoder memory, so the
            # unconditional branch has to mask it in enc_idx too -- masking dec_idx alone (the
            # MDM analogue, evaluate_mdm.py:96-98) would leave the conditioning nearly intact.
            un_enc = enc_idx.clone()
            un_dec = dec_idx.clone()
            un_enc[:, prompt_index] = self.mask_id
            un_dec[:, prompt_index] = self.mask_id
            enc_idx = torch.cat([enc_idx, un_enc])
            dec_idx = torch.cat([dec_idx, un_dec])

        with self._autocast():
            logits = self.model(enc_idx, dec_idx)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits

    def _logits_fn(self, prompt_index):
        return lambda enc_idx, dec_idx: self.get_logits(enc_idx, dec_idx, prompt_index)

    @torch.no_grad()
    def _eval_target_nll_duel(self, prefix, target):
        n = padded_length(len(prefix) + len(target), self.mtp_window, self.padding,
                          self.max_length)
        enc_idx, _, to_score, prompt_index = build_inputs(
            prefix, target, n, self.mask_id, self.device)
        return duel_target_nll(
            self._logits_fn(prompt_index), enc_idx, to_score, self.mtp_window, self.mask_id,
            self.model.config.padded_vocab_size, duel_alg=self.duel_alg).cpu()

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        n = padded_length(len(prefix) + len(target), self.mtp_window, self.padding,
                          self.max_length)
        enc_idx, _, to_score, prompt_index = build_inputs(
            prefix, target, n, self.mask_id, self.device)
        return mc_target_nll(
            self._logits_fn(prompt_index), enc_idx, to_score, self.mtp_window, self.mask_id,
            self.model.config.padded_vocab_size, self.mc_num, self.batch_size)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
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


def patch_bootstrap_stderr():
    '''Bound lm_eval's bootstrap worker pool to something the container can actually host.

    lm_eval 0.4.4's api/metrics.py:462 does `mp.Pool(mp.cpu_count())`, and cpu_count() is
    the HOST core count -- 192 on the machines these runs used -- not the container's cgroup
    cpu allocation. sched_getaffinity reports the host count too, so there is nothing to read
    the real limit off. Forking that many copies of a process that already holds CUDA and a
    loaded model
    exhausts the fd table (the parent was at 14.6k fds of its 65536 when it died), and
    multiprocessing's resource_sharer then spins forever in

        connection.accept() -> socket.fromfd() -> dup() -> OSError: [Errno 24]

    which is a DEADLOCK, not a crash: the job hangs at 0% GPU with the results table
    unprinted. This is what cost the first pass of eval_e2d2_*.yaml its lambada_standard
    numbers -- the forwards had all completed. Only the metrics whose stderr has no
    closed form take this path (perplexity here; acc/acc_norm use mean_stderr), which is
    why the multiple-choice runs survived and lambada_standard did not.

    Patching the module attribute is enough: stderr_for_metric hands back
    `lambda x: bootstrap_stderr(metric, x, iters=...)`, which resolves the name in this
    module's namespace at call time.

    E2D2_BOOTSTRAP_ITERS overrides the iteration count; 0 skips the bootstrap and reports
    the same "N/A" that lm_eval itself stores for a metric with no stderr function.
    '''
    import multiprocessing as mp
    import os

    import lm_eval.api.metrics as _metrics

    max_workers = int(os.environ.get("E2D2_BOOTSTRAP_WORKERS", 8))

    def bootstrap_stderr(f, xs, iters):
        iters = int(os.environ.get("E2D2_BOOTSTRAP_ITERS", iters))
        if iters <= 0:
            return "N/A"

        chunk_size = min(1000, iters)
        n_chunks = max(1, iters // chunk_size)
        tasks = [(i, xs) for i in range(n_chunks)]
        worker = _metrics._bootstrap_internal(f, chunk_size)

        print(f"bootstrapping for stddev: {f.__name__} "
              f"({iters} iters, {min(max_workers, n_chunks)} workers)")
        res = []
        # A pool is only worth its fork cost when there is more than one chunk to hand
        # out; at iters <= 1000 the serial path is both faster and unable to deadlock.
        if n_chunks == 1 or max_workers <= 1:
            for t in tqdm(tasks, total=n_chunks):
                res.extend(worker(t))
        else:
            with mp.Pool(min(max_workers, n_chunks)) as pool:
                for chunk in tqdm(pool.imap(worker, tasks), total=n_chunks):
                    res.extend(chunk)
        return _metrics.sample_stddev(res)

    _metrics.bootstrap_stderr = bootstrap_stderr


def patch_extra_acc_norm():
    '''Report acc_norm for tasks whose lm_eval yaml declares only acc (`race`).

    WHY. For output_type=multiple_choice, api/task.py:1428 ALWAYS computes
    acc_norm = argmax(lls / completion_len) == gold, but :1440 only emits it when "acc_norm"
    is in self._metric_fn_list -- which is populated from the yaml's metric_list. Of the ten
    tasks scored here, only race's yaml (lm_eval/tasks/race/race.yaml) lists acc alone, so
    race is the one task with no acc_norm row in the results table. That is a reporting gate,
    not a modelling limitation: race is multiple_choice over the four option TEXTS
    (preprocess_race.doc_to_choice), which is exactly the unequal-length setting acc_norm
    exists for, and is why the other five text-choice tasks report it.

    WHY A RUNTIME PATCH rather than a task yaml under --include_path. A yaml can only add a
    metric by redefining the task, and `include:` needs a path into the installed lm_eval
    package (utils.py:449 resolves a non-file include relative to OUR yaml dir, not to the
    package), so it would mean vendoring race.yaml + preprocess_race.py. A vendored copy that
    drifts from the installed one changes the CONTEXTS, i.e. silently invalidates the acc
    numbers already collected in eval/results/. Adding a metric to the task lm_eval itself
    built cannot: the requests, the prompts and the loglikelihoods are untouched, so `acc` is
    reproduced exactly and doubles as a check that nothing else moved.

    Set E2D2_ACC_NORM_TASKS="" to disable, or to a comma-separated task list to extend.
    '''
    import os

    import lm_eval.evaluator as _evaluator
    from lm_eval.api.registry import get_metric, get_metric_aggregation, is_higher_better

    wanted = [t for t in os.environ.get("E2D2_ACC_NORM_TASKS", "race").split(",") if t]
    if not wanted:
        return

    def add(name, task_obj):
        # Only meaningful for multiple_choice: it is the sole branch of process_results that
        # computes an acc_norm, so on any other output_type the metric would be registered,
        # never produced, and reported as a missing key at aggregation time.
        if getattr(task_obj, "OUTPUT_TYPE", None) != "multiple_choice":
            raise ValueError(
                f"E2D2_ACC_NORM_TASKS names {name!r}, whose output_type is "
                f"{getattr(task_obj, 'OUTPUT_TYPE', None)!r}; acc_norm is only computed for "
                f"multiple_choice tasks")
        if "acc_norm" in task_obj._metric_fn_list:
            return False
        task_obj._metric_fn_list["acc_norm"] = get_metric("acc_norm")
        task_obj._metric_fn_kwargs["acc_norm"] = {}
        task_obj._aggregation_list["acc_norm"] = get_metric_aggregation("acc_norm")
        task_obj._higher_is_better["acc_norm"] = is_higher_better("acc_norm")
        # Keep the config in step with the four dicts: dump_config() feeds the results JSON
        # and TaskOutput.task_config, so a config that still claims `acc` only would make the
        # saved metadata disagree with the table it sits next to.
        task_obj.config.metric_list = list(task_obj.config.metric_list or []) + [
            {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
        ]
        return True

    def walk(task_dict):
        for name, obj in task_dict.items():
            if isinstance(obj, dict):
                walk(obj)
            elif name in wanted:
                # A group entry is (group_name, task); a plain task is the task itself.
                task_obj = obj[1] if isinstance(obj, tuple) else obj
                if task_obj and add(name, task_obj):
                    print(f"[e2d2] added acc_norm to task {name!r}")

    # get_task_dict is imported into evaluator's namespace (evaluator.py:31) and called by
    # that name (:232), so rebinding the attribute is enough; the tasks are patched after
    # lm_eval has fully built them and before simple_evaluate reads their metric lists.
    _inner = _evaluator.get_task_dict

    def get_task_dict(*args, **kwargs):
        task_dict = _inner(*args, **kwargs)
        walk(task_dict)
        return task_dict

    _evaluator.get_task_dict = get_task_dict


if __name__ == "__main__":
    set_seed(1234)
    patch_bootstrap_stderr()
    patch_extra_acc_norm()
    cli_evaluate()
