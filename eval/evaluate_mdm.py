'''
This file is inspired by the code provided by the author of https://arxiv.org/abs/2406.11473
'''
import torch
import contextlib
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer
import sys
from pathlib import Path

# This file lives in eval/, so the repo root has to be on sys.path for `import lit_gpt` to
# resolve when it is run as `python eval/evaluate_mdm.py` from the repo root.
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def duel_target_nll(logits_fn, seq, to_score, mask_id, duel_alg='greedy'):
    '''Exact conditional NLL of the scored span under an unmasking order: -sum_t log p(x_s(t) | z_t).

    This is the estimator of pretrain/train_mdm.py's duel_validate_sweep restricted to a
    target span, i.e. the conditional form of the DUEL number that train_mdm.py logs as
    metric/duel_nll and that the paper reports as DUEL NLL. The reveal order s is chosen by duel_alg
    and the token revealed is the TRUE one (teacher forcing), so s is measurable w.r.t. the
    already-revealed prefix and the telescoping product is a properly normalised density: EXACT,
    not a bound, and seed-independent. It is NOT the diffusion ELBO -- the two must never be
    mixed inside one number.

    duel_alg is part of the estimator's definition, not an implementation detail: a masked model's
    conditionals are not a consistent joint, so different reveal orders give different values
    (evaluate_e2d2.py). 'greedy' (top-1 confidence) matches pretrain/train_mdm.py and is the
    order every DUEL number in the paper uses. 'left_to_right' is the
    positional order, which makes this function reproduce _eval_target_nll_ar (nll_type=
    'chain_rule') exactly -- that equality is the regression anchor, not a production setting,
    because chain_rule computes the same order in ceil(l2/batch_size) BATCHED forwards.

    Cost is exactly to_score.sum() forwards, one revealed position each. There is no cross-block
    parallelism to exploit as in evaluate_e2d2.py:158-227: TransEncoder has no block structure
    (lit_gpt/diffmodel.py:55-98), so the greedy order is inherently sequential at batch 1.

    logits_fn  z (1, L) int64 -> logits (1, L, V). Owns CFG, padding and autocast.
    seq        (1, L) int64, the CLEAN prefix+target, no pad tail.
    to_score   (1, L) bool, True exactly on the positions to score.
    returns    0-d fp32 CPU tensor, the SUMMED NLL over the scored positions.
    '''
    assert seq.shape == to_score.shape, (seq.shape, to_score.shape)
    assert seq.shape[0] == 1, 'one request at a time (mirrors loglikelihood()\'s loop)'
    # seq must hold real tokens only. mask_id == padded_vocab_size is a legal embedding input but
    # NOT a legal logit index (lit_gpt/diffmodel.py:28,31), so a MASK/pad tail in seq would make
    # the gather below index out of bounds. Padding is applied inside get_logits and trimmed off
    # again (:100-102, :112), which is what keeps this invariant true -- and is why no clamp is
    # needed here, unlike evaluate_e2d2.py:181-185.
    assert int(seq.max()) < mask_id, 'seq must be clean tokens, no MASK / pad tail'

    L = seq.shape[1]
    z = seq.clone()
    z[to_score] = mask_id
    remaining = to_score.clone()
    pos = torch.arange(L, device=seq.device).view(1, L)

    ll = torch.zeros((), dtype=torch.float32)
    for _ in range(int(remaining.sum().item())):
        # .float() BEFORE logsumexp: accumulating the lse in bf16 visibly drifts the NLL
        # (pretrain/train_mdm.py's duel_validate_sweep, evaluate_e2d2.py:201-202).
        logits = logits_fn(z).float()
        lse = torch.logsumexp(logits, dim=-1)
        true_lp = torch.gather(logits, -1, seq.unsqueeze(-1)).squeeze(-1) - lse

        if duel_alg == 'greedy':
            score = logits.max(dim=-1).values - lse
        elif duel_alg == 'left_to_right':
            score = (-pos).float().expand_as(lse)
        else:
            raise NotImplementedError(duel_alg)
        # -inf on everything already revealed and on the whole prefix, so a position is never
        # scored twice and prefix tokens are never folded into the target's likelihood.
        score = score.masked_fill(~remaining, float('-inf'))

        sel = score.argmax(dim=-1, keepdim=True)                 # (1, 1)
        ll = ll + torch.gather(true_lp, -1, sel).sum().cpu()
        z.scatter_(-1, sel, torch.gather(seq, -1, sel))           # reveal the TRUE token
        remaining.scatter_(-1, sel, torch.zeros_like(sel, dtype=torch.bool))

    return -ll


@register_model("mdm")
class MDMEvalHarness(LM):
    def __init__(
            self,
            model_name="tiny",
            ckpt_path=None,
            mask_id=32000,
            max_length=2048,
            batch_size=32,
            mc_num=1024,
            padding=False,
            nll_type='mc',
            duel_alg='greedy',
            greddy=False,
            cfg=0.,
            device="cuda",
    ):
        super().__init__()
        assert nll_type in ['mc', 'chain_rule', 'duel'], nll_type
        assert duel_alg in ['greedy', 'left_to_right'], duel_alg

        model_name = f'Diff_LLaMA_{model_name}M'
        config = Config.from_name(model_name)
        self.model = TransEncoder(config).to(device)

        self.model.load_state_dict(load_file(ckpt_path))
        self.model.eval()

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        # Only the MC path consumes mc_num, so this cannot be asserted unconditionally: it would
        # fail the duel path over an argument duel never reads (evaluate_e2d2.py:393-396 documents
        # the same hazard). chain_rule does read batch_size -- it chunks the l2 triangular rows at
        # :132-134 -- but handles the remainder itself, so it needs no divisibility either.
        if nll_type == 'mc':
            assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.padding = padding
        self.nll_type = nll_type
        self.duel_alg = duel_alg
        self.greddy = greddy

        self.cfg = cfg
        self.device = torch.device(device)

    def _autocast(self):
        '''bf16 autocast on CUDA, a no-op anywhere else.

        Keyed on the device so the scorers can be exercised on CPU (torch.cuda.amp.autocast is
        not a no-op off CUDA -- it warns and, on some builds, errors). Mirrors
        evaluate_e2d2.py:411-414, which is what makes its numeric core unit-testable.
        '''
        if self.device.type == 'cuda':
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.mask_id, batch)

        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        if self.padding:
            input = torch.full((batch.size(0), 2048), self.mask_id, device=self.device)
            input[:, :batch.shape[1]] = batch
        else:
            input = batch

        with self._autocast():
            logits = self.model(input)

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        '''
        Utilize the chain rule to compute the likelihood
        We need to perform len(target) forward passes in parallel
        '''
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2

        prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        mask_index = torch.triu(mask_index)

        perturbed_[mask_index] = self.mask_id
        perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        temp_index = torch.triu(temp_index, diagonal=1)
        mask_index[temp_index] = False
        logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().float()
        return loss


    @torch.no_grad()
    def _eval_target_nll_duel(self, prefix, target):
        '''DUEL: the exact conditional NLL under an adaptively chosen unmasking order.

        len(target) sequential forwards at batch 1 (2 rows under CFG, still one model call), so
        this is the same total token-compute as chain_rule but launch-bound where chain_rule is
        batched. See duel_target_nll for what the number means.
        '''
        seq = torch.concatenate([prefix, target])[None, :].to(self.device)
        to_score = torch.zeros_like(seq, dtype=torch.bool)
        to_score[:, -len(target):] = True
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        # The closure is the seam that keeps CFG / padding / autocast in the class and the
        # numerics in a free function (mirrors evaluate_e2d2.py:441-442).
        return duel_target_nll(
            lambda z: self.get_logits(z, prompt_index),
            seq, to_score, self.mask_id, duel_alg=self.duel_alg,
        )

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        '''
        Employ Monte Carlo estimation to establish a lower bound of the log-likelihood
        '''
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.cpu())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.greddy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct


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

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 2048

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                elif self.nll_type == 'chain_rule':
                    ll = -self._eval_target_nll_ar(prefix, target)
                elif self.nll_type == 'duel':
                    ll = -self._eval_target_nll_duel(prefix, target)
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
    cli_evaluate()