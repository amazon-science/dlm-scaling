import csv
import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.e2d2_model import E2D2Model, CrossDecoderBlock
from lit_gpt.model import Block, Config
from lit_gpt.flops import flops_per_sample_e2d2, max_step_from_flops, max_step_from_tokens, format_report
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    # Exactly one of --flops / --tokens sets the training length (see the budget block below).
    parse.add_argument('--flops', type=float, default=None, help='FLOP budget, *e18')
    parse.add_argument('--tokens', type=float, default=None,
                       help='token budget, *e9 (mutually exclusive with --flops)')
    parse.add_argument('--batch_size', type=int, default=256, help='global_batch_size')
    parse.add_argument('--n_dec_layers', type=int, default=None)
    parse.add_argument('--mtp_window', type=int, default=2)
    parse.add_argument('--micro_batch_size', type=int, default=None,
                       help='per-device micro batch size; overrides the size-based default if set')
    parse.add_argument('--low_var_loss', action='store_true',
                       help='use the low-variance training objective (Sahoo et al. 2026, Eq. 8): '
                            'unweighted CE over masked positions, dropping the 1/p_mask weighting. '
                            'Evaluation still uses the weighted ELBO.')
    parse.add_argument('--learning_rate', type=float, default=4e-4,
                       help='peak learning rate; min_lr is learning_rate / 10')
    args = parse.parse_args()
    if (args.flops is None) == (args.tokens is None):
        parse.error('specify exactly one of --flops (*e18 FLOPs) or --tokens (*e9 tokens)')
    return args

args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'
out_dir = Path('workdir')

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328,
    '2718': 2718.058752
}

# Hyperparameters
num_of_devices = 8
global_batch_size = int(args.batch_size / args.nodes_num)
learning_rate = args.learning_rate
low_var_loss = args.low_var_loss  # train with unweighted CE (Sahoo et al. 2026, Eq. 8); eval stays weighted

# if args.model <= 20:
#     smdm_micro_batch_size = 32
# elif args.model <= 50:
#     smdm_micro_batch_size = 16
# elif args.model <= 1028:
#     smdm_micro_batch_size = 8
# elif args.model <= 2000:
#     smdm_micro_batch_size = 4
# else:
#     smdm_micro_batch_size = 2
smdm_micro_batch_size = 32
if args.micro_batch_size is not None:
    micro_batch_size = args.micro_batch_size
else:
    micro_batch_size = smdm_micro_batch_size

# --- FLOP accounting (real FLOPs; see lit_gpt/flops.py and Appendix B of the paper) ---
# Encoder-decoder, both towers length n (decoder blocks are NON-overlapping). Encoder
# attention is block-causal (e2d2_model.py:_build_encoder_mask), decoder self-attn is
# block-diagonal, cross-attn is a strict block prefix, and lm_head runs over n positions.
# Training = 3x forward (activation checkpointing is OFF, see setup()).
_config = Config.from_name(model_name)
_n_dec = args.n_dec_layers if args.n_dec_layers is not None else _config.n_layer // 2
_f_parts, _f_fwd, _f_old_6nd = flops_per_sample_e2d2(
    _config, model_para_config[f'{args.model}'], b=args.mtp_window, n_dec_layers=_n_dec)
_flops_per_sample = 3 * _f_fwd
# Training length comes from exactly one budget (enforced in parse_args). With --tokens the
# step count is FLOP-independent, so the 6ND step cross-reference is dropped (None).
if args.flops is not None:
    max_step      = max_step_from_flops(args.flops, _flops_per_sample, global_batch_size, args.nodes_num)
    _max_step_6nd = max_step_from_flops(args.flops, _f_old_6nd,        global_batch_size, args.nodes_num)
    _budget_desc  = f'--flops {args.flops}e18 -> max_step={max_step}'
else:
    max_step      = max_step_from_tokens(args.tokens, _config.block_size, global_batch_size, args.nodes_num)
    _max_step_6nd = None
    _budget_desc  = (f'--tokens {args.tokens}e9 -> max_step={max_step}'
                     f' (real FLOPs {max_step * _flops_per_sample * global_batch_size * args.nodes_num / 1e18:.2f}e18)')
# Warmup = 1% of the steps with a floor of 100 steps (Section 3.2; inherited from SMDM).
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
# --- Evaluation protocol (frozen across every variant) ---
# Kept identical to pretrain/train_mdm.py (and train_ar.py) so E2D2 and MDM L_val are scored
# on the same tokens in the same order, and so E2D2 runs at different sizes are comparable
# with each other.
#
# The previous formula was
#     eval_iters = int(100 * 1024 * smdm_micro_batch_size / (global_batch_size * micro_batch_size))
# which carried smdm_micro_batch_size -- a per-size *throughput* knob -- into the eval
# protocol, and also made the eval set depend on --micro_batch_size (which does NOT appear in
# hp_name, so the change was invisible from the run name). Consequence at 2121M with
# --micro_batch_size 8: 800 seq/rank = 13.1M tokens scored, vs MDM's 6144 seq/rank = 100.7M,
# i.e. not comparable and far below MDM's scored token count.
#
# 6144 seq/rank = exactly 6 validation files (2098176 tokens / 2049-token block = 1024 blocks
# each), divides evenly by every micro_batch_size in use (2/4/8/16/32) so the set is identical
# token-for-token across sizes, and fits inside the first n_chunks=8 mmap load -- no file-index
# wrap. At world_size=8: 49152 seq = 100.7M tokens scored.
VAL_SEQ_PER_RANK = 6144   # 6 SlimPajama validation files; frozen, see the note above
assert VAL_SEQ_PER_RANK % micro_batch_size == 0, (
    f'VAL_SEQ_PER_RANK={VAL_SEQ_PER_RANK} is not divisible by micro_batch_size='
    f'{micro_batch_size}: the eval set would differ across runs and L_val would not be '
    'comparable. Pick a micro_batch_size that divides it.'
)
eval_iters = VAL_SEQ_PER_RANK // micro_batch_size
# Validation is sharded by rank over world size, so the eval set is only well defined at the
# frozen 1 node x 8 GPU layout.
assert args.nodes_num == 1, (
    f'--nodes_num={args.nodes_num}: the frozen eval set is defined at world_size=8 '
    '(1 node x 8 GPUs). Validation is sharded by rank, so a different world size scores '
    'different tokens and L_val is no longer comparable across the project.'
)
save_step_interval = 9999999999999   # checkpoint only at the end of training (step_count == max_step)
eval_step_interval = 9999999999999

# DUEL exact-likelihood evaluation run once at the end of training.
# E2D2 is block-autoregressive: log p(x) = sum_k log p(x^k | x^{<k}). Within-block
# unmasking runs for mtp_window steps; MASK_ID is read from config.padded_vocab_size.
duel_max_eval_iters = eval_iters
duel_alg = 'greedy'                # 'greedy' (top-1 confidence) or 'left_to_right'

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = learning_rate / 10

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

train_data_config = [
    ("train_slim", 1.0),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
# CSV metrics log, written next to the checkpoints (OUT_ROOT, default "workdir").
_csv_root = os.path.join(os.environ.get('OUT_ROOT', 'workdir'), "out")
logger = step_csv_logger(_csv_root, model_name, flush_logs_every_n_steps=log_iter_interval)


def ensure_one_mask_per_block(xt, mask_indices, mask_id, block_size):
    B, l = xt.shape
    n_blocks = l // block_size
    has_mask = mask_indices.view(B, n_blocks, block_size).any(dim=-1)  # (B, n_blocks)
    empty = ~has_mask
    if not empty.any():
        return xt, mask_indices
    rand = torch.rand(B, n_blocks, block_size, device=xt.device)
    rand = rand * empty.unsqueeze(-1).float()
    force = (rand == rand.max(dim=-1, keepdim=True).values) & empty.unsqueeze(-1)
    force = force.view(B, l)
    xt = torch.where(force, mask_id, xt)
    mask_indices = mask_indices | force
    return xt, mask_indices


def forward_process(batch, total_dim, block_size, is_training=True, eps=1e-3):
    b, l = batch.shape
    n_blocks = l // block_size
    t = torch.rand((b, n_blocks), device=batch.device).repeat_interleave(block_size, dim=1)
    p_mask = (1 - eps) * t + eps
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    if is_training:
        noisy_batch, mask_indices = ensure_one_mask_per_block(noisy_batch, mask_indices, total_dim, block_size)
    return noisy_batch, mask_indices, p_mask


def setup(
    devices: int = 8,
    # DATA_DIR overrides the location of the packed SlimPajama chunks written by
    # scripts/prepare_slimpajama.py.
    train_data_dir: Path = Path(os.environ.get('DATA_DIR', 'data/slim_star_combined')),
    val_data_dir: Path = Path(os.environ.get('DATA_DIR', 'data/slim_star_combined')),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    _n_dec_layers = args.n_dec_layers if args.n_dec_layers is not None else Config.from_name(model_name).n_layer // 2
    _budget_tag = f'{args.flops}flop' if args.flops is not None else f'{args.tokens}tok'
    hp_name = f'e2d2-{args.model}M-{_n_dec_layers}dec-w{args.mtp_window}-{_budget_tag}'
    if low_var_loss:
        hp_name += '-lowvar'
    # Batch geometry and lr only appear when they deviate from the default, so the default
    # sweep keeps short names -- but two runs that differ in either no longer share out_dir
    # (and hence no longer pick up each other's checkpoint through resume=True).
    if args.batch_size != 256:
        hp_name += f'-gbs{args.batch_size}'
    if args.learning_rate != 4e-4:
        hp_name += f'-lr{args.learning_rate}'
    # One directory per run under OUT_ROOT (default "workdir"): the single
    # end-of-training checkpoint, the CSV metrics log and the wandb run files.
    out_dir = Path(os.environ.get('OUT_ROOT', 'workdir')) / hp_name
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, log_model=False,
                               project=os.environ.get('WANDB_PROJECT', 'dlm-scaling'))

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block, CrossDecoderBlock},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    _tag = f"E2D2(n_dec={_n_dec},b={args.mtp_window})"
    fabric.print(format_report(_tag,
                               _f_parts, _f_fwd, _f_old_6nd, max_step, _max_step_6nd, _budget_desc))
    fabric.launch(main, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    config.n_dec_layers = args.n_dec_layers if args.n_dec_layers is not None else config.n_layer // 2
    config.mtp_window = args.mtp_window

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = E2D2Model(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        import re
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # --- DUEL exact-likelihood evaluation at the end of training (block-autoregressive) ---
    if val_dataloader is not None:
        t0 = time.perf_counter()
        # No csv_path: the sweep is reported through the run log and wandb only.
        duel_results = duel_validate_sweep(
            fabric, model, val_dataloader,
            max_eval_iters=duel_max_eval_iters,
            alg=duel_alg,
            block_size=config.block_size,
            b=config.mtp_window,
        )
        fabric.print(f"DUEL sweep done in {time.perf_counter() - t0:.1f}s")
        if duel_results:
            final_nll, final_ppl = duel_results[-1][3], duel_results[-1][4]
            fabric.log_dict({"metric/duel_nll": final_nll, "metric/duel_ppl": final_ppl}, state["step_count"])


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    with torch.device("meta"):
        meta_model = E2D2Model(model.config)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        del meta_model

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0

    loss_func = CrossEntropyLoss(reduction='none')

    for train_data in train_dataloader:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()  # (B, n)
        padded_vocab_size = model.config.padded_vocab_size

        noisy_ids, mask_indices, p_mask = forward_process(input_ids, total_dim=padded_vocab_size, block_size=model.mtp_window, is_training=True)

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, noisy_ids)  # (B, n, vocab)
            ce = loss_func(logits[mask_indices], input_ids[mask_indices])
            ce = ce if low_var_loss else ce / p_mask[mask_indices]
            B_, n_ = input_ids.shape
            loss = ce.sum() / (B_ * n_)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours."
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days."
        )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item()
        )

        if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step):
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(
                f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            # Checkpoint save must never abort the run: the end-of-training DUEL eval
            # (and its wandb logging) has to run even if the disk write fails. fabric.save
            # is a collective (all-gather to rank0), so every rank wraps it identically and
            # continues unconditionally afterwards -- keeping control flow rank-consistent
            # so no rank deadlocks in a later collective.
            try:
                fabric.save(checkpoint_path, state)
            except Exception as e:
                fabric.print(f"WARNING: fabric.save failed, continuing to evaluation: {e}")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader):
    """Weighted-ELBO L_val, estimated with 128 Monte-Carlo samples per batch."""
    fabric.print("Validating ...")
    model.eval()

    padded_vocab_size = model.config.padded_vocab_size
    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        mc_loss = torch.zeros(128, device=fabric.device)
        for i in range(128):
            input_ids = val_data[:, 0 : model.config.block_size].contiguous()
            noisy_ids, mask_indices, p_mask = forward_process(input_ids, total_dim=padded_vocab_size, block_size=model.mtp_window, is_training=False)
            logits = model(input_ids, noisy_ids)  # (B, n, vocab)
            loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none')
            loss = (loss / p_mask[mask_indices]).sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss

        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")

    model.train()
    return losses.mean()


@torch.no_grad()
def duel_validate_sweep(fabric, model, val_dataloader, max_eval_iters, alg, block_size, b, csv_path=None):
    """DUEL exact-likelihood NLL/PPL for E2D2 (block-autoregressive encoder-decoder MDM).

    E2D2 factorizes log p(x) = sum_k log p(x^k | x^{<k}). Its decoder mask keeps enc_idx clean and
    lets decoder block k attend to encoder positions 0..k*b-1 (cross) and to its own block (self).
    Because the encoder is block-causal (so encoder position k*b-1 sees exactly the clean
    prefix x^{<k} and nothing beyond it), one
    pass over the full clean sequence yields the same
    prefix representations as growing it block-by-block, so all n_blocks = block_size // b blocks
    are conditionally independent given enc_idx and can be unmasked in parallel. DUEL therefore
    unmasks WITHIN each block: reveal one position per block per step (greedy top-1 confidence, or
    left_to_right), for b steps total. Summing the true-token conditional log-probs along this
    within-block order telescopes to exact log p(x).
    """
    fabric.print(f"DUEL validating (alg={alg}, mtp_window={b}, max_eval_iters={max_eval_iters}) ...")
    model.eval()

    MASK_ID = model.config.padded_vocab_size
    n = block_size
    n_full = (n // b) * b           # drop any trailing partial block
    n_blocks = n // b
    block_base = (torch.arange(n_blocks, device=fabric.device) * b).view(1, n_blocks, 1)

    total_nll = torch.zeros((), device=fabric.device)
    total_tokens = torch.zeros((), device=fabric.device)
    results = []

    fh = writer = None
    if fabric.global_rank == 0 and csv_path is not None:
        fh = open(csv_path, 'w', newline='')
        writer = csv.writer(fh)
        writer.writerow(['duel_eval_iters', 'n_seqs', 'n_tokens', 'nll', 'ppl'])

    for k, val_data in enumerate(val_dataloader):
        if k >= max_eval_iters:
            break
        t0 = time.perf_counter()

        x_full = val_data[:, 0:n].contiguous().to(fabric.device)   # (B, n) clean true tokens
        clean_idx = x_full                                         # (B, n) CLEAN, never mutated
        x = x_full[:, :n_full]                                     # (B, n_full) scored targets
        B = x.shape[0]

        z = x_full.clone()                                         # decoder input; start fully masked over n_full
        z[:, :n_full] = MASK_ID
        masked = torch.zeros((B, n), dtype=torch.bool, device=fabric.device)
        masked[:, :n_full] = True
        ll = torch.zeros(B, device=fabric.device)
        pos = torch.arange(n_full, device=fabric.device).unsqueeze(0).expand(B, n_full)

        for step in range(b):                                      # one position revealed per block per step
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(clean_idx, z)                       # (B, n, V)
            logits = logits[:, :n_full, :].float()
            lse = torch.logsumexp(logits, dim=-1)
            true_lp = torch.gather(logits, -1, x.unsqueeze(-1)).squeeze(-1) - lse

            if alg == 'greedy':
                score = logits.max(dim=-1).values - lse
            elif alg == 'left_to_right':
                score = (-pos).float()
            else:
                raise NotImplementedError(alg)

            score = score.masked_fill(~masked[:, :n_full], float('-inf'))
            sel_local = torch.topk(score.view(B, n_blocks, b), 1, dim=-1).indices
            sel = (sel_local + block_base).view(B, n_blocks)

            ll = ll + torch.gather(true_lp, -1, sel).sum(dim=-1)
            true_at_sel = torch.gather(x, -1, sel)
            z.scatter_(-1, sel, true_at_sel)
            masked.scatter_(-1, sel, torch.zeros_like(sel, dtype=torch.bool))

        total_nll = total_nll + (-ll).sum()
        total_tokens = total_tokens + B * n_full

        red_nll = fabric.all_reduce(total_nll.clone(), reduce_op="sum")
        red_tokens = fabric.all_reduce(total_tokens.clone(), reduce_op="sum")
        nll = red_nll / red_tokens
        ppl = torch.exp(nll)

        batches_done = k + 1
        n_tokens = int(red_tokens.item())
        n_seqs = n_tokens // n_full
        dt = time.perf_counter() - t0
        results.append((batches_done, n_seqs, n_tokens, nll.item(), ppl.item()))
        fabric.print(
            f"duel_eval_iters {batches_done}: DUEL nll {nll.item():.4f}, DUEL ppl {ppl.item():.4f}"
            f" (n_seqs {n_seqs}, n_tokens {n_tokens}, batch time {dt:.1f}s)"
        )
        if writer is not None:
            writer.writerow([batches_done, n_seqs, n_tokens, f"{nll.item():.6f}", f"{ppl.item():.6f}"])
            fh.flush()

    if fh is not None:
        fh.close()
    model.train()
    return results


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size (and, when shuffle=True, the shuffle window).
            # 8 on BOTH splits, matching train_mdm.py: the frozen validation set (6 files, see
            # VAL_SEQ_PER_RANK) is required to fit inside the first mmap load so _load_n_chunks
            # never wraps _file_idx back to 0 mid-eval.
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()
