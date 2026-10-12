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
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.flops import flops_per_sample_mdm, max_step_from_flops, max_step_from_tokens, format_report
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
    parse.add_argument('--micro_batch_size', type=int, default=None, help='per-device micro batch size; overrides the size-based default if set')
    parse.add_argument('--low_var_loss', action='store_true', help='use the low-variance training objective (Sahoo et al. 2026, Eq. 8): unweighted CE over masked positions, dropping the 1/p_mask weighting. Evaluation still uses the weighted ELBO.')
    parse.add_argument('--learning_rate', type=float, default=4e-4,
                       help='peak learning rate; min_lr is learning_rate / 10')
    args = parse.parse_args()
    if (args.flops is None) == (args.tokens is None):
        parse.error('specify exactly one of --flops (*e18 FLOPs) or --tokens (*e9 tokens)')
    return args

args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'  # config
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
# Training = 3x forward (activation checkpointing is OFF, see setup()). Matmul 2 FLOPs per
# param per token, attention 4d per algorithmically-necessary (q,k) pair (bidirectional
# denoiser -> full n^2, all pairs are algorithmically necessary), lm_head 2*d*V*n
# (model_para_config excludes the head, so it is added explicitly).
_config = Config.from_name(model_name)
_f_parts, _f_fwd, _f_old_6nd = flops_per_sample_mdm(_config, model_para_config[f'{args.model}'])
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
# Warmup = 1% of the steps, with a floor of 100 steps (Section 3.2; inherited from SMDM).
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
# --- Evaluation protocol (frozen across every variant) ---
# Kept identical to pretrain/train_ar.py so MDM and ARM L_val are scored on the same tokens
# in the same order, and so MDM runs at different sizes are comparable with each other.
#
# The previous formula was
#     eval_iters = int(100 * 1024 * smdm_micro_batch_size / (global_batch_size * micro_batch_size))
# On this path micro_batch_size defaults to smdm_micro_batch_size, so the two cancel in the
# ratio -- but the sequence count still scales with it: sizes <=1028M scored 1x, 1233M/1678M
# 1/2x and 2121M 1/4x. Those are exactly the points near the 1e20 vertex, i.e. the eval set
# shrank where the cross-size L_val comparison matters most.
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

# DUEL exact-likelihood evaluation run once at the end of training (duel_validate_sweep below)
MASK_ID = 32000  # == padded_vocab_size; the masked-input token id
duel_max_eval_iters = eval_iters    # number of validation batches per rank
duel_num_steps = 2048       # NFE per sequence; == block_size -> fully sequential
duel_alg = 'greedy'         # 'greedy' (top-1 confidence) or 'left_to_right'


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


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
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


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
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
    _budget_tag = f'{args.flops}flop' if args.flops is not None else f'{args.tokens}tok'
    hp_name = f'mdm-{args.model}M-{_budget_tag}'
    if low_var_loss:
        hp_name += '-lowvar'
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
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    fabric.print(format_report("MDM", _f_parts, _f_fwd, _f_old_6nd, max_step, _max_step_6nd, _budget_desc))
    fabric.launch(main, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

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

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
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
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # --- DUEL exact-likelihood evaluation at the end of training ---
    if val_dataloader is not None:
        t0 = time.perf_counter()
        # No csv_path: the sweep is reported through the run log and wandb only.
        duel_results = duel_validate_sweep(
            fabric, model, val_dataloader,
            max_eval_iters=duel_max_eval_iters,
            num_steps=duel_num_steps,
            alg=duel_alg,
            block_size=config.block_size,
        )
        fabric.print(f"DUEL sweep done in {time.perf_counter() - t0:.1f}s")
        if duel_results:
            final_nll, final_ppl = duel_results[-1][3], duel_results[-1][4]
            fabric.log_dict({"metric/duel_nll": final_nll, "metric/duel_ppl": final_ppl}, state["step_count"])


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    # if val_dataloader is not None:
    #     validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = CrossEntropyLoss(reduction='none')
    for  train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
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
        
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(noisy_input)
            ce = loss_func(logits[mask_indices], input_ids[mask_indices])
            # low-variance objective (Sahoo et al. 2026, Eq. 8): unweighted CE over masked positions.
            # The default weighted ELBO (Eq. 7) divides by p_mask, which blows up at small mask ratios.
            loss = ce if low_var_loss else ce / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

            
            
            
        if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step):

            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
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
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        mc_loss = torch.zeros(128, device=fabric.device)  # mc_num=128
        for i in range(128):
            input_ids = val_data[:, 0 : model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            logits = model(noisy_input)
            loss = torch.nn.functional.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


@torch.no_grad()
def duel_validate_sweep(fabric, model, val_dataloader, max_eval_iters, num_steps, alg, block_size, csv_path=None):
    """DUEL exact-likelihood NLL/PPL, recorded cumulatively after every batch (k=1..max_eval_iters).

    Runs the DUEL exact-likelihood estimator (Turok et al., Algorithm 2: greedy top-1
    confidence, fully-sequential unmasking) over a fixed number of validation batches and
    records the cumulative NLL/PPL after every batch. Because each batch's trajectory is
    independent and the metric is a running sum, the cumulative value at k equals re-running
    the estimator over the first k batches. Returns a list of (k, n_seqs, n_tokens, nll, ppl).
    """
    fabric.print(f"DUEL validating (alg={alg}, num_steps={num_steps}, max_eval_iters={max_eval_iters}) ...")
    model.eval()

    L_ = block_size
    total_nll = torch.zeros((), device=fabric.device)     # summed over all tokens & sequences
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

        x = val_data[:, 0:L_].contiguous().to(fabric.device)   # (B, L) true tokens, in [0, 31999]
        B = x.shape[0]

        z = torch.full((B, L_), MASK_ID, dtype=torch.long, device=fabric.device)
        masked = torch.ones((B, L_), dtype=torch.bool, device=fabric.device)
        ll = torch.zeros(B, device=fabric.device)              # accumulated log p(x) per sequence
        pos = torch.arange(L_, device=fabric.device).unsqueeze(0).expand(B, L_)

        remaining = L_
        for step in range(num_steps):
            steps_left = num_steps - step
            k_step = min((remaining + steps_left - 1) // steps_left, remaining)  # ceil; ==1 when num_steps==L
            if k_step == 0:
                break

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(z)
            logits = logits.float()                            # (B, L, V) -> fp32 for an exact likelihood sum
            lse = torch.logsumexp(logits, dim=-1)              # (B, L)
            true_lp = torch.gather(logits, -1, x.unsqueeze(-1)).squeeze(-1) - lse  # (B, L) = log p(x_l | z)

            if alg == 'greedy':                                # top-1 confidence = log P^(1)_l
                score = logits.max(dim=-1).values - lse        # (B, L)
            elif alg == 'left_to_right':                       # smallest masked index first (sanity-check rule)
                score = (-pos).float()
            else:
                raise NotImplementedError(alg)

            score = score.masked_fill(~masked, float('-inf'))  # never re-pick revealed positions
            sel = torch.topk(score, k_step, dim=-1).indices    # (B, k_step); all rows in lockstep

            ll = ll + torch.gather(true_lp, -1, sel).sum(dim=-1)
            true_at_sel = torch.gather(x, -1, sel)
            z.scatter_(-1, sel, true_at_sel)                   # reveal the TRUE token
            masked.scatter_(-1, sel, torch.zeros_like(sel, dtype=torch.bool))
            remaining -= k_step

        total_nll = total_nll + (-ll).sum()
        total_tokens = total_tokens + B * L_

        # --- record cumulative metrics after this batch (k+1 batches consumed) ---
        # .clone() so the in-place all_reduce does not corrupt the running accumulators.
        red_nll = fabric.all_reduce(total_nll.clone(), reduce_op="sum")
        red_tokens = fabric.all_reduce(total_tokens.clone(), reduce_op="sum")
        nll = (red_nll / red_tokens)
        ppl = torch.exp(nll)

        batches_done = k + 1
        n_tokens = int(red_tokens.item())
        n_seqs = n_tokens // L_
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
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
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
    # Increase by one because we need the next word as well
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


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
