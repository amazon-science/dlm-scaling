"""Convert a Fabric training checkpoint (.pth) into a bare-state_dict .safetensors.

The pretraining scripts save with `fabric.save(path, state)`, so an `iter-*-ckpt.pth`
holds the FULL training state -- model, optimizer moments, iter_num, step_count -- and is
~3x the size of the weights alone. Every evaluation entry point instead does

    self.model.load_state_dict(load_file(ckpt_path))     # evaluate_mdm.py, evaluate_ar.py

i.e. safetensors, strict, prefix-free keys. This script bridges the two.

    python scripts/pth_to_safetensors.py \
        --in  workdir/mdm-944M-100.0flop-lowvar/iter-027637-ckpt.pth \
        --out workdir/mdm-944M-100.0flop-lowvar/iter-027637-ckpt.safetensors \
        --model 944 --arch mdm

Architectures: mdm (TransEncoder), ar (GPT), e2d2 (E2D2Model), bd3lm (BD3LM).

`--model`/`--arch` are optional but recommended: they build the target module on the meta
device and check the produced key set matches it EXACTLY, which is the only thing that
tells you `load_state_dict` will not raise later. Without them the conversion still runs
and is still verified for self-consistency (round-trip reload), just not against the
architecture.

Reads the .pth with mmap=True so the optimizer moments are never faulted into RAM: peak
RSS is one copy of the weights (944M fp32 ~ 3.8 GiB), not the whole 12.5 GiB file.
"""
import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def strip_wrapper_prefix(key):
    """Drop an FSDP/DDP wrapper prefix.

    Fabric writes clean prefix-free keys today; this is defensive so a checkpoint saved
    through a future wrapper still converts.
    """
    for pre in ('_forward_module.', 'module.'):
        if key.startswith(pre):
            return key[len(pre):]
    return key


def load_model_state_dict(pth_path):
    """The model sub-dict of a Fabric checkpoint, prefix-free, without the optimizer."""
    try:
        ckpt = torch.load(pth_path, map_location='cpu', mmap=True)
    except (TypeError, RuntimeError):
        # mmap= needs torch>=2.1 and a zipfile-serialized checkpoint; fall back to a
        # full read, which costs the optimizer moments in RAM as well.
        print('note: mmap load unavailable, falling back to a full torch.load')
        ckpt = torch.load(pth_path, map_location='cpu')

    if not (isinstance(ckpt, dict) and 'model' in ckpt):
        raise SystemExit(
            f'{pth_path} is not a Fabric checkpoint: expected a dict with a "model" key, '
            f'got {type(ckpt).__name__}'
            + (f' with keys {sorted(ckpt)[:8]}' if isinstance(ckpt, dict) else '')
        )

    meta = {k: v for k, v in ckpt.items() if k in ('iter_num', 'step_count')}
    # .clone() detaches each tensor from the mmap and from any shared storage, both of
    # which save_file rejects; .contiguous() first so a non-contiguous view stays cheap.
    sd = {strip_wrapper_prefix(k): v.contiguous().clone()
          for k, v in ckpt['model'].items()}
    return sd, meta


def expected_keys(arch, model_tag, n_dec_layers=None):
    """Key set of a freshly built module, on the meta device so nothing is allocated.

    e2d2 needs one extra that is NOT part of the Diff_LLaMA config name: n_dec_layers, which
    splits n_layer into encoder_blocks / decoder_blocks. It defaults to n_layer // 2, same as
    the trainer. mtp_window is deliberately absent: it only selects attention masks, no
    parameters.

    bd3lm needs neither extra -- it is a single tower with no conditional parameters
    (bd3lm_model.py:46-51) -- so its key set is fully determined by the config name. Note that
    this means the key check CANNOT catch a wrong mtp_window; only the w{b} tag in the run name
    can (see evaluate_bd3lm.py:parse_hp_name).
    """
    from lit_gpt.config import Config
    if arch == 'mdm':
        from lit_gpt.diffmodel import TransEncoder as Model
    elif arch == 'e2d2':
        from lit_gpt.e2d2_model import E2D2Model as Model
    elif arch == 'bd3lm':
        from lit_gpt.bd3lm_model import BD3LM as Model
    else:
        from lit_gpt.model import GPT as Model
    config = Config.from_name(f'Diff_LLaMA_{model_tag}M')
    if arch == 'e2d2':
        config.n_dec_layers = n_dec_layers if n_dec_layers is not None else config.n_layer // 2
    with torch.device('meta'):
        model = Model(config)
    return set(model.state_dict().keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='src', required=True, help='source iter-*-ckpt.pth')
    p.add_argument('--out', dest='dst', required=True, help='destination .safetensors')
    p.add_argument('--model', default=None,
                   help='model size tag, e.g. 944 -- enables the architecture key check')
    p.add_argument('--arch', default='mdm', choices=('mdm', 'ar', 'e2d2', 'bd3lm'),
                   help='mdm -> TransEncoder (diffmodel.py), ar -> GPT (model.py), '
                        'e2d2 -> E2D2Model (e2d2_model.py), bd3lm -> BD3LM (bd3lm_model.py)')
    p.add_argument('--n_dec_layers', type=int, default=None,
                   help='--arch e2d2 only: decoder depth, e.g. 14 for '
                        'e2d2-666M-14dec-w4-100.0flop-lowvar. Defaults to n_layer // 2')
    p.add_argument('--overwrite', action='store_true',
                   help='replace --out if it already exists (refuses by default)')
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        raise SystemExit(f'no such checkpoint: {src}')
    if dst.exists() and not args.overwrite:
        raise SystemExit(f'{dst} already exists; pass --overwrite to replace it')

    print(f'reading  {src}  ({src.stat().st_size / 2**30:.2f} GiB)')
    sd, meta = load_model_state_dict(src)
    n_params = sum(v.numel() for v in sd.values())
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    print(f'  {len(sd)} tensors, {n_params:,} elements, dtype(s) {dtypes}, '
          f'checkpoint {meta}')

    if args.model:
        want = expected_keys(args.arch, args.model, n_dec_layers=args.n_dec_layers)
        got = set(sd)
        missing, unexpected = sorted(want - got), sorted(got - want)
        if missing or unexpected:
            raise SystemExit(
                f'key mismatch against Diff_LLaMA_{args.model}M ({args.arch}): '
                f'{len(missing)} missing {missing[:5]}, '
                f'{len(unexpected)} unexpected {unexpected[:5]}'
            )
        print(f'  keys match Diff_LLaMA_{args.model}M ({args.arch}) exactly')

    # Write beside the destination and rename, so an interrupted run can never leave a
    # truncated .safetensors that looks loadable.
    tmp = dst.with_suffix(dst.suffix + '.partial')
    print(f'writing  {tmp}')
    save_file(sd, str(tmp))
    os.replace(tmp, dst)
    print(f'wrote    {dst}  ({dst.stat().st_size / 2**30:.2f} GiB)')

    # Round-trip: the file on disk must reproduce the dict we just built, exactly.
    del sd
    reread = load_file(str(dst))
    src_sd, _ = load_model_state_dict(src)
    assert set(reread) == set(src_sd), 'round-trip key set differs'
    for k, v in src_sd.items():
        assert reread[k].dtype == v.dtype, f'{k}: dtype {reread[k].dtype} != {v.dtype}'
        assert reread[k].shape == v.shape, f'{k}: shape {reread[k].shape} != {v.shape}'
        assert torch.equal(reread[k], v), f'{k}: values differ'
    print(f'verified {len(reread)} tensors reload bit-identical')


if __name__ == '__main__':
    main()
