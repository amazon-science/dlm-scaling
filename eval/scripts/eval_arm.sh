#!/usr/bin/env bash
# Zero-shot benchmark evaluation of an ARM checkpoint (Section 4.3 / Appendix B.4).
#
# The example below is the compute-optimal ARM cell at C = 1e20 (944M non-embedding
# parameters). Point CKPT at your own run and set MODEL to the matching size.
#
# Checkpoints must be .safetensors: evaluate_ar.py does
# load_state_dict(load_file(ckpt_path)) -- strict, prefix-free keys -- while training
# writes a full Fabric state (.pth, weights + optimizer moments). Convert first:
#
#     python scripts/pth_to_safetensors.py \
#       --in  workdir/arm-944M-100.0flop/iter-118656-ckpt.pth \
#       --out workdir/arm-944M-100.0flop/iter-118656-ckpt.safetensors \
#       --model 944 --arch ar
#
# ⚠️ THE ITER NUMBER IS max_step x grad_accum, NOT max_step: train_ar.py names the file
#    from iter_num, which counts MICRO-steps. save_step_interval is 9999999999999, so each
#    run directory holds exactly one .pth and there is nothing to choose.
#
# ONE SCORER, NOT TWO. The diffusion variants are scored twice (exact DUEL and the MC
# ELBO); an ARM has no such choice, since log p(target | prefix) is a single teacher-forced
# forward and is exact. There is no nll_type, mc_num, cfg or padding to pass -- model_name
# and ckpt_path are the only model_args evaluate_ar.py accepts. Its accuracies are
# therefore comparable with both diffusion columns, its NLL with neither.
#
# --batch_size is a deliberate no-op: ArEvalHarness ignores it and loglikelihood() loops one
# request at a time. lambada needs no `greddy` flag either -- suffix_greedy_prediction()
# runs unconditionally and lm_eval scores lambada's acc from it.
#
# ⚠️ TASK NAMES / ENV. It is `winogrande`, not `winogrande_xl` (the latter is the HF dataset
#    CONFIG name; lm_eval aborts with "Tasks not found"). `social_iqa` has a custom loading
#    script, so it needs HF_DATASETS_TRUST_REMOTE_CODE=true in the environment -- never
#    lm_eval's --trust_remote_code flag, which also appends trust_remote_code=True to
#    --model_args, an argument the harness rejects.

MODEL=944
CKPT=workdir/arm-944M-100.0flop/iter-118656-ckpt.safetensors

export HF_DATASETS_TRUST_REMOTE_CODE=true
export E2D2_BOOTSTRAP_WORKERS=8
export E2D2_ACC_NORM_TASKS=race

# --- why these two commands go through a wrapper ------------------------------------------
#
# evaluate_ar.py's __main__ is a bare cli_evaluate(), while the diffusion entry points call
# two lm_eval patches first. Both patches are model-agnostic (they patch lm_eval, not the
# model) and both are needed here as well:
#
#   patch_extra_acc_norm()   lm_eval's race.yaml declares `metric_list: [acc]`, which
#                            suppresses the acc_norm the harness already computed. Without
#                            this, race is the one task of the ten with no acc_norm row --
#                            and acc_norm is the metric race is reported at.
#   patch_bootstrap_stderr() lm_eval 0.4.4 does mp.Pool(mp.cpu_count()), and cpu_count() is
#                            the HOST core count, not the container's cpu limit. Forking
#                            that many copies of a process holding CUDA and a loaded model
#                            exhausts the fd table and multiprocessing then spins forever:
#                            a deadlock at 0% GPU with the results table unprinted, after
#                            every forward has already finished. Only metrics with no
#                            closed-form stderr take that path -- perplexity, i.e.
#                            lambada_standard -- so the risk is confined to the second
#                            command, but it is the same risk. E2D2_BOOTSTRAP_WORKERS caps
#                            the pool (default 8).
#
# lm_eval 0.4.4 exposes no --bootstrap_iters CLI flag, so neither fix is reachable from the
# command line. Importing evaluate_ar registers @register_model("ar"). Run from the repo root.

run_ar() {
  python - "$@" <<'PY'
import sys
sys.path.insert(0, 'eval')   # run from the repo root; the evaluators live in eval/

import evaluate_ar  # registers @register_model("ar")
from evaluate_e2d2 import patch_bootstrap_stderr, patch_extra_acc_norm
from lm_eval.__main__ import cli_evaluate

patch_bootstrap_stderr()
patch_extra_acc_norm()
cli_evaluate()
PY
}

run_ar \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model ar \
  --model_args model_name=$MODEL,ckpt_path=$CKPT

run_ar \
  --tasks lambada_standard \
  --model ar \
  --model_args model_name=$MODEL,ckpt_path=$CKPT
