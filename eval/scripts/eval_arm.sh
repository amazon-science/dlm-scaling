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
# ⚠️ race REPORTS acc_norm ONLY WITH THE PATCH -- lm_eval's race.yaml declares
#    `metric_list: [acc]`. evaluate_ar.py's __main__ calls patch_extra_acc_norm()
#    (E2D2_ACC_NORM_TASKS, default "race") itself, so there is nothing to pass here. It also
#    calls patch_bootstrap_stderr(), which is NOT optional for lambada_standard: without it
#    lm_eval 0.4.4 forks mp.cpu_count() workers (the HOST core count, not the container's cpu
#    limit) and deadlocks at 0% GPU after every forward has already completed.
#    E2D2_BOOTSTRAP_WORKERS caps the pool (default 8).
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

# Two commands, not three: there is no second scorer to run. Run from the repo root.

python eval/evaluate_ar.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model ar \
  --model_args model_name=$MODEL,ckpt_path=$CKPT

python eval/evaluate_ar.py \
  --tasks lambada_standard \
  --model ar \
  --model_args model_name=$MODEL,ckpt_path=$CKPT
