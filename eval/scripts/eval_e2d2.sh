#!/usr/bin/env bash
# Zero-shot benchmark evaluation of an E2D2 checkpoint (Section 4.3 / Appendix B.4).
#
# Three commands, the same ones (and the same task order) as eval_mdm.sh / eval_bd3lm.sh: the
# ten multiple-choice tasks under the exact DUEL likelihood, the same ten under the 128-sample
# MC ELBO, then lambada_standard with the greedy block decode.
#
# The example below is the compute-optimal E2D2 cell of the b = 32 arm at C = 1e20
# (472M parameters, n_dec = 14). Point CKPT at your own run and set MODEL / B / N_DEC to match
# it. Convert the training checkpoint first:
#
#     python scripts/pth_to_safetensors.py \
#       --in  workdir/e2d2-472M-14dec-w32-100.0flop-lowvar/iter-048715-ckpt.pth \
#       --out workdir/e2d2-472M-14dec-w32-100.0flop-lowvar/iter-048715-ckpt.safetensors \
#       --model 472 --arch e2d2 --n_dec_layers 14
#
# ⚠️ THE ITER NUMBER IS PER-ARM. The same (N, n_dec, C) cell has a different max_step in each
#    mtp_window arm, because the FLOP accounting does. Copying an iter number across arms gives
#    a file-not-found at load time.
# ⚠️ mtp_window must be passed: it only selects attention masks, so a wrong value scores a
#    different factorization without erroring. The w{b} tag in the run name is the ground truth.
# ⚠️ race REPORTS acc_norm ONLY WITH THE PATCH. lm_eval's race.yaml declares
#    `metric_list: [acc]`, so a plain run prints a race `acc` row and no `acc_norm` -- the one
#    task of the ten that could not otherwise be reported at its primary metric.
#    evaluate_e2d2.py's patch_extra_acc_norm() adds the metric (E2D2_ACC_NORM_TASKS, default
#    "race") to the task lm_eval built, leaving contexts and loglikelihoods untouched, so `acc`
#    is unchanged. Nothing to pass here. Its patch_bootstrap_stderr() is likewise not optional
#    for lambada_standard: without it lm_eval 0.4.4 forks mp.cpu_count() workers (the HOST core
#    count, not the container's cpu limit) and deadlocks at 0% GPU after every forward has
#    completed. E2D2_BOOTSTRAP_WORKERS caps the pool (default 8).
# ⚠️ TASK NAMES / ENV. It is `winogrande`, not `winogrande_xl`. `social_iqa` needs
#    HF_DATASETS_TRUST_REMOTE_CODE=true in the environment -- never --trust_remote_code, which
#    appends trust_remote_code=True to --model_args, an argument __init__ rejects. And never
#    double-quote --model_args, or the harness receives the literal "'duel'" and asserts.

MODEL=472
B=32
N_DEC=14
CKPT=workdir/e2d2-472M-14dec-w32-100.0flop-lowvar/iter-048715-ckpt.safetensors

export HF_DATASETS_TRUST_REMOTE_CODE=true
export E2D2_BOOTSTRAP_WORKERS=8
export E2D2_ACC_NORM_TASKS=race

# --- 1 of 3: the ten multiple-choice tasks under the EXACT likelihood ----------------------
#
# This is the number to report. No --batch_size: the duel path is one request at a time and
# never reads it.

python eval/evaluate_e2d2.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model e2d2 \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,mtp_window=$B,n_dec_layers=$N_DEC,nll_type='duel',duel_alg='greedy',greddy=False,cfg=0.

# --- 2 of 3: the same ten scored by the MC weighted ELBO instead ---------------------------
#
# WHY BOTH. nll_type=duel is E2D2's own exact factorization and is the number to report; this
# run exists for two other reasons:
#   1. Comparability with the MDM arm, which has NO cheap exact estimator, so eval_mdm.sh
#      scores its ten with nll_type=mc. Accuracies from the two mc runs are on the same
#      footing; the NLLs are not (E2D2's is a per-block ELBO, MDM's per-sequence).
#   2. A ranking sanity check: duel and mc are different scoring rules over the same
#      candidates, so a large accuracy gap between these two commands points at a scorer bug
#      rather than at the model.
#
# mc_num=128 matches validate()'s hardcoded 128 in pretrain/train_e2d2.py, so this is on the
# same scale as the metric/val_loss logged during training. --batch_size 128 makes it exactly
# ONE forward per request (evaluate_e2d2.py asserts mc_num % batch_size == 0, and only the mc
# path reads batch_size at all).

python eval/evaluate_e2d2.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model e2d2 --batch_size 128 \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,mtp_window=$B,n_dec_layers=$N_DEC,nll_type='mc',mc_num=128,greddy=False,cfg=0.

# --- 3 of 3: lambada_standard needs the greedy block decode (greddy=True) ------------------

python eval/evaluate_e2d2.py \
  --tasks lambada_standard \
  --model e2d2 \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,mtp_window=$B,n_dec_layers=$N_DEC,nll_type='duel',duel_alg='greedy',greddy=True,cfg=0.

# padding stays at its default False, NOT the padding=True of the MDM protocol. For MDM that
# flag genuinely changes the score -- it is fully bidirectional, so the 2048-token MASK tail is
# visible to every scored position. For E2D2 it cannot: whole MASK blocks are unreachable from
# an earlier block. All padding=True would change here is how many random draws
# block_forward_process consumes, so it buys nothing and costs b times the tokens per forward.
#
# ⚠️ MEMORY on `race`. batch_size=128 materialises a (128, n, 32000) logits tensor twice (bf16
#    from autocast, then .float()). At n=128 that is ~2 GB and fine; race's contexts run to
#    nearly 2048 tokens, where it is ~50 GB. If it OOMs, drop to --batch_size 32
#    (128 % 32 == 0, four forwards of a quarter the size) rather than lowering mc_num, which
#    would change the estimator's variance and break comparability with validate().
