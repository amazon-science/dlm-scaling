#!/usr/bin/env bash
# Zero-shot benchmark evaluation of a BD3LM checkpoint (Section 4.3 / Appendix B.4).
#
# Three commands, the same ones (and the same task order) as eval_mdm.sh / eval_e2d2.sh: the
# ten multiple-choice tasks under the exact DUEL likelihood, the same ten under the 128-sample
# MC ELBO, then lambada_standard with the greedy block decode.
#
# The example below is the compute-optimal BD3LM cell of the b = 4 arm at C = 1e19. Point CKPT
# at your own run and set MODEL / B to match it. Convert the training checkpoint first:
#
#     python scripts/pth_to_safetensors.py \
#       --in  workdir/bd3lm-142M-w4-10.0flop-lowvar/iter-008954-ckpt.pth \
#       --out workdir/bd3lm-142M-w4-10.0flop-lowvar/iter-008954-ckpt.safetensors \
#       --model 142 --arch bd3lm
#
# ⚠️ mtp_window IS THE ONE FLAG THE WEIGHTS CANNOT CHECK. BD3LM has no block-size-conditional
#    parameters, so --arch bd3lm's key check passes for ANY b and a strict load_state_dict
#    cannot fail on a wrong one -- it just scores a different factorization. The w{b} tag in
#    the run name is the ground truth; evaluate_bd3lm.py infers b from it and prints what it
#    resolved. Pass mtp_window= explicitly anyway so the intent is on record.
# ⚠️ THE ITER NUMBER IS PER-ARM: the FLOP accounting differs per b, so the same (N, C) cell has
#    a different max_step in the w4 / w8 / w16 / w32 arms. Never copy an iter number across arms.
# ⚠️ race REPORTS acc_norm ONLY WITH THE PATCH -- lm_eval's race.yaml declares
#    `metric_list: [acc]`. evaluate_bd3lm.py's __main__ calls patch_extra_acc_norm()
#    (E2D2_ACC_NORM_TASKS, default "race") itself, so there is nothing to pass here. It also
#    calls patch_bootstrap_stderr(), which is NOT optional for lambada_standard: without it
#    lm_eval 0.4.4 forks mp.cpu_count() workers (the HOST core count, not the container's cpu
#    limit) and deadlocks at 0% GPU after every forward has already completed.
#    E2D2_BOOTSTRAP_WORKERS caps the pool (default 8).
# ⚠️ TASK NAMES / ENV. It is `winogrande`, not `winogrande_xl`. `social_iqa` needs
#    HF_DATASETS_TRUST_REMOTE_CODE=true in the environment -- never --trust_remote_code, which
#    appends trust_remote_code=True to --model_args, an argument __init__ rejects. And never
#    double-quote --model_args, or the harness receives the literal "'duel'" and asserts.

MODEL=142
B=4
CKPT=workdir/bd3lm-142M-w4-10.0flop-lowvar/iter-008954-ckpt.safetensors

export HF_DATASETS_TRUST_REMOTE_CODE=true
export E2D2_BOOTSTRAP_WORKERS=8
export E2D2_ACC_NORM_TASKS=race

# --- 1 of 3: the ten multiple-choice tasks under the EXACT likelihood ----------------------
#
# This is the number to report. DUEL costs <= mtp_window forwards per request regardless of
# how long the target is, because BD3LM's block-lower quadrant is a strict block prefix
# (kq > kk), so all n/b blocks are conditionally independent given the clean sequence. Compare
# with MDM, where evaluate_mdm.py's duel/chain_rule needs len(target) passes -- BD3LM
# zero-shot is cheap.
#
# No --batch_size: the duel path is one request at a time and never reads it.

python eval/evaluate_bd3lm.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model bd3lm \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,mtp_window=$B,nll_type='duel',duel_alg='greedy',greddy=False,cfg=0.

# --- 2 of 3: the same ten scored by the MC weighted ELBO instead ---------------------------
#
# WHY BOTH. nll_type=duel is BD3LM's own exact factorization; this run exists for two other
# reasons:
#   1. Comparability with the MDM arm, which has no cheap exact estimator. The ACCURACIES from
#      the two mc runs are on the same footing; the NLLs are not (BD3LM's is a per-block ELBO,
#      MDM's is per-sequence) -- and an ELBO at b=4 is a different functional from one at b=32,
#      so only the DUEL column is readable ACROSS arms.
#   2. A ranking sanity check: duel and mc are different scoring rules over the same
#      candidates, so a large accuracy gap between commands 1 and 2 points at a scorer bug
#      rather than at the model.
#
# mc_num=128 matches validate()'s hardcoded 128 in pretrain/train_bd3lm.py, putting this on the
# same scale as the metric/val_loss logged during training. --batch_size 128 makes it exactly
# ONE forward per request (mc_num % batch_size == 0 is asserted, and only the mc path reads
# batch_size at all).

python eval/evaluate_bd3lm.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model bd3lm --batch_size 128 \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,mtp_window=$B,nll_type='mc',mc_num=128,greddy=False,cfg=0.

# --- 3 of 3: lambada_standard needs the greedy block decode (greddy=True) ------------------
#
# lambada goes through loglikelihood(), never generate_until(): lm_eval reads its `perplexity`
# off the log-likelihood and its `acc` off the second element of each tuple, which is
# suffix_greedy_prediction's exact-match bit. That decode is block-AUTOREGRESSIVE and, unlike
# the scorers, cannot be parallel across blocks -- block k's clean context has to contain what
# block k-1 generated -- so it costs ~len(target) forwards. lambada's targets are 1-3 tokens,
# so it is cheap here; that is why greddy defaults to False for the runs above.

python eval/evaluate_bd3lm.py \
  --tasks lambada_standard \
  --model bd3lm \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,mtp_window=$B,nll_type='duel',duel_alg='greedy',greddy=True,cfg=0.

# --- padding stays at its default False ----------------------------------------------------
#
# Score-identical to padding=True (whole extra MASK blocks cannot reach an earlier block)
# and much cheaper: BD3LM is a
# SINGLE tower over 2n tokens with a dense (1,1,2n,2n) boolean mask and no flash-causal fast
# path, so padding=True would make every forward a 4096-token dense-attention one whatever the
# real request length. The MDM protocol's padding=True is not transferable.
#
# --- If command 2 OOMs on `race` -----------------------------------------------------------
#
# batch_size=128 materialises a (128, n, 32000) logits tensor twice (bf16 from autocast, then
# .float()), and race's contexts run to nearly 2048 tokens. Drop to --batch_size 32
# (128 % 32 == 0, four forwards of a quarter the size) rather than lowering mc_num, which would
# change the estimator's variance and break comparability with validate().
