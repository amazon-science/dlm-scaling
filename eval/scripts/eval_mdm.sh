#!/usr/bin/env bash
# Zero-shot benchmark evaluation of an MDM checkpoint (Section 4.3 / Appendix B.4).
#
# Three commands: the ten multiple-choice tasks under the exact DUEL likelihood, the same ten
# under the 128-sample MC ELBO, then lambada_standard under chain_rule with the greedy decode.
# The task list and its ORDER are shared with eval_bd3lm.sh / eval_e2d2.sh on purpose -- a mean
# over a different subset is not comparable across variants.
#
# The example below is the compute-optimal MDM cell at C = 1e20. Point CKPT at your own run
# (converted to .safetensors with scripts/pth_to_safetensors.py --arch mdm) and set MODEL.
#
# ⚠️ TASK NAMES / ENV. It is `winogrande`, not `winogrande_xl` (the latter is the HF dataset
#    CONFIG name; lm_eval aborts with "Tasks not found" before scoring anything). `social_iqa`
#    needs HF_DATASETS_TRUST_REMOTE_CODE=true in the environment (custom loading script, which
#    raises at TASK LOAD time under datasets 3.0.1) -- never lm_eval's --trust_remote_code,
#    which also appends trust_remote_code=True to --model_args, an argument the harness
#    rejects. And never double-quote --model_args, or the harness receives the literal
#    "'duel'" and asserts.
# ⚠️ ~79,500 requests per ten-task command (hellaswag alone is 40,168 of them). Budget hours.

MODEL=944
CKPT=workdir/mdm-944M-100.0flop-lowvar/iter-027637-ckpt.safetensors

export HF_DATASETS_TRUST_REMOTE_CODE=true

# --- 1 of 3: DUEL, the exact conditional likelihood ----------------------------------------
#
# nll_type='duel' is the any-order chain rule with the reveal order chosen by top-1 confidence
# and the TRUE token revealed at each step -- the conditional form of the DUEL NLL that
# pretrain/train_mdm.py logs with the same duel_alg='greedy'. Exact rather than the MC bound
# below, and seed-independent. It is NOT comparable with the mc numbers of the same cell, and
# being optimistic it makes any gap against an exact-NLL arm (the ARM) a LOWER bound.
#
# ⚠️ COST. Greedy cannot be batched -- step t's choice depends on step t-1's logits -- so this
#    is len(target) SEQUENTIAL batch-1 forwards per request, each a full 2048-token forward
#    under padding=True. Measure on a small task first. The cheap alternative is
#    nll_type='chain_rule', the SAME estimator at duel_alg='left_to_right' but batched
#    (~1 forward/request) -- a different reveal order, so a different number, not a faster
#    route to this one.
# ⚠️ NO --batch_size: the duel path does not read it (only mc and chain_rule do).
# ⚠️ race REPORTS acc ONLY here. lm_eval's race.yaml declares `metric_list: [acc]` and
#    evaluate_mdm.py's __main__ is a bare cli_evaluate() with no patch_extra_acc_norm()
#    (unlike evaluate_bd3lm.py / evaluate_e2d2.py), so this run cannot fill race's acc_norm.

python eval/evaluate_mdm.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model mdm \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,nll_type='duel',duel_alg='greedy',greddy=False,cfg=0.,padding=True

# --- 2 of 3: the same ten scored by the 128-sample MC ELBO ---------------------------------
#
# mc_num=128 matches validate()'s hardcoded 128 in pretrain/train_mdm.py, putting this on the
# same scale as the metric/val_loss logged during training, and --batch_size 128 makes it
# exactly ONE forward per request (mc_num % batch_size == 0 is asserted, and only the mc path
# reads batch_size at all).

python eval/evaluate_mdm.py \
  --tasks copa,hellaswag,piqa,sciq,arc_easy,social_iqa,race,openbookqa,boolq,winogrande \
  --model mdm --batch_size 128 \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,nll_type='mc',mc_num=128,greddy=False,cfg=0.,padding=True

# --- 3 of 3: lambada_standard needs the greedy decode (greddy=True) ------------------------
#
# lambada goes through loglikelihood(), never generate_until(): lm_eval reads its `perplexity`
# off the log-likelihood and its `acc` off the second element of each tuple, which is
# suffix_greedy_prediction's exact-match bit. lambada's targets are 1-3 tokens, so the decode
# is cheap here; that is why greddy defaults to False for the multiple-choice runs above.

python eval/evaluate_mdm.py \
  --tasks lambada_standard \
  --model mdm --batch_size 128 \
  --model_args model_name=$MODEL,ckpt_path=$CKPT,nll_type='chain_rule',greddy=True,cfg=0.,padding=True

# padding=True is the MDM protocol. It genuinely changes MDM scores (the model is fully
# bidirectional, so the 2048-token MASK tail is visible to every scored position), so it has
# to be held fixed across anything meant to be compared. It is NOT transferable to BD3LM /
# E2D2, where whole extra MASK blocks cannot reach an earlier block.
