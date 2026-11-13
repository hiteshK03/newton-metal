#!/bin/bash
cd "$(dirname "$0")/../.."
PY=${PY:-python}
run() {  # name rsi et fpr jref
  export FORCE_SKILL=2 REF_RSI=$2 REF_ET=$3 REF_FPR=$4 REF_JREF=$5
  INIT=checkpoints/anymal_walk.pt INIT_STD=0.5 $PY -m nmrl.ppo_multi 4096 300 abl_$1.pt > abl_$1.log 2>&1
  for u in 100 200; do $PY experiments/jump/eval_ref.py abl_$1_upd$u.pt "$1@$u" 2>&1 | grep RESULT >> ablation_results.txt; done
  $PY experiments/jump/eval_ref.py abl_$1.pt "$1@300" 2>&1 | grep RESULT >> ablation_results.txt
}
: > ablation_results.txt
run A0_baseline 0 0 0 0
run A1_rsi 1 0 0 0
run A2_rsi_et 1 1 0 0
run A3_rsi_et_fpr 1 1 1 0
run A4_all_jref 1 1 1 1
echo ABLATION_DONE >> ablation_results.txt
