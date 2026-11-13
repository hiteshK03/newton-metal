#!/bin/bash
cd "$(dirname "$0")/../.."
PY=${PY:-python}
: > curriculum_results.txt
ev() { $PY experiments/jump/eval_curr.py "$1" "$2" 2>&1 | grep RESULT | sed "s/^RESULT/RESULT [$3]/" >> curriculum_results.txt; }
ev checkpoints/experimental/anymal_multi.pt 0.25 "start (stage-5 policy)"
ev checkpoints/experimental/anymal_multi.pt 0.5 "start (stage-5 policy)"
stage() { # name lam_max updates init eval_lams...
  export FORCE_SKILL=2 REF_RSI=0 REF_ET=0 REF_FPR=0 REF_JREF=0 LAM_MIN=0.1 LAM_MAX=$2
  INIT=$4 INIT_STD=0.4 $PY -m nmrl.ppo_multi 4096 $3 curr_$1.pt > curr_$1.log 2>&1
  shift 4
  for l in "$@"; do ev curr_$1.pt $l "$1 done"; done
}
stage C1 0.25 200 checkpoints/experimental/anymal_multi.pt
ev curr_C1.pt 0.25 "C1 (max lam 0.25)"; ev curr_C1.pt 0.5 "C1 (max lam 0.25)"
stage C2 0.5 200 curr_C1.pt
ev curr_C2.pt 0.25 "C2 (max lam 0.5)"; ev curr_C2.pt 0.5 "C2 (max lam 0.5)"; ev curr_C2.pt 1.0 "C2 (max lam 0.5)"
stage C3 1.0 300 curr_C2.pt
ev curr_C3.pt 0.5 "C3 (max lam 1.0)"; ev curr_C3.pt 0.75 "C3 (max lam 1.0)"; ev curr_C3.pt 1.0 "C3 (max lam 1.0)"
echo CURRICULUM_DONE >> curriculum_results.txt
