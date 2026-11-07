#!/bin/bash
# Student runs (run from scripts/). Usage: run_students.sh <queue name>
set -e
out=../results/study
run() { name=$1; shift; test -f $out/${name}_scores.csv && return; python -u distill.py --output_dir $out --name $name "$@" 2>&1 | grep --line-buffered -v -i warn | tee ../results/raw/$name.log; }
case $1 in
  local)
    run hgb_q7_lagdiff_resid --supervision labels --ratio 0 --residual --lag_diffs
    run kd_hgb_cutmix2 --ratio 2 --wait
    run kd_hgb_real_only --ratio 0 --wait
    run hgb_q7_labels --supervision labels --ratio 0
    ;;
  remote)
    run kd_hgb_lagdiff_resid_cutmix2 --ratio 2 --residual --lag_diffs
    run kd_hgb_lagdiff_resid_real_only --ratio 0 --residual --lag_diffs
    ;;
esac
