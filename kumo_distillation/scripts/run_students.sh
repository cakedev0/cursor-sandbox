#!/bin/bash
# Student runs of iteration 2, in priority order (run from scripts/).
set -e
out=../results/study
run() { name=$1; shift; test -f $out/${name}_scores.csv && return; python distill.py --output_dir $out --name $name "$@" 2>&1 | grep -v -i warn | tee ../results/raw/$name.log; }
run hgb_q7_lagdiff_resid --supervision labels --ratio 0 --residual --lag_diffs
run kd_hgb_cutmix2 --ratio 2 --wait
run kd_hgb_lagdiff_resid_cutmix2 --ratio 2 --residual --lag_diffs --wait
run kd_hgb_real_only --ratio 0 --wait
run kd_hgb_lagdiff_resid_real_only --ratio 0 --residual --lag_diffs --wait
