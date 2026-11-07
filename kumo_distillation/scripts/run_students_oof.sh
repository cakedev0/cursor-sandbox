#!/bin/bash
# Students whose real rows are labelled by the blocked out-of-fold teacher
# (gen_teacher_oof.py) instead of the full-context one, which memorizes them.
# Run from scripts/. Usage: run_students_oof.sh <queue name>
set -e
out=../results/study
oof=../results/raw/teacher_oof4
run() { name=$1; shift; test -f $out/${name}_scores.csv && return; python -u distill.py --output_dir $out --name $name --oof_dir $oof --wait "$@" 2>&1 | grep --line-buffered -v -i warn | tee ../results/raw/$name.log; }
case $1 in
  local)
    run kd_hgb_oof_cutmix2 --ratio 2
    ;;
  remote)
    run kd_hgb_lagdiff_resid_oof_cutmix2 --ratio 2 --residual --lag_diffs
    ;;
esac
