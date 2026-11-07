#!/bin/bash
# One process per fold/horizon: Kumo-Tabular's cached fit keeps the context
# key/value cache in pinned host memory, which is only released reliably when
# the process exits (the largest fold needs ~8.4 GB on a 15 GB host).
# usage: run_teacher.sh <script> <output_dir> [extra args...]
script=$1; out=$2; shift 2
for fold in 0 1 2 3 4 5 6 7 -4 -3 -2 -1; do
  for h in 1 12 24; do
    test -f "$out/fold${fold}_h${h}.npz" && continue
    python "$script" --output_dir "$out" --folds "$fold" --horizons "$h" "$@" || exit 1
  done
done
