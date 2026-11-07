# Distilling Kumo-Tabular into HistGradientBoosting quantile models

Electricity load forecasting benchmark of
[probabl-ai/skore-example-electricity-load](https://github.com/probabl-ai/skore-example-electricity-load)
(walk-forward CV with forecast weather, 8 folds of 13 weeks, horizons 1h /
12h / 24h). The question: can a family of scikit-learn
`HistGradientBoostingRegressor` models, distilled from Kumo-Tabular with the
CutMix recipe of [arXiv:2610.01435](https://arxiv.org/html/2610.01435),
close the gap between `hgb_quantile_q7` and `kumo_tabular`? And does
post-hoc calibration help on top?

Status: intermediate results. The runs listed under "In progress" are still
training.

## Iterations

1. [Reproduction of `kumo_tabular` and `hgb_quantile_q7`](01-reproduction/report.md)
   on an AWS L4 GPU.
2. [Distillation into HGB students](02-distillation/report.md): the paper
   recipe, teacher diagnostics and single-fold probes.
3. [Post-hoc calibration](03-calibration/report.md) from past out-of-fold
   predictions.

## Consolidated metrics so far

Pooled over the 8 evaluation folds. MAE and width in MW; "pinball q7" is the
mean pinball loss over the 7 levels and 3 horizons. Lower is better except
for d2 (higher is better) and coverage (target 0.90).

| model | MAE avg | MAE 1h | MAE 12h | MAE 24h | pinball 5/50/95 avg | pinball q7 avg | d2 pinball avg | cov 90 avg | width 90 avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_quantile_q7 (upstream) | 859 | 534 | 1001 | 1042 | 244 | 278 | 0.8595 | 0.766 | 2987 |
| kumo_tabular_q7 (teacher run) | 686 | 275 | 827 | 955 | 190 | 213 | 0.8918 | 0.959 | 4298 |
| KD CutMix 2:1 (paper recipe) | 847 | 524 | 969 | 1049 | 241 | 273 | 0.8625 | 0.719 | 2584 |
| KD CutMix 2:1 + constant shift, median kept | 852 | 524 | 976 | 1055 | 232 | 267 | 0.8712 | 0.888 | 3963 |
| hgb_q7 + lag diffs + residual | 747 | 320 | 913 | 1008 | 216 | 245 | 0.8742 | 0.763 | 2583 |
| hgb_q7 + lag diffs + residual + spline recal, median kept | 747 | 320 | 913 | 1008 | 205 | 232 | 0.8854 | 0.911 | 3761 |

## Findings so far

- Both models are reproduced: `hgb_quantile_q7` exactly (MAE 859.0 MW), and
  `kumo_tabular` within its ensemble's run-to-run noise (683 vs 688 MW
  published).
- The paper's distillation recipe (squared-error students on teacher
  quantiles only, CutMix 2:1) closes only 7% of the MAE gap and 8% of the q7
  pinball gap. It also makes the under-coverage worse (0.72).
- The recipe assumes a teacher whose training-row predictions look like its
  test predictions, and queries on the data manifold. Neither holds here:
  - The teacher memorizes its in-context training rows (35-51 MW error,
    intervals 4 to 16 times narrower than on test rows).
  - CutMix of two random hours yields incoherent rows on which the teacher
    is 3 to 10 times wider than on test rows.
  - Blocked out-of-fold teacher predictions match the test statistics. They
    gave the best fold-0 probe, and a full run is in progress.
- The largest gain so far does not come from distillation but from features.
  Lag differences and a residual target relative to the last observed load
  close 65% of the MAE gap (747 MW), most of it at 1h (320 vs 534 MW).
- Calibration from past folds helps the under-covering HGB models when the
  median is left untouched: 90% coverage reaches 0.88-0.91 and the q7
  pinball loss drops by 4-13 MW. Shifting the median always hurts, and
  calibrating the over-covering teacher does not help.

## In progress

- `KD real rows only`: in-context teacher targets, no synthetic rows.
- `hgb_q7 (this pipeline)`: label-trained baseline on all 12 folds, needed to
  calibrate it.
- `KD lag diffs + residual, CutMix 2:1`.
- `KD out-of-fold targets, CutMix 2:1` and `KD lag diffs + residual,
  out-of-fold targets, CutMix 2:1`.
- Calibration of the runs above.
