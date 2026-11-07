| model | MAE avg | MAE 1h | MAE 12h | MAE 24h | MAPE avg (%) | pinball 5/50/95 avg | pinball q7 avg | d2 pinball avg | cov 90 avg | width 90 avg | cov 50 avg | fit+predict (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_quantile_q7 (upstream) | 859 | 534 | 1001 | 1042 | 1.75 | 244 | 278 | 0.8595 | 0.766 | 2987 | 0.305 | 1851 |
| kumo_tabular_q7 (teacher run) | 686 | 275 | 827 | 955 | 1.40 | 190 | 213 | 0.8918 | 0.959 | 4298 | 0.594 | 1255 |
| kumo_tabular_q7 + spline recal | 820 | 291 | 1009 | 1158 | 1.64 | 212 | 239 | 0.8875 | 0.859 | 3209 | 0.445 | 1256 |
| kumo_tabular_q7 + spline recal, median kept | 715 | 276 | 868 | 1003 | 1.45 | 194 | 227 | 0.8920 | 0.859 | 3210 | 0.460 | 1256 |
| kumo_tabular_q7 + constant shift | 789 | 284 | 966 | 1116 | 1.61 | 204 | 229 | 0.8906 | 0.883 | 3492 | 0.469 | 1255 |
| kumo_tabular_q7 + constant shift, median kept | 698 | 275 | 842 | 978 | 1.43 | 189 | 220 | 0.8945 | 0.883 | 3493 | 0.488 | 1255 |
| hgb_q7 + lag diffs + residual | 747 | 320 | 913 | 1008 | 1.51 | 216 | 245 | 0.8742 | 0.763 | 2583 | 0.297 | 4336 |
| hgb_q7 + lag diffs + residual + spline recal | 768 | 320 | 945 | 1040 | 1.55 | 209 | 233 | 0.8845 | 0.911 | 3761 | 0.520 | 4337 |
| hgb_q7 + lag diffs + residual + spline recal, median kept | 747 | 320 | 913 | 1008 | 1.51 | 205 | 232 | 0.8854 | 0.911 | 3761 | 0.520 | 4337 |
| hgb_q7 + lag diffs + residual + constant shift | 764 | 320 | 936 | 1034 | 1.54 | 209 | 234 | 0.8833 | 0.913 | 3759 | 0.518 | 4336 |
| hgb_q7 + lag diffs + residual + constant shift, median kept | 747 | 320 | 913 | 1008 | 1.51 | 206 | 233 | 0.8840 | 0.913 | 3759 | 0.518 | 4336 |
| KD CutMix 2:1 (paper recipe) | 847 | 524 | 969 | 1049 | 1.73 | 241 | 273 | 0.8625 | 0.719 | 2584 | 0.286 | 10608 |
| KD CutMix 2:1 + spline recal | 956 | 557 | 1109 | 1202 | 1.92 | 249 | 279 | 0.8672 | 0.876 | 3839 | 0.465 | 10609 |
| KD CutMix 2:1 + spline recal, median kept | 861 | 525 | 990 | 1067 | 1.75 | 234 | 269 | 0.8711 | 0.876 | 3839 | 0.479 | 10609 |
| KD CutMix 2:1 + constant shift | 937 | 541 | 1086 | 1184 | 1.91 | 246 | 275 | 0.8676 | 0.888 | 3963 | 0.477 | 10608 |
| KD CutMix 2:1 + constant shift, median kept | 852 | 524 | 976 | 1055 | 1.74 | 232 | 267 | 0.8712 | 0.888 | 3963 | 0.488 | 10608 |
