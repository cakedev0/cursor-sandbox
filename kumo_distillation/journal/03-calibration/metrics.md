| model | MAE avg | MAE 1h | MAE 12h | MAE 24h | MAPE avg (%) | pinball 5/50/95 avg | pinball q7 avg | d2 pinball avg | cov 90 avg | width 90 avg | cov 50 avg | fit+predict (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_quantile_q7 (upstream) | 859 | 534 | 1001 | 1042 | 1.75 | 244 | 278 | 0.8595 | 0.766 | 2987 | 0.305 | 1851 |
| kumo_tabular_q7 (teacher run) | 686 | 275 | 827 | 955 | 1.40 | 190 | 213 | 0.8918 | 0.959 | 4298 | 0.594 | 1255 |
| kumo_tabular_q7 + spline recal | 820 | 291 | 1009 | 1158 | 1.64 | 212 | 239 | 0.8875 | 0.859 | 3209 | 0.445 | 1256 |
| kumo_tabular_q7 + spline recal, median kept | 715 | 276 | 868 | 1003 | 1.45 | 194 | 227 | 0.8920 | 0.859 | 3210 | 0.460 | 1256 |
| kumo_tabular_q7 + constant shift | 789 | 284 | 966 | 1116 | 1.61 | 204 | 229 | 0.8906 | 0.883 | 3492 | 0.469 | 1255 |
| kumo_tabular_q7 + constant shift, median kept | 698 | 275 | 842 | 978 | 1.43 | 189 | 220 | 0.8945 | 0.883 | 3493 | 0.488 | 1255 |
| hgb_q7 (this pipeline) | 862 | 535 | 995 | 1056 | 1.76 | 243 | 277 | 0.8606 | 0.771 | 3012 | 0.309 | 3994 |
| hgb_q7 (this pipeline) + spline recal | 928 | 545 | 1093 | 1144 | 1.88 | 248 | 277 | 0.8645 | 0.902 | 4284 | 0.511 | 3995 |
| hgb_q7 (this pipeline) + spline recal, median kept | 866 | 535 | 1002 | 1061 | 1.77 | 238 | 272 | 0.8670 | 0.902 | 4284 | 0.516 | 3995 |
| hgb_q7 (this pipeline) + constant shift | 910 | 541 | 1063 | 1127 | 1.86 | 244 | 274 | 0.8659 | 0.906 | 4293 | 0.509 | 3994 |
| hgb_q7 (this pipeline) + constant shift, median kept | 862 | 535 | 996 | 1057 | 1.76 | 236 | 270 | 0.8679 | 0.906 | 4293 | 0.515 | 3994 |
| hgb_q7 + lag diffs + residual | 747 | 320 | 913 | 1008 | 1.51 | 216 | 245 | 0.8742 | 0.763 | 2583 | 0.297 | 4336 |
| hgb_q7 + lag diffs + residual + spline recal | 768 | 320 | 945 | 1040 | 1.55 | 209 | 233 | 0.8845 | 0.911 | 3761 | 0.520 | 4337 |
| hgb_q7 + lag diffs + residual + spline recal, median kept | 747 | 320 | 913 | 1008 | 1.51 | 205 | 232 | 0.8854 | 0.911 | 3761 | 0.520 | 4337 |
| hgb_q7 + lag diffs + residual + constant shift | 764 | 320 | 936 | 1034 | 1.54 | 209 | 234 | 0.8833 | 0.913 | 3759 | 0.518 | 4336 |
| hgb_q7 + lag diffs + residual + constant shift, median kept | 747 | 320 | 913 | 1008 | 1.51 | 206 | 233 | 0.8840 | 0.913 | 3759 | 0.518 | 4336 |
| KD real rows only | 846 | 510 | 976 | 1053 | 1.73 | 313 | 341 | 0.7876 | 0.387 | 956 | 0.114 | 5466 |
| KD real rows only + spline recal | 908 | 518 | 1067 | 1140 | 1.84 | 240 | 270 | 0.8697 | 0.899 | 3764 | 0.502 | 5466 |
| KD real rows only + spline recal, median kept | 851 | 510 | 980 | 1062 | 1.74 | 230 | 265 | 0.8721 | 0.899 | 3764 | 0.505 | 5466 |
| KD real rows only + constant shift | 895 | 516 | 1051 | 1120 | 1.83 | 239 | 270 | 0.8686 | 0.902 | 3810 | 0.504 | 5466 |
| KD real rows only + constant shift, median kept | 848 | 510 | 976 | 1059 | 1.74 | 231 | 266 | 0.8705 | 0.902 | 3810 | 0.506 | 5466 |
| KD CutMix 2:1 (paper recipe) | 847 | 524 | 969 | 1049 | 1.73 | 241 | 273 | 0.8625 | 0.719 | 2584 | 0.286 | 10608 |
| KD CutMix 2:1 + spline recal | 956 | 557 | 1109 | 1202 | 1.92 | 249 | 279 | 0.8672 | 0.876 | 3839 | 0.465 | 10609 |
| KD CutMix 2:1 + spline recal, median kept | 861 | 525 | 990 | 1067 | 1.75 | 234 | 269 | 0.8711 | 0.876 | 3839 | 0.479 | 10609 |
| KD CutMix 2:1 + constant shift | 937 | 541 | 1086 | 1184 | 1.91 | 246 | 275 | 0.8676 | 0.888 | 3963 | 0.477 | 10608 |
| KD CutMix 2:1 + constant shift, median kept | 852 | 524 | 976 | 1055 | 1.74 | 232 | 267 | 0.8712 | 0.888 | 3963 | 0.488 | 10608 |
| KD out-of-fold targets, CutMix 2:1 | 905 | 544 | 1031 | 1139 | 1.85 | 247 | 275 | 0.8628 | 0.942 | 5256 | 0.543 | 13450 |
| KD out-of-fold targets, CutMix 2:1 + spline recal | 973 | 565 | 1114 | 1240 | 1.97 | 253 | 283 | 0.8650 | 0.891 | 4236 | 0.480 | 13451 |
| KD out-of-fold targets, CutMix 2:1 + spline recal, median kept | 901 | 544 | 1027 | 1131 | 1.84 | 241 | 275 | 0.8681 | 0.892 | 4237 | 0.497 | 13451 |
| KD out-of-fold targets, CutMix 2:1 + constant shift | 950 | 556 | 1081 | 1215 | 1.95 | 249 | 278 | 0.8662 | 0.903 | 4447 | 0.498 | 13450 |
| KD out-of-fold targets, CutMix 2:1 + constant shift, median kept | 902 | 543 | 1025 | 1138 | 1.85 | 241 | 273 | 0.8682 | 0.903 | 4447 | 0.511 | 13450 |
| KD lag diffs + residual, CutMix 2:1 | 763 | 371 | 897 | 1019 | 1.55 | 224 | 253 | 0.8699 | 0.662 | 2085 | 0.252 | 17580 |
| KD lag diffs + residual, CutMix 2:1 + spline recal | 856 | 389 | 1026 | 1154 | 1.72 | 223 | 248 | 0.8817 | 0.872 | 3345 | 0.463 | 17580 |
| KD lag diffs + residual, CutMix 2:1 + spline recal, median kept | 772 | 371 | 912 | 1033 | 1.56 | 209 | 241 | 0.8852 | 0.872 | 3346 | 0.471 | 17580 |
| KD lag diffs + residual, CutMix 2:1 + constant shift | 840 | 379 | 999 | 1141 | 1.71 | 219 | 246 | 0.8826 | 0.886 | 3438 | 0.476 | 17580 |
| KD lag diffs + residual, CutMix 2:1 + constant shift, median kept | 765 | 371 | 901 | 1022 | 1.55 | 207 | 240 | 0.8857 | 0.886 | 3438 | 0.483 | 17580 |
