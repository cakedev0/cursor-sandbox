# Lloyd k-means in the Python array API

## What this PR explores

This experiment studies nearest-centroid assignment, Lloyd updates, empty
clusters, and centroid initialization through the Python array API. The eager
implementation runs with PyTorch on CPU, CUDA, and XPU, with CuPy on CUDA,
and with dpnp on a Level Zero GPU.
Comparisons cover stock scikit-learn, scikit-learn with k-means++ performance
patches, Cython, and cuML.

The journal progresses from kernel-fusion probes to bounded-memory tiled
assignment, full fits, repeated-seed and shared-start comparisons, empty-center
relocation, and greedy versus scalable initialization. The current six-task mix
spans 3 to 3000 clusters, 2 to 1000 features, and 135000 to 2100000 samples,
including a balanced 256-cluster, 128-feature case. Runs use float32, eager
execution, TF32 disabled, three seeds, and a 60-second worker limit.

## Latest results

[024](024-cupy-kmeans-plusplus-and-lloyd/index.html) measures k-means++ and Lloyd
iterations written with CuPy array operations only, and with CUDA kernels that CuPy
compiles at run time, against cuML 26.10 on an L4. With the kernels, a full fit
(`n_init=10`, `max_iter=100`) is 1.7x to 2.6x faster than cuML on 128- to 512-dimensional
embeddings, and ~6x faster on 1M × 3 data. With array operations only, it is 1.15x to 1.5x
faster, except at d=3 where it is 1.4x slower. Most of the gain comes from k-means++, which
is 3.5x to 3.8x faster than cuML's with the kernels and 1.6x to 2.8x faster with array
operations.

[023](023-dpnp-array-api/index.html) adds the portable array-API path on dpnp
GPU arrays to the [022](022-intel-arc-xpu/index.html) Arc B390 measurements.
022 holds PyTorch XPU (native and one-hot), PyTorch CPU, stock scikit-learn
1.9.1, and cakedev0. CuPy and cuML are absent on this host.

Main conclusions:

- dpnp, which always uses the one-hot update, is faster than native Torch CPU
  on every full fit. It is also faster than the XPU one-hot update at 17
  clusters (2.57 s against 2.83 s) and at 256 (1.11 s against 1.87 s).
- The XPU one-hot update remains the fastest full fit at 3 clusters (3.22 s;
  dpnp is 4.83 s). From 17 clusters up, cakedev0 is still the fastest full fit.
  dpnp is second at 17 clusters (2.57 s against 2.35 s) and at 256 (1.11 s
  against 1.06 s).
- At 256 clusters dpnp Lloyd takes 0.47 s, level with stock scikit-learn and
  cakedev0. At 550 and 3000 clusters both XPU updates and Cython Lloyd are
  faster than dpnp.
- On Arc B390 the one-hot PyTorch update is faster than `index_add_` through
  256 clusters (022). The CUDA few-cluster cutoff would keep `index_add_` on
  `k17-d180`, where the one-hot update is the faster XPU path.

## Journal

- [001 — fusion probe](001-fusion-probe/index.html): eager and compiled distance-plus-argmin kernels.
- [002 — large-shape OOM](002-large-shape-oom/index.html): which schedules avoid the full distance matrix.
- [003 — three shapes](003-resized-shapes/index.html): sample-, centroid-, and feature-heavy fusion behavior.
- [004 — tile trade-offs](004-tile-tradeoff/index.html): bounded-memory tiles, Cython Lloyd, and k-means++.
- [005 — full k-means](005-full-kmeans/index.html): eager array API, Cython, patched scikit-learn, and cuML.
- [006 — benchmark breadth](006-breadth/index.html): structure and cluster-count sensitivity across five implementations.
- [007 — repeated seeds and shared centers](007-seeded-fixed-centers/index.html): uncertainty bars and initializer-independent Lloyd comparisons.
- [008 — empty-center task mix](008-empty-reassignment-task-mix/index.html): stochastic relocation across five bounded tasks.
- [009 — k-means++ only](009-kmeans-plusplus-only/index.html): timing, memory, and quality before Lloyd.
- [010 — scalable initialization](010-scalable-initialization/index.html): farthest relocation, cuML initializer options, and array-API k-means||.
- [011 — consolidated benchmark](011-consolidated-benchmark/index.html): initialization, Lloyd-only, and full-fit results on one page, plus the scalable-init ablation.
- [012 — source link audit](012-source-link-audit/index.html): source links pinned to the commits that each report measured.
- [013 — large-k Lloyd optimization](013-large-k-lloyd-optimization/index.html): scalable-init sensitivity, cuML kernel analysis, optimization probes, and next steps.
- [014 — portable array-API KMeans](014-portable-array-api-kmeans/index.html): sklearn-compatible estimator extraction and CuPy-native centroid-update probes.
- [015 — native backend updates in the estimator](015-native-backend-estimator-benchmark/index.html): PyTorch and CuPy scatter-add updates, the CUDA contention crossover, and the full benchmark rerun.
- [016 — GEMM versus broadcast distances](016-gemm-vs-broadcast-distances/index.html): time, memory, and quality across initialization, Lloyd, and full fit on CPU and CUDA.
- [017 — GEMM cancellation fallback](017-gemm-cancellation-fallback/index.html): translated-data failures, threshold sensitivity, and selective direct-distance repair.
- [018 — backend-neutral CPU optimizations](018-cpu-array-api-optimization/index.html): labels-only assignment, tiling, interleaving, augmented GEMM, compilation, and sampling experiments.
- [019 — CPU and GPU optimization rerun](019-cpu-gpu-optimization-rerun/index.html): the 018 optimizations on three backends and five tasks, beside cakedev0 on the same host.
- [020 — balanced clusters and features](020-balanced-kd-device-resident-cuml/index.html): a `k256-d128` task, stock-scikit-learn calibration, and a device-resident cuML comparison.
- [021 — device-resident cuML on the original five tasks](021-device-resident-cuml-five-tasks/index.html): the same placement protocol as 020, applied to the cuML cells that 015 timed from host memory.
- [022 — PyTorch XPU on Arc B390](022-intel-arc-xpu/index.html): the six-task mix on an Intel Arc B390, with PyTorch CPU, stock scikit-learn 1.9.1, and cakedev0 on the same laptop.
- [023 — dpnp array API on Arc B390](023-dpnp-array-api/index.html): the portable one-hot update on dpnp GPU arrays, beside the 022 measurements.
- [024: k-means++ and Lloyd in CuPy](024-cupy-kmeans-plusplus-and-lloyd/index.html): CuPy array operations against JIT CUDA kernels, both against cuML, for the initialization, Lloyd iterations, and full fits.
