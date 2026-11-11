# 024: k-means++ and Lloyd in CuPy, with and without custom kernels

How close can CuPy code get to cuML on k-means? This iteration measures two versions of
greedy k-means++ and of Lloyd iterations, on one NVIDIA L4:

- **cupy-ops** uses CuPy array operations only (GEMM, elementwise ops, reductions,
  `argmin`, `cumsum`, `searchsorted`, `add.at`). It stands for what a generic array API
  implementation can express.
- **cupy-fused** runs CUDA kernels that CuPy compiles at run time with NVRTC. It needs no
  build step, so an array API estimator could dispatch to it for CuPy input, the same way
  it already dispatches the centroid update to `native_accumulate`.

On the benchmarked cases:

- cupy-fused k-means++ is 3.5x to 3.8x faster than cuML on 128- to 512-dimensional
  embeddings, and ~26x faster on 1M × 3 data. cupy-ops is 1.6x to 2.8x faster than cuML.
  The current portable `kmeans_plusplus` ranges from 0.3x to 1.9x.
- cupy-fused Lloyd iterations are 1.0x to 1.3x faster than cuML's, and ~2.5x on 1M × 3.
  cupy-ops iterations are 1.3x to 1.7x slower than cuML's (3x slower on 1M × 3).
- A full fit with `n_init=10`, `max_iter=100` is 1.7x to 2.6x faster with cupy-fused than
  with cuML (6x on 1M × 3), and 1.15x to 1.5x faster with cupy-ops (1.4x slower on 1M × 3).
- Init and final inertias match cuML's within the spread across seeds.

## What was added

At commit [`0646406`](https://github.com/ogrisel/cursor-sandbox/commit/06464065a85d52e4708297a9730fb3f34258e87e):

- [`cupy_plusplus.py`](https://github.com/ogrisel/cursor-sandbox/blob/06464065a85d52e4708297a9730fb3f34258e87e/kmeans_lloyd/src/kmeans_lloyd/cupy_plusplus.py)
  and [`cupy_plusplus.cu`](https://github.com/ogrisel/cursor-sandbox/blob/06464065a85d52e4708297a9730fb3f34258e87e/kmeans_lloyd/src/kmeans_lloyd/cupy_plusplus.cu):
  `kmeans_plusplus_cupy` and `kmeans_plusplus_cupy_fused`, with the signature of the
  portable `kmeans_plusplus` plus an optional `sample_weight`.
- [`cupy_lloyd.py`](https://github.com/ogrisel/cursor-sandbox/blob/06464065a85d52e4708297a9730fb3f34258e87e/kmeans_lloyd/src/kmeans_lloyd/cupy_lloyd.py)
  and [`cupy_lloyd.cu`](https://github.com/ogrisel/cursor-sandbox/blob/06464065a85d52e4708297a9730fb3f34258e87e/kmeans_lloyd/src/kmeans_lloyd/cupy_lloyd.cu):
  `lloyd_cupy`, `lloyd_cupy_fused`, and `kmeans_cupy`, which runs k-means++ then Lloyd
  `n_init` times and keeps the best run.
- `native_kmeans_plusplus` in [`backends.py`](https://github.com/ogrisel/cursor-sandbox/blob/06464065a85d52e4708297a9730fb3f34258e87e/kmeans_lloyd/src/kmeans_lloyd/backends.py):
  `KMeans(init="k-means++")` now uses the fused initializer on float32 CuPy input. The
  estimator's Lloyd loop is unchanged.
- Property tests in `tests/test_cupy_plusplus.py` and `tests/test_cupy_lloyd.py`, and the
  benchmark scripts `scripts/bench_plusplus_cupy_cuml.py` and
  `scripts/bench_lloyd_cupy_cuml.py`.

How the versions work:

- **k-means++, both versions:** no host synchronization in the loop. The sampled
  candidates and the chosen index stay on the device, so Python only queues work. The
  portable `kmeans_plusplus` reads the argmin and a cumulative sum back at every step.
- **k-means++, cupy-fused:** three kernels per center. The first reads `X` once: it
  updates the closest distances with the center chosen at the previous step, then sums the
  candidates' costs per chunk of 256 rows. The second picks the best candidate and finds
  the chunk of each new draw; the third picks the row inside that chunk. Distances are sums
  of squared differences, so there is no cancellation and a row equal to a center gets a
  distance of exactly 0.
- **Lloyd, cupy-ops:** the augmented GEMM `[X, 1] @ [-2 C, |c|^2].T` from
  [018](../018-cpu-array-api-optimization/report.md), tiled over rows, then `argmin`,
  `add.at` and `bincount`.
- **Lloyd, cupy-fused:** one kernel per iteration, a tiled FP32 GEMM (128 × 128 tiles,
  8 × 8 per thread, double-buffered shared memory, no tensor cores) that keeps a running
  argmin over the center tiles. The `n × k` score matrix is never written. The same kernel
  then adds each row to the sum of its new center with atomics.
- **Stopping rule:** both Lloyd versions follow scikit-learn's, a squared center shift of
  at most `tol` times the mean feature variance. Empty clusters keep their center, as in
  cuML.

For context, an nsys profile of cuML's k-means++ on GloVe at k=256 gives ~15 ms per center:
9.8 ms in a cutlass GEMM of `X` against the 8 candidates, which runs at ~40% of the L4's
memory bandwidth, and 4.2 ms recomputing the row norms of `X`. A step of cupy-fused reads
`X` once, at ~240 GB/s of the ~258 GB/s a plain read kernel gets on this GPU.

## Setup

- One NVIDIA L4 per host, driver 595.112.07, cuML 26.10.00 (`libcuvs` 26.10.0), CuPy 14.2.0,
  float32, TF32 disabled, data already on the device before the timer starts. The
  k-means++ benchmark ran on aws-l4-3. The Lloyd and full-fit benchmarks ran on aws-l4-2,
  the same instance type. Every run held the host's whole-machine lock.
- Datasets: GloVe-200 (1,183,514 × 200), SIFT1M (1,000,000 × 128), Wolt CLIP ViT-B/32
  (1,000,000 × 512, L2-normalized, 23% of the first 200,000 rows are duplicates), and two
  standard normal sets,
  150,000 × 128 and 1,000,000 × 3. These are the converted files from the real-embedding
  runs on PR 17.
- Variants of one comparison are interleaved, in a rotating order, for each seed.
- cuML's k-means++ is timed as `KMeans(init="k-means++", n_init=1, max_iter=0)`, which
  returns the initial centers. That fit also validates the input and computes the
  inertia, which costs ~0.05 s on GloVe.
- Every k-means++ variant uses cuVS's trial count, `2 + ceil(log(k))`, so all of them do
  the same work. The repository default stays scikit-learn's `2 + int(log(k))`.
- cuML switches CuPy's allocator to `rmm_cupy_allocator`, which pays a `cudaMalloc` and a
  synchronizing `cudaFree` per temporary array. That made an earlier GEMM-based CuPy
  k-means++ ~2.6x slower on the 150,000 × 128 set, so the CuPy variants run under CuPy's
  own memory pool, as they would in a program without cuML.
- The L4 runs at its 72 W power cap under sustained load, with SM clocks at 1.2 to 1.4 GHz
  instead of up to 2 GHz. That slows the compute-bound GEMMs (cuML and both Lloyd
  versions) more than the memory-bound k-means++ steps.

## k-means++ initialization

Median time over seeds 0 to 4 for one initialization, and speed-up over cuML.

| data (n × d) | k | cuML | `kmeans_plusplus` (portable) | cupy-ops | cupy-fused |
| --- | ---: | ---: | ---: | ---: | ---: |
| GloVe (1,183,514 × 200) | 256 | 3.86 s | 3.07 s (1.3x) | 1.47 s (2.6x) | 1.02 s (3.8x) |
| GloVe | 1024 | 15.7 s | 13.3 s (1.2x) | 6.24 s (2.5x) | 4.15 s (3.8x) |
| SIFT1M (1,000,000 × 128) | 256 | 2.01 s | 1.94 s (1.0x) | 0.83 s (2.4x) | 0.53 s (3.8x) |
| SIFT1M | 1024 | 8.12 s | 9.50 s (0.9x) | 3.53 s (2.3x) | 2.12 s (3.8x) |
| CLIP (1,000,000 × 512) | 256 | 7.39 s | 3.98 s (1.9x) | 2.61 s (2.8x) | 2.10 s (3.5x) |
| CLIP | 1024 | 29.8 s | 17.4 s (1.7x) | 10.7 s (2.8x) | 8.41 s (3.5x) |
| normal (150,000 × 128) | 256 | 0.340 s | 0.331 s (1.0x) | 0.134 s (2.5x) | 0.092 s (3.7x) |
| normal | 1024 | 1.36 s | 1.34 s (1.0x) | 0.555 s (2.4x) | 0.385 s (3.5x) |
| normal (1,000,000 × 3) | 256 | 0.560 s | 1.45 s (0.4x) | 0.298 s (1.9x) | 0.021 s (27x) |
| normal | 1024 | 2.33 s | 7.39 s (0.3x) | 1.43 s (1.6x) | 0.088 s (26x) |

- The first cupy-fused call for a new (d, number of trials) pair compiles the kernels,
  0.4 to 0.75 s with an empty CuPy cache. CuPy caches the binary on disk afterwards.
- The two CuPy versions use the same random draws. They end with the same init inertia,
  so the same centers, on 28 of the 50 runs. The others diverge after a draw that lands within rounding of a row boundary,
  where the expanded-form distances of cupy-ops and the direct distances of cupy-fused
  round differently.
- The fused kernel keeps the candidates of a step in shared memory, which limits it to
  ~800 features at the default trial count. Wider data falls back to the portable path.

## Lloyd iterations

All three versions start from the same cupy-fused k-means++ centers, with `max_iter=100`
and `tol=1e-9`. Median time per iteration over seeds 0 to 2, and speed-up over cuML.

| data (n × d) | k | cuML | cupy-ops | cupy-fused |
| --- | ---: | ---: | ---: | ---: |
| GloVe (1,183,514 × 200) | 256 | 23.0 ms | 33.3 ms (0.69x) | 19.7 ms (1.17x) |
| GloVe | 1024 | 70.9 ms | 105 ms (0.67x) | 57.4 ms (1.24x) |
| SIFT1M (1,000,000 × 128) | 256 | 12.8 ms | 20.3 ms (0.63x) | 10.0 ms (1.29x) |
| SIFT1M | 1024 | 39.3 ms | 65.8 ms (0.60x) | 30.2 ms (1.30x) |
| CLIP (1,000,000 × 512) | 256 | 44.1 ms | 61.0 ms (0.72x) | 44.6 ms (0.99x) |
| CLIP | 1024 | 134 ms | 175 ms (0.76x) | 122 ms (1.10x) |
| normal (150,000 × 128) | 256 | 2.1 ms | 3.0 ms (0.71x) | 1.6 ms (1.34x) |
| normal | 1024 | 6.4 ms | 11.0 ms (0.58x) | 5.0 ms (1.27x) |
| normal (1,000,000 × 3) | 256 | 3.2 ms | 9.5 ms (0.34x) | 1.3 ms (2.50x) |
| normal | 1024 | 10.6 ms | 34.4 ms (0.31x) | 4.4 ms (2.43x) |

- Every run did 100 iterations, except on the 150,000 × 128 set at k=1024: 67 iterations
  for cuML and 58 for both CuPy versions on average. cuML's rule stops when the float32
  inertia stops decreasing, since `1 - 1e-9` rounds to 1.
- The times include each call's fixed costs: copying the centers, row norms, the
  tolerance, and the final assignment.
- In a separate check on aws-l4-2 (20 iterations from random centers, not in the result
  files), the fused assignment took 0.85x to 1.3x the time of cuBLAS's GEMM `X @ C.T`
  alone on the eight high-dimensional cases. That GEMM also has to write the `n × k`
  result. A first version without double buffering and 16-byte loads took 1.1x to 2.1x
  the GEMM time and only matched cuML.
- cupy-ops takes about twice the time of that GEMM (33 ms against ~14 to 16 ms on GloVe at
  k=256). The rest goes to the `argmin` pass over the `n × k` scores and to `add.at`. At
  d=3 the GEMM itself is the problem: cuBLAS takes 4.4 ms for `X @ C.T` at k=256, against
  1.1 ms for the whole fused assignment.

## Full fit

`n_init=10`, `max_iter=100`, `tol=1e-9`, seed 0, one fit per variant. cuML runs
`KMeans(init="k-means++")`. The CuPy variants run `kmeans_cupy`, with cupy-ops or
cupy-fused for both the initialization and Lloyd. CLIP at k=1024 was left out: one cuML fit
takes ~9 minutes there.

| data (n × d) | k | cuML | cupy-ops | cupy-fused |
| --- | ---: | ---: | ---: | ---: |
| GloVe (1,183,514 × 200) | 256 | 63.4 s | 51.4 s (1.23x) | 32.1 s (1.98x) |
| GloVe | 1024 | 232 s | 174 s (1.33x) | 101 s (2.29x) |
| SIFT1M (1,000,000 × 128) | 256 | 33.3 s | 29.0 s (1.15x) | 15.4 s (2.16x) |
| SIFT1M | 1024 | 121 s | 105 s (1.15x) | 51.2 s (2.36x) |
| CLIP (1,000,000 × 512) | 256 | 116 s | 88.7 s (1.31x) | 66.8 s (1.74x) |
| normal (150,000 × 128) | 256 | 5.5 s | 4.4 s (1.25x) | 2.6 s (2.11x) |
| normal | 1024 | 18.2 s | 12.5 s (1.46x) | 7.0 s (2.60x) |
| normal (1,000,000 × 3) | 256 | 8.9 s | 12.5 s (0.71x) | 1.4 s (6.2x) |
| normal | 1024 | 34.2 s | 48.5 s (0.71x) | 5.1 s (6.7x) |

- All 10 runs of each CuPy fit did 100 iterations, except on the 150,000 × 128 set at
  k=1024: 55 to 74 for cupy-ops and 51 to 95 for cupy-fused. cuML only reports the
  iterations of its best run (74 there, 100 elsewhere).
- The best inertias are within 0.1% of cuML's, above or below depending on the case.

## Quality

- **k-means++ benchmark:** each initialization was followed by the same cuML Lloyd run
  (`max_iter=100`, `tol=1e-9`). Mean final inertia of every variant is within 0.09% of
  cuML's. The standard deviation of cuML's own final inertia across seeds is 0.008% to
  0.10%. Mean init inertias also agree within that spread.
- **Lloyd benchmark:** from the same centers, both CuPy versions end within 0.012% of
  cuML's mean final inertia, against a spread across seeds of 0.002% to 0.073%.
- **Tests:** both k-means++ versions pick the same rows as a float64 NumPy implementation
  with the same random draws, with and without sample weights. The fused assignment finds
  the nearest center up to float32 rounding, with k=300 to cover partial center tiles.

## Not covered

- PyTorch. A CUDA tensor could reach the same kernels through `cupy.from_dlpack` without a
  copy.
- Other GPUs, float64, and `sample_weight` in Lloyd.
- The estimator's own Lloyd loop: `KMeans.fit` uses the fused k-means++, but its iterations
  are still the portable ones measured in 015.
- Run-to-run reproducibility: the fused Lloyd sums, like `add.at`, use atomics. cuML's
  k-means++ also gave different centers for the same `random_state` across repeated runs.
- FAISS, whose GPU `Kmeans` is another reference, and e-means
  ([arXiv 2611.09501](https://arxiv.org/abs/2611.09501)), a mini-batch variant in FAISS
  that its authors report reaching Lloyd's loss in ~10x fewer epochs. It changes the
  algorithm, so it needs a time-to-inertia comparison rather than time per iteration.

The raw measurements are in [init_runs.jsonl](init_runs.jsonl) and
[lloyd_and_fit_runs.jsonl](lloyd_and_fit_runs.jsonl), one JSON line per fit, with library
versions.

## Next steps

I think the next step should be either:

- wire `lloyd_cupy_fused` into `KMeans.fit` behind the same native-backend switch, and
  rerun the six-task mix of 015 against cuML on the estimator itself;
- or add FAISS GPU `Kmeans` and e-means to these benchmarks, comparing the time to reach a
  given inertia.
