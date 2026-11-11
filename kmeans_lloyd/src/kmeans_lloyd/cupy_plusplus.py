"""Greedy k-means++ initialization for CuPy arrays, without host synchronization.

Both functions implement the algorithm of :func:`kmeans_lloyd.iteration.kmeans_plusplus`
(and of scikit-learn and cuVS): the first center is drawn at random, then each later
center draws ``n_local_trials`` candidates with probability proportional to the squared
distance to the closest center so far and keeps the one that lowers the total most.
With an optional ``sample_weight``, the first center and the draws are proportional to
the weights and the costs are weighted sums.

- :func:`kmeans_plusplus_cupy` uses only CuPy array operations: a GEMM for the
  candidate distances, then elementwise ops, reductions, ``cumsum`` and
  ``searchsorted``. Each of them has an array API counterpart. The chosen index stays
  on the device, so the loop never waits for the GPU.
- :func:`kmeans_plusplus_cupy_fused` runs three JIT-compiled CUDA kernels per center
  (``cupy_plusplus.cu``). ``kpp_eval`` reads ``X`` once per center: it updates the
  closest distances with the center chosen at the previous step and sums, per chunk of
  rows, the costs of the new candidates. ``kpp_select_a`` picks the best candidate and
  locates each new draw in the chunk sums, and ``kpp_select_b`` picks the row inside the
  chunk. Distances are sums of squared differences, so a row equal to a center gets a
  distance of exactly 0.

For a given ``rng`` the two functions use the same random draws and select the same
rows, up to float32 rounding.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_KERNEL_SOURCE = Path(__file__).with_name("cupy_plusplus.cu")
_EVAL_THREADS = 256
_modules = {}


def default_n_local_trials(n_clusters):
    """scikit-learn's rule; cuVS uses ``2 + ceil(log(n_clusters))``."""
    return 2 + int(math.log(n_clusters))


def _first_center(rng, n_samples, sample_weight):
    if sample_weight is None:
        return int(rng.integers(n_samples))
    import cupy as cp

    cumulative = cp.cumsum(sample_weight, dtype=cp.float64)
    target = rng.random() * float(cumulative[-1])
    return min(int(cp.searchsorted(cumulative, target, side="right")), n_samples - 1)


def kmeans_plusplus_cupy(X, n_clusters, rng, n_local_trials=None, sample_weight=None):
    """k-means++ with CuPy array operations only. Returns ``(centers, indices)``."""
    import cupy as cp

    n_samples, n_features = (int(s) for s in X.shape)
    n_clusters = int(n_clusters)
    n_trials = n_local_trials or default_n_local_trials(n_clusters)
    first = _first_center(rng, n_samples, sample_weight)
    # One row of uniforms per center, as in kmeans_plusplus_cupy_fused.
    uniforms = cp.asarray(rng.random((n_clusters, n_trials)))
    x_norm = cp.sum(X * X, axis=1)
    closest = cp.maximum(x_norm - 2 * (X @ X[first]) + x_norm[first], 0)
    centers = cp.empty((n_clusters, n_features), dtype=X.dtype)
    indices = cp.empty(n_clusters, dtype=cp.int64)
    centers[0] = X[first]
    indices[0] = first
    for step in range(1, n_clusters):
        weights = closest if sample_weight is None else closest * sample_weight
        cumulative = cp.cumsum(weights, dtype=cp.float64)
        picks = cp.searchsorted(cumulative, uniforms[step - 1] * cumulative[-1], side="right")
        picks = cp.minimum(picks, n_samples - 1)
        candidates = cp.take(X, picks, axis=0)
        distances = candidates @ X.T
        distances *= -2
        distances += x_norm
        distances += cp.sum(candidates * candidates, axis=1)[:, None]
        updated = cp.minimum(cp.maximum(distances, 0), closest)
        costs = cp.sum(updated if sample_weight is None else updated * sample_weight, axis=1)
        best = cp.argmin(costs)
        closest = cp.take(updated, best, axis=0)
        centers[step] = cp.take(candidates, best, axis=0)
        indices[step] = cp.take(picks, best)
    return centers, indices


def _launch_config(n_features):
    """Lanes per row (G) and rows per lane group (R) for kpp_eval."""
    vec = n_features % 4 == 0
    width = n_features // 4 if vec else n_features
    if width <= 2 or (not vec and n_features <= 8):
        lanes = 1
    else:
        lanes = min(32, 1 << (width - 1).bit_length())
    return vec, lanes, 4


def fused_supported(n_features, n_trials):
    """The candidates of a step must fit in kpp_eval's static shared memory."""
    return (n_trials + 1) * n_features * 4 <= 32 * 1024


def _fused_module(n_features, n_trials, weighted):
    import cupy as cp

    key = (n_features, n_trials, weighted)
    if key not in _modules:
        vec, lanes, rows = _launch_config(n_features)
        chunk = max(256, _EVAL_THREADS // lanes * rows)
        defines = dict(D=n_features, T=n_trials, G=lanes, R=rows, VEC=int(vec),
                       WEIGHTED=int(weighted), EVAL_THREADS=_EVAL_THREADS, CH=chunk)
        options = ("-std=c++17", *(f"-DKPP_{name}={value}" for name, value in defines.items()))
        module = cp.RawModule(code=_KERNEL_SOURCE.read_text(), options=options)
        _modules[key] = (module, chunk)
    return _modules[key]


def kmeans_plusplus_cupy_fused(X, n_clusters, rng, n_local_trials=None, sample_weight=None):
    """k-means++ with fused CUDA kernels. Returns ``(centers, indices)``.

    ``X`` must be a float32 CuPy array; ``fused_supported`` bounds its width.
    """
    import cupy as cp

    if X.dtype != cp.float32:
        raise TypeError(f"X must be float32, got {X.dtype}")
    X = cp.ascontiguousarray(X)
    n_samples, n_features = (int(s) for s in X.shape)
    n_clusters = int(n_clusters)
    n_trials = n_local_trials or default_n_local_trials(n_clusters)
    if not fused_supported(n_features, n_trials):
        raise ValueError(f"{n_features} features with {n_trials} trials exceed shared memory")
    weighted = sample_weight is not None
    module, chunk = _fused_module(n_features, n_trials, weighted)
    evaluate, select_a, select_b = (
        module.get_function(name) for name in ("kpp_eval", "kpp_select_a", "kpp_select_b")
    )
    n_chunks = (n_samples + chunk - 1) // chunk
    first = _first_center(rng, n_samples, sample_weight)
    uniforms = cp.asarray(rng.random((n_clusters, n_trials)))
    weights = (
        cp.ascontiguousarray(sample_weight, dtype=cp.float32) if weighted
        else cp.empty(1, cp.float32)
    )
    # Step 0 evaluates n_trials copies of the first center against closest = inf, so
    # every step runs the same three kernels.
    closest = cp.full(n_samples, cp.inf, dtype=cp.float32)
    candidates = cp.ascontiguousarray(cp.tile(X[first], (n_trials, 1)))
    candidate_indices = cp.full(n_trials, first, dtype=cp.int64)
    previous = cp.zeros(n_features, dtype=cp.float32)
    centers = cp.empty((n_clusters, n_features), dtype=cp.float32)
    indices = cp.empty(n_clusters, dtype=cp.int64)
    chunk_sums = cp.empty((n_trials, n_chunks), dtype=cp.float64)
    target_chunk = cp.empty(n_trials, dtype=cp.int32)
    residual = cp.empty(n_trials, dtype=cp.float64)
    total = cp.empty(1, dtype=cp.float64)
    n64, nc = np.int64(n_samples), np.int32(n_chunks)
    for step in range(n_clusters):
        evaluate((n_chunks,), (_EVAL_THREADS,),
                 (X, weights, closest, previous, np.int32(step > 0), candidates, chunk_sums,
                  n64, nc))
        last = step == n_clusters - 1
        select_a((1,), (1024,),
                 (chunk_sums, nc, candidates, candidate_indices, centers, indices, previous,
                  np.int32(step), np.int32(last), uniforms[step], target_chunk, residual, total))
        if not last:
            select_b((n_trials,), (1024,),
                     (X, weights, closest, previous, target_chunk, residual, total,
                      uniforms[step], candidates, candidate_indices, n64))
    return centers, indices
