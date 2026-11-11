"""Lloyd iterations and full k-means fits for CuPy arrays.

- :func:`lloyd_cupy` uses CuPy array operations only: an augmented GEMM
  ``[X, 1] @ [-2 C, |c|^2].T`` tiled over rows, ``argmin``, then ``add.at`` and
  ``bincount`` for the update.
- :func:`lloyd_cupy_fused` runs one JIT-compiled kernel per iteration
  (``cupy_lloyd.cu``): a tiled FP32 GEMM with a running argmin over the centers, so the
  ``n x k`` score matrix is never written, followed by atomic adds of each row to the
  sum of its new center. A second small kernel divides the sums and measures the shift.

Both follow scikit-learn's stopping rule: stop once the squared center shift is at most
``tol`` times the mean feature variance. Empty clusters keep their center, as in cuML.
The atomic additions make the fused sums, and the ``add.at`` sums, non-deterministic in
their last bits.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from kmeans_lloyd.cupy_plusplus import kmeans_plusplus_cupy, kmeans_plusplus_cupy_fused

_KERNEL_SOURCE = Path(__file__).with_name("cupy_lloyd.cu")
_BLOCK_ROWS = 128
_modules = {}


def _tolerance(X, tol):
    import cupy as cp

    return tol * float(cp.mean(cp.var(X, axis=0)))


def lloyd_cupy(X, centers, max_iter=300, tol=1e-4, augmented=None):
    """Lloyd with CuPy array operations. Returns ``(centers, labels, inertia, n_iter)``.

    ``augmented`` is ``[X, 1]``; pass it to reuse it across calls.
    """
    import cupy as cp

    n_samples, n_features = (int(s) for s in X.shape)
    n_clusters = int(centers.shape[0])
    if augmented is None:
        augmented = cp.concatenate([X, cp.ones((n_samples, 1), dtype=X.dtype)], axis=1)
    tile = max(1024, min(n_samples, (1 << 26) // n_clusters))
    tol = _tolerance(X, tol)
    centers = centers.copy()
    labels = cp.empty(n_samples, dtype=cp.int64)

    def scores_matrix(C):
        return cp.concatenate([-2 * C, cp.sum(C * C, axis=1)[:, None]], axis=1).T

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        scores = scores_matrix(centers)
        for start in range(0, n_samples, tile):
            labels[start:start + tile] = cp.argmin(augmented[start:start + tile] @ scores, axis=1)
        sums = cp.zeros((n_clusters, n_features), dtype=X.dtype)
        cp.add.at(sums, labels, X)
        counts = cp.bincount(labels, minlength=n_clusters).astype(X.dtype)[:, None]
        new_centers = cp.where(counts > 0, sums / cp.maximum(counts, 1), centers)
        shift = float(cp.sum((new_centers - centers) ** 2))
        centers = new_centers
        if shift <= tol:
            break
    scores = scores_matrix(centers)
    row_norm = cp.sum(X * X, axis=1)
    inertia = 0.0
    for start in range(0, n_samples, tile):
        block = augmented[start:start + tile] @ scores
        labels[start:start + tile] = cp.argmin(block, axis=1)
        closest = cp.min(block, axis=1) + row_norm[start:start + tile]
        inertia += float(cp.sum(cp.maximum(closest, 0), dtype=cp.float64))
    return centers, labels, inertia, n_iter


def _kernels(n_features):
    import cupy as cp

    vec4 = int(n_features % 4 == 0)
    if vec4 not in _modules:
        _modules[vec4] = cp.RawModule(code=_KERNEL_SOURCE.read_text(),
                                      options=("-std=c++17", f"-DLL_VEC4={vec4}"))
    module = _modules[vec4]
    return module.get_function("lloyd_assign"), module.get_function("lloyd_finalize")


def lloyd_cupy_fused(X, centers, max_iter=300, tol=1e-4):
    """Lloyd with fused CUDA kernels. Returns ``(centers, labels, inertia, n_iter)``.

    ``X`` and ``centers`` must be float32 CuPy arrays.
    """
    import cupy as cp

    if X.dtype != cp.float32:
        raise TypeError(f"X must be float32, got {X.dtype}")
    X = cp.ascontiguousarray(X)
    n_samples, n_features = (int(s) for s in X.shape)
    assign, finalize = _kernels(n_features)
    n_clusters = int(centers.shape[0])
    tol = _tolerance(X, tol)
    centers = cp.array(centers, dtype=cp.float32, order="C", copy=True)
    row_norm = cp.sum(X * X, axis=1)
    center_norm = cp.sum(centers * centers, axis=1)
    labels = cp.empty(n_samples, dtype=cp.int32)
    sums = cp.empty((n_clusters, n_features), dtype=cp.float32)
    counts = cp.empty(n_clusters, dtype=cp.float32)
    inertia = cp.zeros(1, dtype=cp.float64)
    shift = cp.zeros(1, dtype=cp.float64)
    grid = ((n_samples + _BLOCK_ROWS - 1) // _BLOCK_ROWS,)
    args = (np.int32(n_samples), np.int32(n_features), np.int32(n_clusters))
    finalize_threads = min(256, max(32, 1 << (n_features - 1).bit_length()))
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        sums.fill(0)
        counts.fill(0)
        inertia.fill(0)
        shift.fill(0)
        assign(grid, (256,), (X, centers, center_norm, row_norm, *args, labels, sums, counts,
                              inertia, np.int32(1)))
        finalize((n_clusters,), (finalize_threads,),
                 (sums, counts, centers, center_norm, shift, np.int32(n_features)))
        if float(shift[0]) <= tol:
            break
    inertia.fill(0)
    assign(grid, (256,), (X, centers, center_norm, row_norm, *args, labels, sums, counts,
                          inertia, np.int32(0)))
    return centers, labels.astype(cp.int64), float(inertia[0]), n_iter


def kmeans_cupy(X, n_clusters, *, fused, n_init=10, max_iter=300, tol=1e-4, random_state=None,
                n_local_trials=None):
    """k-means++ then Lloyd, ``n_init`` times; keeps the lowest inertia.

    Returns ``(centers, labels, inertia, n_iter)`` of the best run, and the
    ``(inertia, n_iter)`` of every run.
    """
    import cupy as cp

    rng = np.random.default_rng(random_state)
    if n_local_trials is None:
        n_local_trials = 2 + int(math.log(n_clusters))
    init = kmeans_plusplus_cupy_fused if fused else kmeans_plusplus_cupy
    augmented = None
    if not fused:
        augmented = cp.concatenate([X, cp.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    best, runs = None, []
    for _ in range(n_init):
        centers, _ = init(X, n_clusters, rng, n_local_trials=n_local_trials)
        if fused:
            result = lloyd_cupy_fused(X, centers, max_iter=max_iter, tol=tol)
        else:
            result = lloyd_cupy(X, centers, max_iter=max_iter, tol=tol, augmented=augmented)
        runs.append((result[2], result[3]))
        if best is None or result[2] < best[2]:
            best = result
    return best, runs
