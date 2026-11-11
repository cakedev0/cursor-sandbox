import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)

from kmeans_lloyd.backends import native_kmeans_plusplus
from kmeans_lloyd.cupy_plusplus import (
    default_n_local_trials,
    kmeans_plusplus_cupy,
    kmeans_plusplus_cupy_fused,
)

IMPLEMENTATIONS = [kmeans_plusplus_cupy, kmeans_plusplus_cupy_fused]


def reference_plusplus(X, n_clusters, rng, sample_weight=None):
    """Greedy k-means++ in float64 NumPy, with the same random draws."""
    X = X.astype(np.float64)
    n_samples = X.shape[0]
    n_trials = default_n_local_trials(n_clusters)
    w = np.ones(n_samples) if sample_weight is None else sample_weight.astype(np.float64)
    if sample_weight is None:
        first = int(rng.integers(n_samples))
    else:
        cumulative = np.cumsum(w)
        first = min(int(np.searchsorted(cumulative, rng.random() * cumulative[-1], "right")),
                    n_samples - 1)
    uniforms = rng.random((n_clusters, n_trials))
    closest = ((X - X[first]) ** 2).sum(axis=1)
    indices = [first]
    for step in range(1, n_clusters):
        cumulative = np.cumsum(w * closest)
        picks = np.minimum(
            np.searchsorted(cumulative, uniforms[step - 1] * cumulative[-1], "right"),
            n_samples - 1,
        )
        distances = ((X[None, :, :] - X[picks][:, None, :]) ** 2).sum(axis=2)
        updated = np.minimum(closest, distances)
        best = int(np.argmin((updated * w).sum(axis=1)))
        closest = updated[best]
        indices.append(int(picks[best]))
    return np.asarray(indices)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_same_rows_as_float64_reference(implementation):
    rng = np.random.default_rng(0)
    n_runs = n_identical = 0
    for n_samples, n_features, n_clusters in [(4000, 3, 32), (3000, 7, 40), (3000, 128, 48),
                                              (2000, 200, 24)]:
        X = rng.standard_normal((n_samples, n_features), dtype=np.float32)
        for weighted in (False, True):
            w = rng.random(n_samples).astype(np.float32) * 2 if weighted else None
            for seed in range(2):
                expected = reference_plusplus(X, n_clusters, np.random.default_rng(seed), w)
                centers, indices = implementation(
                    cp.asarray(X), n_clusters, np.random.default_rng(seed),
                    sample_weight=None if w is None else cp.asarray(w),
                )
                indices = cp.asnumpy(indices)
                np.testing.assert_array_equal(cp.asnumpy(centers), X[indices])
                n_runs += 1
                n_identical += np.array_equal(indices, expected)
    # float32 vs float64: a draw that lands within rounding of a row boundary picks the
    # neighboring row, after which the runs diverge. It is rare.
    assert n_identical >= 0.85 * n_runs, f"{n_identical}/{n_runs}"


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_zero_weight_rows_are_never_chosen(implementation):
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((20_000, 16), dtype=np.float32))
    w = cp.asarray((rng.random(20_000) < 0.5).astype(np.float32))
    for seed in range(3):
        _, indices = implementation(X, 32, np.random.default_rng(seed), sample_weight=w)
        assert bool((w[indices] > 0).all())


def test_fused_never_picks_a_copy_of_a_center():
    rng = np.random.default_rng(0)
    # far from the origin, where expanded-form distances to a copy are not exactly 0
    distinct = rng.standard_normal((10, 16), dtype=np.float32) + 1000
    X = cp.asarray(np.repeat(distinct, 100, axis=0))
    centers, _ = kmeans_plusplus_cupy_fused(X, 10, np.random.default_rng(0))
    assert len(np.unique(cp.asnumpy(centers), axis=0)) == 10


def test_native_hook_selects_fused_for_float32_cupy_only():
    X = cp.ones((100, 8), dtype=cp.float32)
    assert native_kmeans_plusplus(X, 4) is kmeans_plusplus_cupy_fused
    assert native_kmeans_plusplus(X.astype(cp.float64), 4) is None
    assert native_kmeans_plusplus(cp.asnumpy(X), 4) is None
