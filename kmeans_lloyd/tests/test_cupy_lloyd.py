import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)

from kmeans_lloyd.cupy_lloyd import kmeans_cupy, lloyd_cupy, lloyd_cupy_fused

LLOYDS = [lloyd_cupy, lloyd_cupy_fused]


def squared_distances(X, C):
    X, C = X.astype(np.float64), C.astype(np.float64)
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


@pytest.mark.parametrize("n_features", [3, 7, 200])
def test_fused_assignment_is_the_nearest_center(n_features):
    # 300 centers: two full 128-center tiles and a partial one
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, n_features), dtype=np.float32)
    C = rng.standard_normal((300, n_features), dtype=np.float32)
    _, labels, inertia, _ = lloyd_cupy_fused(cp.asarray(X), cp.asarray(C), max_iter=0)
    distances = squared_distances(X, C)
    chosen = distances[np.arange(len(X)), cp.asnumpy(labels)]
    nearest = distances.min(axis=1)
    np.testing.assert_allclose(chosen, nearest, rtol=1e-5, atol=1e-5)
    assert inertia == pytest.approx(nearest.sum(), rel=1e-5)


def test_ops_and_fused_lloyd_reach_the_same_inertia():
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((20_000, 24), dtype=np.float32))
    C = X[:50].copy()
    initial = float(cp.sum(cp.min(cp.asarray(squared_distances(cp.asnumpy(X), cp.asnumpy(C))),
                                  axis=1)))
    results = [lloyd(X, C, max_iter=30, tol=0) for lloyd in LLOYDS]
    for _, _, inertia, _ in results:
        assert inertia < initial
    assert results[0][2] == pytest.approx(results[1][2], rel=1e-4)


@pytest.mark.parametrize("lloyd", LLOYDS)
def test_empty_cluster_keeps_its_center(lloyd):
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((5000, 8), dtype=np.float32))
    C = cp.concatenate([X[:9], cp.full((1, 8), 100, dtype=cp.float32)])
    centers, labels, _, _ = lloyd(X, C, max_iter=5, tol=0)
    assert not bool((labels == 9).any())
    np.testing.assert_array_equal(cp.asnumpy(centers[9]), np.full(8, 100, dtype=np.float32))


@pytest.mark.parametrize("fused", [False, True])
def test_kmeans_returns_the_best_initialization(fused):
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((10_000, 16), dtype=np.float32))
    (centers, labels, inertia, _), runs = kmeans_cupy(
        X, 20, fused=fused, n_init=3, max_iter=20, random_state=0)
    assert len(runs) == 3
    assert inertia == min(run_inertia for run_inertia, _ in runs)
    distances = squared_distances(cp.asnumpy(X), cp.asnumpy(centers))
    chosen = distances[np.arange(distances.shape[0]), cp.asnumpy(labels)]
    np.testing.assert_allclose(chosen, distances.min(axis=1), rtol=1e-5, atol=1e-5)
    assert inertia == pytest.approx(chosen.sum(), rel=1e-5)
