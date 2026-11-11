"""A scikit-learn compatible KMeans estimator using the Python array API."""

from __future__ import annotations

import numbers
import warnings
from typing import ClassVar

import numpy as np
from array_api_compat import array_namespace, device, is_array_api_obj, is_numpy_array
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
)
from sklearn.utils import check_random_state
from sklearn.utils._array_api import check_same_namespace
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from kmeans_lloyd.backends import native_accumulate, native_kmeans_plusplus
from kmeans_lloyd.iteration import assign_gemm_tiled, fit_lloyd, inertia, kmeans_plusplus


class _RandomStateAdapter:
    """Expose the Generator methods used by the initializer on RandomState."""

    def __init__(self, random_state):
        self.random_state = random_state

    def integers(self, high, size=None):
        return self.random_state.randint(high, size=size)

    def random(self, size=None):
        return self.random_state.random_sample(size)

    def choice(self, *args, **kwargs):
        return self.random_state.choice(*args, **kwargs)


class KMeans(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    ClusterMixin,
    BaseEstimator,
):
    """K-means clustering implemented with portable array API operations.

    The public hyperparameters intentionally follow :class:`sklearn.cluster.KMeans`.
    Only dense inputs and Lloyd iterations are supported.

    The centroid update uses a native scatter-add for PyTorch and CuPy
    arrays (see :mod:`kmeans_lloyd.backends`) and portable one-hot matrix
    products for every other array API namespace.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {"k-means++", "random"}, callable or array-like, default="k-means++"
        Initialization strategy. A callable is called as
        ``init(X, n_clusters, random_state=random_state)``. An array of shape
        ``(n_clusters, n_features)`` is converted to the namespace and
        device of ``X``.
    n_init : "auto" or int, default="auto"
        Number of independent initializations. ``"auto"`` uses one run for
        ``"k-means++"`` and explicit centers and ten runs otherwise.
    max_iter : int, default=300
        Maximum Lloyd iterations per initialization.
    tol : float, default=1e-4
        Relative convergence tolerance, scaled by the mean feature variance.
    verbose : int, default=0
        Print one summary line per initialization when non-zero.
    random_state : int, numpy.random.RandomState, numpy.random.Generator, or None
        Random state controlling initialization.
    copy_x : bool, default=True
        Accepted for API compatibility. The implementation never modifies
        ``X``, so no defensive copy is required.
    algorithm : {"lloyd"}, default="lloyd"
        K-means algorithm. Elkan is intentionally unsupported.
    """

    _parameter_constraints: ClassVar[dict] = {
        "n_clusters": [Interval(numbers.Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "tol": [Interval(numbers.Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "copy_x": ["boolean"],
        "algorithm": [StrOptions({"lloyd"})],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def fit(self, X, y=None):
        """Compute k-means clustering."""
        del y
        self._validate_params()
        X = self._validate_input(X, reset=True)
        n_samples = int(X.shape[0])
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )

        rng = self._make_rng()
        n_init = self._resolve_n_init()
        tiles = _tile_schedule(X, int(self.n_clusters))
        # PyTorch index_add_ and CuPy add.at replace the one-hot update;
        # any other namespace, or a shape where the one-hot GEMM is faster,
        # gets None and the portable fallback.
        accumulate = native_accumulate(X, int(self.n_clusters))
        best = None
        for run in range(n_init):
            centers = self._initialize(X, rng)

            # TODO: Accept an optional assignment ``step`` here so a backend
            # adapter can supply a torch.compile'd score tile (2.1x to 4.8x
            # faster assignment in journal 013, but with compile latency) or a
            # CuPy RawKernel that fuses distances with the top-1 reduction
            # like cuVS. Cache either by shape and dtype; keep the portable
            # gemm_scores_argmin as the default for other namespaces.
            result = fit_lloyd(
                X,
                centers,
                max_iter=self.max_iter,
                tol=self.tol,
                tile_n=tiles["tile_n"],
                tile_k=tiles["tile_k"],
                update_tile_k=tiles["update_tile_k"],
                update_tile_n=tiles["update_tile_n"],
                reassign_empty=True,
                rng=rng,
                accumulate=accumulate,
            )
            labels = assign_gemm_tiled(
                X, result["centers"], tiles["tile_n"], tiles["tile_k"]
            )
            objective = float(inertia(X, labels, result["centers"], tiles["tile_n"]))
            if self.verbose:
                print(
                    f"Initialization {run + 1}/{n_init}: inertia={objective:.6g}, "
                    f"iterations={result['n_iter']}"
                )
            if best is None or objective < best["inertia"]:
                best = {**result, "labels": labels, "inertia": objective}

        self.cluster_centers_ = best["centers"]
        self.labels_ = best["labels"]
        self.inertia_ = best["inertia"]
        self.n_iter_ = best["n_iter"]
        self._n_features_out = int(self.n_clusters)
        return self

    def predict(self, X):
        """Assign each sample to its nearest fitted center."""
        check_is_fitted(self)
        check_same_namespace(
            X, self, attribute="cluster_centers_", method="predict"
        )
        X = self._validate_input(X, reset=False)
        tiles = _tile_schedule(X, int(self.n_clusters))
        return assign_gemm_tiled(
            X, self.cluster_centers_, tiles["tile_n"], tiles["tile_k"]
        )

    def transform(self, X):
        """Return Euclidean distances to every fitted center."""
        check_is_fitted(self)
        check_same_namespace(
            X, self, attribute="cluster_centers_", method="transform"
        )
        X = self._validate_input(X, reset=False)
        xp = array_namespace(X, self.cluster_centers_)
        n_samples = int(X.shape[0])
        distances = xp.empty(
            (n_samples, int(self.n_clusters)), dtype=X.dtype, device=device(X)
        )
        tile_n = _tile_schedule(X, int(self.n_clusters))["tile_n"]
        center_norm = xp.sum(self.cluster_centers_ * self.cluster_centers_, axis=1)
        for n0 in range(0, n_samples, tile_n):
            n1 = min(n0 + tile_n, n_samples)
            block = X[n0:n1, :]
            scores = (
                xp.sum(block * block, axis=1)[:, None]
                + center_norm[None, :]
                - 2 * xp.matmul(block, xp.matrix_transpose(self.cluster_centers_))
            )
            distances[n0:n1, :] = xp.sqrt(
                xp.maximum(scores, xp.zeros_like(scores))
            )
        return distances

    def score(self, X, y=None):
        """Return the negative k-means objective."""
        del y
        check_is_fitted(self)
        check_same_namespace(X, self, attribute="cluster_centers_", method="score")
        X = self._validate_input(X, reset=False)
        labels = self.predict(X)
        return -float(inertia(X, labels, self.cluster_centers_))

    def _resolve_n_init(self):
        explicit = not isinstance(self.init, str) and not callable(self.init)
        if self.n_init == "auto":
            if isinstance(self.init, str) and self.init == "k-means++":
                return 1
            return 1 if explicit else 10
        if explicit and int(self.n_init) != 1:
            warnings.warn(
                "Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self.n_init}.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1
        return int(self.n_init)

    def _initialize(self, X, rng):
        xp = array_namespace(X)
        if isinstance(self.init, str) and self.init == "k-means++":
            plusplus = native_kmeans_plusplus(X, int(self.n_clusters)) or kmeans_plusplus
            return plusplus(X, int(self.n_clusters), rng)[0]
        if isinstance(self.init, str):
            indices = xp.asarray(
                rng.choice(int(X.shape[0]), size=int(self.n_clusters), replace=False),
                dtype=xp.int64,
                device=device(X),
            )
            return xp.take(X, indices, axis=0)
        if callable(self.init):
            random_state = (
                self.random_state
                if isinstance(self.random_state, np.random.Generator)
                else check_random_state(self.random_state)
            )
            centers = self.init(X, int(self.n_clusters), random_state=random_state)
        else:
            centers = self.init
        centers = xp.asarray(centers, dtype=X.dtype, device=device(X))
        expected = (int(self.n_clusters), int(X.shape[1]))
        if tuple(int(size) for size in centers.shape) != expected:
            raise ValueError(
                f"The shape of the initial centers {tuple(centers.shape)} does not "
                f"match the number of clusters and features {expected}."
            )
        return centers

    def _make_rng(self):
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        if isinstance(self.random_state, np.random.RandomState):
            return _RandomStateAdapter(self.random_state)
        return np.random.default_rng(self.random_state)

    def _validate_input(self, X, *, reset):
        if not is_array_api_obj(X) or is_numpy_array(X):
            return validate_data(
                self,
                X,
                reset=reset,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                ensure_min_samples=1,
                ensure_min_features=1,
            )

        xp = array_namespace(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if not xp.isdtype(X.dtype, "real floating"):
            raise TypeError(
                f"KMeans requires float32 or float64 input, got dtype {X.dtype}."
            )
        if int(X.shape[0]) < 1 or int(X.shape[1]) < 1:
            raise ValueError("KMeans requires at least one sample and one feature.")
        if not _all_finite(xp, X):
            raise ValueError("Input X contains NaN or infinity.")
        if reset:
            self.n_features_in_ = int(X.shape[1])
        elif int(X.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KMeans is expecting "
                f"{self.n_features_in_} features as input."
            )
        return X

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        tags.input_tags.sparse = False
        tags.transformer_tags.preserves_dtype = ["float32", "float64"]
        return tags


def _all_finite(xp, X):
    """Check finiteness without an input-sized temporary.

    Eager ``xp.isfinite(X)`` costs a boolean array as large as ``X``, and
    PyTorch adds float temporaries on top: 5.3 GiB extra for a 3.8 GB input.
    Column sums are finite for finite data unless they overflow, so only a
    non-finite sum triggers the exact check, one row tile at a time.
    """
    if bool(xp.all(xp.isfinite(xp.sum(X, axis=0)))):
        return True
    n_samples, n_features = map(int, X.shape)
    tile_n = max(1, (1 << 24) // n_features)
    for n0 in range(0, n_samples, tile_n):
        if not bool(xp.all(xp.isfinite(X[n0 : min(n0 + tile_n, n_samples), :]))):
            return False
    return True


def _tile_schedule(X, n_clusters):
    """Choose bounded tiles from shape and device, without backend checks."""
    n_samples, n_features = map(int, X.shape)
    is_cpu = "cpu" in str(device(X)).lower()
    if is_cpu:
        return {
            "tile_n": min(n_samples, 4096),
            "tile_k": min(n_clusters, 256),
            "update_tile_n": min(n_samples, 4096),
            "update_tile_k": min(n_clusters, 256),
        }
    return {
        "tile_n": min(n_samples, 65_536),
        "tile_k": min(n_clusters, 128 if n_features <= 8 and n_clusters >= 512 else 256),
        "update_tile_n": min(n_samples, 16_384),
        "update_tile_k": min(n_clusters, 512),
    }
