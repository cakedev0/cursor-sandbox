"""Time greedy k-means++ initializations on a CuPy array against cuML's.

Variants (one initialization each, data already on the GPU):
  array-api   kmeans_lloyd.iteration.kmeans_plusplus, the portable path
  cupy-ops    kmeans_lloyd.cupy_plusplus.kmeans_plusplus_cupy (CuPy array operations)
  cupy-fused  kmeans_lloyd.cupy_plusplus.kmeans_plusplus_cupy_fused (JIT CUDA kernels)
  cuml        cuml.KMeans(init="k-means++", n_init=1, max_iter=0), which returns the
              k-means++ centers (the fit also checks the input and computes the inertia)

All variants use cuVS's trial count, 2 + ceil(log(k)), so they do the same work. Each
initialization is followed by the same cuML Lloyd run (max_iter=--lloyd, tol=1e-9) to
compare center quality. Variants are interleaved, in a rotating order, for each seed.
One JSON line per measurement is appended to --out.

Usage:
  python scripts/bench_plusplus_cupy_cuml.py --data glove200.npy --k 256,1024 --out runs.jsonl
  python scripts/bench_plusplus_cupy_cuml.py --synthetic 150000,128 --out runs.jsonl
"""

import argparse
import importlib.metadata as metadata
import json
import math
import os
import time

import cupy as cp
import numpy as np
from cuml.cluster import KMeans as CumlKMeans

from kmeans_lloyd.cupy_plusplus import kmeans_plusplus_cupy, kmeans_plusplus_cupy_fused
from kmeans_lloyd.iteration import kmeans_plusplus

VARIANTS = ["cuml", "array-api", "cupy-ops", "cupy-fused"]


def sync():
    cp.cuda.Device().synchronize()


def mean_squared_distance_to_closest(X, centers, batch=1 << 16):
    center_norm = cp.sum(centers * centers, axis=1)
    total = 0.0
    for start in range(0, X.shape[0], batch):
        block = X[start:start + batch]
        distances = cp.sum(block * block, axis=1)[:, None] - 2 * block @ centers.T + center_norm
        total += float(cp.maximum(distances.min(axis=1), 0).sum(dtype=cp.float64))
    return total / X.shape[0]


def initialize(variant, X, n_clusters, seed):
    n_trials = 2 + math.ceil(math.log(n_clusters))
    rng = np.random.default_rng(seed)
    sync()
    start = time.perf_counter()
    if variant == "cuml":
        model = CumlKMeans(n_clusters=n_clusters, init="k-means++", n_init=1, max_iter=0,
                           random_state=seed, output_type="cupy").fit(X)
        centers = cp.asarray(model.cluster_centers_)
    else:
        # cuML switches CuPy to rmm_cupy_allocator, which pays a cudaMalloc and a
        # synchronizing cudaFree per temporary array. Use CuPy's pool, as a standalone
        # CuPy program would.
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            function = {"array-api": kmeans_plusplus, "cupy-ops": kmeans_plusplus_cupy,
                        "cupy-fused": kmeans_plusplus_cupy_fused}[variant]
            centers = function(X, n_clusters, rng, n_local_trials=n_trials)[0]
    sync()
    return centers, time.perf_counter() - start


def lloyd(X, centers, n_clusters, max_iter):
    model = CumlKMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=max_iter,
                       tol=1e-9, output_type="cupy").fit(X)
    return float(model.inertia_) / X.shape[0], int(model.n_iter_)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="float32 .npy file")
    parser.add_argument("--synthetic", help="N,D: standard normal float32 data")
    parser.add_argument("--k", default="256,1024")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lloyd", type=int, default=100)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.data:
        X = cp.asarray(np.load(args.data))
        name = os.path.basename(args.data).removesuffix(".npy")
    else:
        n_samples, n_features = map(int, args.synthetic.split(","))
        X = cp.asarray(np.random.default_rng(0).standard_normal((n_samples, n_features),
                                                                dtype=np.float32))
        name = f"normal-{n_samples}x{n_features}"
    assert X.dtype == cp.float32
    variants = args.variants.split(",")
    meta = dict(data=name, n_samples=int(X.shape[0]), n_features=int(X.shape[1]),
                gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                driver=cp.cuda.runtime.driverGetVersion(),
                cupy=cp.__version__, cuml=metadata.version("cuml-cu13"),
                libcuvs=metadata.version("libcuvs-cu13"))

    def emit(record):
        record = {**meta, **record, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(json.dumps(record), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")

    for n_clusters in map(int, args.k.split(",")):
        # The first call of each variant loads modules and, for cupy-fused, compiles
        # the kernels (a cold compile when CUPY_CACHE_DIR is empty).
        for variant in variants:
            _, seconds = initialize(variant, X, n_clusters, seed=0)
            emit(dict(k=n_clusters, variant=variant, kind="first_call", seconds=seconds))
        CumlKMeans(n_clusters=n_clusters, init="random", n_init=1, max_iter=2).fit(X)
        for seed in range(args.seeds):
            shift = seed % len(variants)
            for variant in variants[shift:] + variants[:shift]:
                centers, seconds = initialize(variant, X, n_clusters, seed)
                record = dict(k=n_clusters, seed=seed, variant=variant, kind="init",
                              init_s=seconds,
                              init_inertia=mean_squared_distance_to_closest(X, centers))
                if args.lloyd:
                    record["final_inertia"], record["lloyd_n_iter"] = lloyd(
                        X, centers, n_clusters, args.lloyd)
                emit(record)


if __name__ == "__main__":
    main()
