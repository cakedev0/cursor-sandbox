"""Time Lloyd iterations and full k-means fits on a CuPy array against cuML.

--mode lloyd: from the same k-means++ centers (kmeans_plusplus_cupy_fused, seed s), run
  cuml        cuml.KMeans(init=centers, n_init=1, max_iter, tol)
  cupy-ops    kmeans_lloyd.cupy_lloyd.lloyd_cupy (CuPy array operations)
  cupy-fused  kmeans_lloyd.cupy_lloyd.lloyd_cupy_fused (JIT CUDA kernels)
--mode fit: n_init initializations, each k-means++ then Lloyd
  cuml        cuml.KMeans(init="k-means++", n_init, max_iter, tol)
  cupy-ops    kmeans_cupy(fused=False): kmeans_plusplus_cupy + lloyd_cupy
  cupy-fused  kmeans_cupy(fused=True): kmeans_plusplus_cupy_fused + lloyd_cupy_fused

The CuPy variants use cuVS's k-means++ trial count, 2 + ceil(log(k)). The stopping rules
differ: cuML stops when the inertia stops decreasing in float32 (with tol=1e-9), the CuPy
variants when the center shift is at most tol times the mean feature variance. Iteration
counts are recorded so that time per iteration can be compared. Variants are interleaved,
in a rotating order, for each seed. One JSON line per measurement is appended to --out.
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

from kmeans_lloyd.cupy_lloyd import kmeans_cupy, lloyd_cupy, lloyd_cupy_fused
from kmeans_lloyd.cupy_plusplus import kmeans_plusplus_cupy_fused

VARIANTS = ["cuml", "cupy-ops", "cupy-fused"]


def sync():
    cp.cuda.Device().synchronize()


def cupy_pool():
    # cuML switches CuPy to rmm_cupy_allocator (a cudaMalloc and a synchronizing
    # cudaFree per temporary array); a standalone CuPy program uses CuPy's pool.
    return cp.cuda.using_allocator(cp.get_default_memory_pool().malloc)


def run_lloyd(variant, X, centers, max_iter, tol):
    sync()
    start = time.perf_counter()
    if variant == "cuml":
        model = CumlKMeans(n_clusters=centers.shape[0], init=centers, n_init=1,
                           max_iter=max_iter, tol=tol, output_type="cupy").fit(X)
        inertia, n_iter = float(model.inertia_), int(model.n_iter_)
    else:
        with cupy_pool():
            function = lloyd_cupy if variant == "cupy-ops" else lloyd_cupy_fused
            _, _, inertia, n_iter = function(X, centers, max_iter=max_iter, tol=tol)
    sync()
    seconds = time.perf_counter() - start
    return dict(seconds=seconds, n_iter=n_iter, final_inertia=inertia / X.shape[0],
                seconds_per_iter=seconds / max(n_iter, 1))


def run_fit(variant, X, n_clusters, n_init, max_iter, tol, seed):
    sync()
    start = time.perf_counter()
    if variant == "cuml":
        model = CumlKMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init,
                           max_iter=max_iter, tol=tol, random_state=seed,
                           output_type="cupy").fit(X)
        record = dict(inertia=float(model.inertia_), best_n_iter=int(model.n_iter_))
    else:
        with cupy_pool():
            (_, _, inertia, n_iter), runs = kmeans_cupy(
                X, n_clusters, fused=variant == "cupy-fused", n_init=n_init,
                max_iter=max_iter, tol=tol, random_state=seed,
                n_local_trials=2 + math.ceil(math.log(n_clusters)))
        record = dict(inertia=inertia, best_n_iter=n_iter, run_n_iter=[r[1] for r in runs])
    sync()
    record["inertia"] /= X.shape[0]
    return dict(seconds=time.perf_counter() - start, **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["lloyd", "fit"], required=True)
    parser.add_argument("--data", help="float32 .npy file")
    parser.add_argument("--synthetic", help="N,D: standard normal float32 data")
    parser.add_argument("--k", default="256,1024")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-9)
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
    variants = args.variants.split(",")
    meta = dict(mode=args.mode, data=name, n_samples=int(X.shape[0]),
                n_features=int(X.shape[1]), max_iter=args.max_iter, tol=args.tol,
                gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                driver=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__,
                cuml=metadata.version("cuml-cu13"), libcuvs=metadata.version("libcuvs-cu13"))

    def emit(record):
        record = {**meta, **record, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(json.dumps(record), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")

    for n_clusters in map(int, args.k.split(",")):
        n_trials = 2 + math.ceil(math.log(n_clusters))
        with cupy_pool():  # warm-up: module loading and JIT compilation, untimed
            warm = kmeans_plusplus_cupy_fused(X, n_clusters, np.random.default_rng(0),
                                              n_local_trials=n_trials)[0]
        for variant in variants:
            run_lloyd(variant, X, warm, 2, args.tol)
        for seed in range(args.seeds):
            shift = seed % len(variants)
            order = variants[shift:] + variants[:shift]
            if args.mode == "lloyd":
                with cupy_pool():
                    centers = kmeans_plusplus_cupy_fused(
                        X, n_clusters, np.random.default_rng(seed), n_local_trials=n_trials)[0]
                for variant in order:
                    emit(dict(k=n_clusters, seed=seed, variant=variant,
                              **run_lloyd(variant, X, centers, args.max_iter, args.tol)))
            else:
                for variant in order:
                    emit(dict(k=n_clusters, seed=seed, variant=variant, n_init=args.n_init,
                              **run_fit(variant, X, n_clusters, args.n_init, args.max_iter,
                                        args.tol, seed)))


if __name__ == "__main__":
    main()
