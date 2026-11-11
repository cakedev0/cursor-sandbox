// Lloyd iteration kernels (see cupy_lloyd.py), compiled by NVRTC through CuPy.
// -DLL_VEC4=1 when the number of features is a multiple of 4 (16-byte loads).
#define VEC4 LL_VEC4
#define BM 128  // rows per block
#define BN 128  // centers per tile
#define BK 8    // features per shared-memory step
#define TM 8    // rows per thread
#define TN 8    // centers per thread
#define THREADS 256

// One block owns BM rows and loops over all centers, BN at a time: a tiled FP32 GEMM of
// the rows against the center tile, then a running argmin of |c|^2 - 2 x.c. The n x k
// score matrix is never written. With accumulate != 0 the block then adds its rows to
// the sums of their new centers (atomics) and counts them.
extern "C" __global__ void __launch_bounds__(THREADS, 2)
lloyd_assign(const float* __restrict__ X, const float* __restrict__ C,
             const float* __restrict__ center_norm, const float* __restrict__ row_norm,
             const int n, const int d, const int k, int* __restrict__ labels,
             float* __restrict__ sums, float* __restrict__ counts,
             double* __restrict__ inertia, const int accumulate) {
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];
    __shared__ int s_label[BM];
    __shared__ double s_red[THREADS / 32];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const long long row0 = (long long)blockIdx.x * BM;
    float best[TM];
    int best_idx[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) { best[i] = __int_as_float(0x7f800000); best_idx[i] = 0; }
#if VEC4
    // one float4 of X and one of C per thread and step: row tid / 2, features 4 * (tid % 2)
    const int lr = tid / 2, lc = 4 * (tid % 2);
    float4 ra, rb;
#else
    float ra[4], rb[4];
#endif

    for (int c0 = 0; c0 < k; c0 += BN) {
        float acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
        // Global loads of the next feature step are issued before the current step is
        // computed, into registers, then stored into the other shared buffer.
        auto load = [&](int k0) {
#if VEC4
            const long long gr = row0 + lr;
            const int cr = c0 + lr, gc = k0 + lc;
            const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
            ra = (gr < n && gc < d) ? *reinterpret_cast<const float4*>(X + gr * d + gc) : zero;
            rb = (cr < k && gc < d) ? *reinterpret_cast<const float4*>(C + (long long)cr * d + gc) : zero;
#else
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int e = tid + q * THREADS;
                const int r = e % BM, c = e / BM;
                const long long gr = row0 + r;
                const int gc = k0 + c, cr = c0 + r;
                ra[q] = (gr < n && gc < d) ? X[gr * d + gc] : 0.f;
                rb[q] = (cr < k && gc < d) ? C[(long long)cr * d + gc] : 0.f;
            }
#endif
        };
        auto store = [&](int buf) {
#if VEC4
            As[buf][lc][lr] = ra.x; As[buf][lc + 1][lr] = ra.y;
            As[buf][lc + 2][lr] = ra.z; As[buf][lc + 3][lr] = ra.w;
            Bs[buf][lc][lr] = rb.x; Bs[buf][lc + 1][lr] = rb.y;
            Bs[buf][lc + 2][lr] = rb.z; Bs[buf][lc + 3][lr] = rb.w;
#else
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int e = tid + q * THREADS;
                As[buf][e / BM][e % BM] = ra[q];
                Bs[buf][e / BM][e % BM] = rb[q];
            }
#endif
        };
        load(0);
        store(0);
        __syncthreads();
        int buf = 0;
        for (int k0 = 0; k0 < d; k0 += BK) {
            const bool more = k0 + BK < d;
            if (more) load(k0 + BK);
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                const float4 a0 = *reinterpret_cast<const float4*>(&As[buf][kk][ty * TM]);
                const float4 a1 = *reinterpret_cast<const float4*>(&As[buf][kk][ty * TM + 4]);
                const float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * TN]);
                const float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * TN + 4]);
                const float a[TM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float b[TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
            if (more) store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
        // The 16 threads of a row group (same ty) share a warp half: reduce with shuffles.
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            float v = __int_as_float(0x7f800000);
            int idx = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = c0 + tx * TN + j;
                if (col < k) {
                    const float s = center_norm[col] - 2.f * acc[i][j];
                    if (s < v) { v = s; idx = col; }
                }
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ov < v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (v < best[i]) { best[i] = v; best_idx[i] = idx; }  // later tiles: larger indices
        }
    }

    double local = 0.0;
    if (tx == 0) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int r = ty * TM + i;
            s_label[r] = best_idx[i];
            if (row0 + r < n) local += fmaxf(row_norm[row0 + r] + best[i], 0.f);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_xor_sync(0xffffffffu, local, o);
    if (tid % 32 == 0) s_red[tid / 32] = local;
    __syncthreads();
    if (tid == 0) {
        double s = 0.0;
        for (int w = 0; w < THREADS / 32; ++w) s += s_red[w];
        atomicAdd(inertia, s);
    }
    if (tid < BM && row0 + tid < n) labels[row0 + tid] = s_label[tid];
    if (accumulate) {
        const int rows = (int)min((long long)BM, n - row0);
        for (int e = tid; e < rows * d; e += THREADS) {
            const int r = e / d, f = e - r * d;
            atomicAdd(&sums[(long long)s_label[r] * d + f], X[(row0 + r) * d + f]);
        }
        if (tid < rows) atomicAdd(&counts[s_label[tid]], 1.f);
    }
}

// One block per center: new center = sum / count (empty clusters keep their center),
// its squared norm, and the squared shift summed into *shift.
extern "C" __global__ void lloyd_finalize(const float* __restrict__ sums,
                                          const float* __restrict__ counts,
                                          float* __restrict__ C, float* __restrict__ center_norm,
                                          double* __restrict__ shift, const int d) {
    __shared__ double s_shift[32], s_norm[32];
    const int j = blockIdx.x;
    const float count = counts[j];
    double sh = 0.0, nr = 0.0;
    for (int f = threadIdx.x; f < d; f += blockDim.x) {
        const float old = C[(long long)j * d + f];
        const float v = count > 0.f ? sums[(long long)j * d + f] / count : old;
        C[(long long)j * d + f] = v;
        sh += (double)(v - old) * (v - old);
        nr += (double)v * v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        sh += __shfl_xor_sync(0xffffffffu, sh, o);
        nr += __shfl_xor_sync(0xffffffffu, nr, o);
    }
    if (threadIdx.x % 32 == 0) { s_shift[threadIdx.x / 32] = sh; s_norm[threadIdx.x / 32] = nr; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double a = 0.0, b = 0.0;
        for (int w = 0; w < (int)(blockDim.x + 31) / 32; ++w) { a += s_shift[w]; b += s_norm[w]; }
        center_norm[j] = (float)b;
        atomicAdd(shift, a);
    }
}
