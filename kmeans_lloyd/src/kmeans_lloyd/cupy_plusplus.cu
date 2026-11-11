// Greedy k-means++ step kernels (see cupy_plusplus.py), compiled by NVRTC through CuPy.
// The launch configuration comes in as -DKPP_<NAME> options. They are renamed here, after
// the headers NVRTC includes on its own, whose templates use some of these short names.
#define D KPP_D                        // features
#define T KPP_T                        // candidates per step
#define G KPP_G                        // lanes per row
#define R KPP_R                        // rows per lane group
#define VEC KPP_VEC                    // D % 4 == 0: 16-byte loads
#define WEIGHTED KPP_WEIGHTED
#define EVAL_THREADS KPP_EVAL_THREADS
#define CH KPP_CH                      // rows per chunk
#define SEL_THREADS 1024
#define SELB_THREADS 1024

typedef long long i64;
#define VW 4  // floats per 16-byte vector load

__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, G);
    return v;
}

// Load the slice of row i that lane `lane` handles in iteration f (VW elements).
__device__ __forceinline__ void load_vec(const float* __restrict__ X, i64 i, int f, float* xv) {
    float4 v = __ldcs(reinterpret_cast<const float4*>(X + i * D) + f);
    xv[0] = v.x; xv[1] = v.y; xv[2] = v.z; xv[3] = v.w;
}

// Squared distances from R rows of X (rows[r], valid[r]) to the NC rows of s_c, summed over
// the G lanes of the group: all lanes get acc[r][t]. Each candidate value read from shared
// memory is reused for the R rows.
template <int NC, int RR>
__device__ __forceinline__ void rows_dists(const float* __restrict__ X, const i64* rows, const bool* valid,
                                           const float* s_c, int lane, float (*acc)[NC]) {
#pragma unroll
    for (int r = 0; r < RR; ++r)
#pragma unroll
        for (int t = 0; t < NC; ++t) acc[r][t] = 0.f;
#if VEC
    for (int f = lane; f < D / VW; f += G) {
        float xv[RR][VW];
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            if (valid[r]) load_vec(X, rows[r], f, xv[r]);
            else {
#pragma unroll
                for (int q = 0; q < VW; ++q) xv[r][q] = 0.f;
            }
        }
#pragma unroll
        for (int t = 0; t < NC; ++t) {
            const float4* c4 = reinterpret_cast<const float4*>(s_c + t * D + f * VW);
#pragma unroll
            for (int q = 0; q < VW / 4; ++q) {
                float4 c = c4[q];
#pragma unroll
                for (int r = 0; r < RR; ++r) {
                    float e0 = xv[r][4 * q] - c.x, e1 = xv[r][4 * q + 1] - c.y;
                    float e2 = xv[r][4 * q + 2] - c.z, e3 = xv[r][4 * q + 3] - c.w;
                    acc[r][t] = fmaf(e0, e0, fmaf(e1, e1, fmaf(e2, e2, fmaf(e3, e3, acc[r][t]))));
                }
            }
        }
    }
#else
    for (int f = lane; f < D; f += G) {
        float xv[RR];
#pragma unroll
        for (int r = 0; r < RR; ++r) xv[r] = valid[r] ? __ldcs(X + rows[r] * D + f) : 0.f;
#pragma unroll
        for (int t = 0; t < NC; ++t) {
            const float c = s_c[t * D + f];
#pragma unroll
            for (int r = 0; r < RR; ++r) { float a = xv[r] - c; acc[r][t] = fmaf(a, a, acc[r][t]); }
        }
    }
#endif
#pragma unroll
    for (int r = 0; r < RR; ++r)
#pragma unroll
        for (int t = 0; t < NC; ++t) acc[r][t] = group_sum(acc[r][t]);
}

// Owner lanes: lane (r % G) of a group finalizes row r of the group's R rows.
#define NOWN (R < G ? R : G)

__device__ __forceinline__ double warp_sum_d(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ double warp_incl_scan_d(double v, int ln) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        double u = __shfl_up_sync(0xffffffffu, v, o);
        if (ln >= o) v += u;
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(EVAL_THREADS)
kpp_eval(const float* __restrict__ X, const float* __restrict__ w, float* __restrict__ d,
         const float* __restrict__ prev, const int has_prev, const float* __restrict__ cands,
         double* __restrict__ chunk_sums, const i64 n, const int n_chunks) {
    __shared__ __align__(16) float s_c[(T + 1) * D];  // row 0: prev, rows 1..T: candidates
    constexpr int NG = EVAL_THREADS / G;
    __shared__ float s_part[T][NG * NOWN];
    for (int j = threadIdx.x; j < (T + 1) * D; j += EVAL_THREADS)
        s_c[j] = j < D ? prev[j] : cands[j - D];
    __syncthreads();
    const int lane = threadIdx.x % G, grp = threadIdx.x / G;
    float part[T];
#pragma unroll
    for (int t = 0; t < T; ++t) part[t] = 0.f;
    const i64 start = (i64)blockIdx.x * CH;
    for (int it = 0; it < CH / (NG * R); ++it) {
        i64 rows[R]; bool valid[R]; float dpre[R], wpre[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            rows[r] = start + ((i64)it * NG + grp) * R + r;
            valid[r] = rows[r] < n;
            const bool mine = valid[r] && (r % G) == lane;
            dpre[r] = mine ? d[rows[r]] : 0.f;
            wpre[r] = (mine && WEIGHTED) ? w[rows[r]] : 1.f;
        }
        float acc[R][T + 1];
        rows_dists<T + 1, R>(X, rows, valid, s_c, lane, acc);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (valid[r] && (r % G) == lane) {
                float di = dpre[r];
                if (has_prev) { di = fminf(di, acc[r][0]); d[rows[r]] = di; }
#pragma unroll
                for (int t = 0; t < T; ++t) part[t] = fmaf(wpre[r], fminf(di, acc[r][t + 1]), part[t]);
            }
        }
    }
    if (lane < NOWN) {
#pragma unroll
        for (int t = 0; t < T; ++t) s_part[t][grp * NOWN + lane] = part[t];
    }
    __syncthreads();
    const int wid = threadIdx.x / 32, ln = threadIdx.x % 32;
    for (int t = wid; t < T; t += EVAL_THREADS / 32) {
        double s = 0.0;
        for (int g = ln; g < NG * NOWN; g += 32) s += s_part[t][g];
        s = warp_sum_d(s);
        if (ln == 0) chunk_sums[(i64)t * n_chunks + blockIdx.x] = s;
    }
}

// One block. Cost of each candidate, argmin, store the center; then, for each new trial,
// the chunk that holds its draw u * total in the chosen candidate's chunk sums.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
kpp_select_a(const double* __restrict__ chunk_sums, const int n_chunks,
             const float* __restrict__ cands, const i64* __restrict__ cand_idx,
             float* __restrict__ centers, i64* __restrict__ center_idx, float* __restrict__ prev,
             const int step, const int last, const double* __restrict__ u,
             int* __restrict__ tgt_chunk, double* __restrict__ resid, double* __restrict__ total_out) {
    constexpr int NW = SEL_THREADS / 32;
    __shared__ double s_red[NW][T];
    __shared__ double s_wsum[NW + 1];
    __shared__ int s_best;
    const int tid = threadIdx.x, wid = tid / 32, ln = tid % 32;
    double c[T];
#pragma unroll
    for (int t = 0; t < T; ++t) c[t] = 0.0;
    for (int b = tid; b < n_chunks; b += SEL_THREADS) {
#pragma unroll
        for (int t = 0; t < T; ++t) c[t] += chunk_sums[(i64)t * n_chunks + b];
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        double v = warp_sum_d(c[t]);
        if (ln == 0) s_red[wid][t] = v;
    }
    __syncthreads();
    if (wid == 0) {
        // lane t sums candidate t over the warps
        double v = 0.0;
        if (ln < T) for (int q = 0; q < NW; ++q) v += s_red[q][ln];
        int best = 0; double bc = __shfl_sync(0xffffffffu, v, 0);
        for (int t = 1; t < T; ++t) {
            double vt = __shfl_sync(0xffffffffu, v, t);
            if (vt < bc) { bc = vt; best = t; }
        }
        if (ln == 0) { s_best = best; center_idx[step] = cand_idx[best]; }
    }
    __syncthreads();
    const int best = s_best;
    for (int f = tid; f < D; f += SEL_THREADS) {
        float v = cands[best * D + f];
        centers[(i64)step * D + f] = v;
        prev[f] = v;
    }
    if (last) return;
    // block exclusive scan over contiguous segments of the chosen candidate's chunk sums
    const double* cs = chunk_sums + (i64)best * n_chunks;
    const int seg = (n_chunks + SEL_THREADS - 1) / SEL_THREADS;
    const int b0 = min(tid * seg, n_chunks), b1 = min(b0 + seg, n_chunks);
    double loc = 0.0;
    for (int b = b0; b < b1; ++b) loc += cs[b];
    double incl = warp_incl_scan_d(loc, ln);
    if (ln == 31) s_wsum[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        double v = ln < NW ? s_wsum[ln] : 0.0;
        double vi = warp_incl_scan_d(v, ln);
        __syncwarp();
        if (ln < NW) s_wsum[ln + 1] = vi;  // s_wsum[q] = exclusive prefix of warp q
        if (ln == 0) s_wsum[0] = 0.0;
    }
    __syncthreads();
    const double lo = s_wsum[wid] + incl - loc;            // this thread's exclusive prefix
    const double hi_next = __shfl_down_sync(0xffffffffu, lo, 1);
    const double hi = ln == 31 ? s_wsum[wid + 1] : hi_next;  // next thread's exclusive prefix
    const double total = s_wsum[NW];
    if (tid == 0) *total_out = total;
    // The intervals [lo, hi) partition [0, total): each draw has exactly one owner thread.
    for (int t = 0; t < T; ++t) {
        const double target = u[t] * total;
        if (lo <= target && (target < hi || tid == SEL_THREADS - 1)) {
            double run = lo, before = lo;
            int pick = -1;
            for (int b = b0; b < b1; ++b) {
                before = run; run += cs[b];
                if (cs[b] > 0.0 && run > target) { pick = b; break; }
            }
            double r = target - before;
            if (pick < 0) {
                // rounding left the draw past this segment's positive chunks: take the last
                // positive chunk at or before it, and its last positive sample
                for (int b = b1 - 1; b >= 0 && pick < 0; --b) if (cs[b] > 0.0) pick = b;
                if (pick < 0) pick = 0;  // total == 0, select_b draws uniformly
                r = __longlong_as_double(0x7ff0000000000000LL);  // +inf
            }
            tgt_chunk[t] = pick;
            resid[t] = r;
        }
    }
}

// One block per trial: recompute the weighted updated distances of the chunk, pick the sample.
extern "C" __global__ void __launch_bounds__(SELB_THREADS)
kpp_select_b(const float* __restrict__ X, const float* __restrict__ w, const float* __restrict__ d,
             const float* __restrict__ prev, const int* __restrict__ tgt_chunk,
             const double* __restrict__ resid, const double* __restrict__ total,
             const double* __restrict__ u, float* __restrict__ cands, i64* __restrict__ cand_idx,
             const i64 n) {
    __shared__ __align__(16) float s_c[D];
    __shared__ float s_v[CH];
    __shared__ i64 s_pick;
    constexpr int NG = SELB_THREADS / G;
    const int t = blockIdx.x;
    for (int f = threadIdx.x; f < D; f += SELB_THREADS) s_c[f] = prev[f];
    __syncthreads();
    const i64 start = (i64)tgt_chunk[t] * CH;
    const int lane = threadIdx.x % G, grp = threadIdx.x / G;
    for (int it = 0; it < (CH + NG * R - 1) / (NG * R); ++it) {
        i64 rows[R]; bool valid[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int j = (it * NG + grp) * R + r;
            rows[r] = start + j;
            valid[r] = j < CH && rows[r] < n;
        }
        float acc[R][1];
        rows_dists<1, R>(X, rows, valid, s_c, lane, acc);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int j = (it * NG + grp) * R + r;
                if (j < CH)
                    s_v[j] = valid[r] ? (WEIGHTED ? w[rows[r]] : 1.f) * fminf(d[rows[r]], acc[r][0]) : 0.f;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int ln = threadIdx.x;
        constexpr int PER = CH / 32;
        double loc = 0.0;
        for (int q = 0; q < PER; ++q) loc += s_v[ln * PER + q];
        const double incl = warp_incl_scan_d(loc, ln);
        const double lo = incl - loc;
        const double hi = __shfl_down_sync(0xffffffffu, lo, 1);
        const double ctot = __shfl_sync(0xffffffffu, incl, 31);
        const double r = fmin(resid[t], ctot);
        const bool owner = (ln == 31) ? (lo <= r) : (lo <= r && r < hi);
        i64 pick = -1;
        if (owner) {
            // first positive sample past the draw, else the last positive one (rounding)
            double run = lo;
            for (int q = 0; q < PER; ++q) {
                const float v = s_v[ln * PER + q];
                run += v;
                if (v > 0.f) { pick = start + ln * PER + q; if (run > r) break; }
            }
        }
        // the lane with the pick (at most one) publishes it; fall back if none
        const unsigned m = __ballot_sync(0xffffffffu, pick >= 0);
        if (ln == 0) s_pick = -1;
        __syncwarp();
        if (m && ln == __ffs(m) - 1) s_pick = pick;
        __syncwarp();
        if (ln == 0) {
            i64 p = s_pick;
            if (p < 0) {
                // rounding put the draw in a part of the chunk with zero weight, or all distances are 0
                for (int j = CH - 1; j >= 0 && p < 0; --j) if (s_v[j] > 0.f) p = start + j;
                if (p < 0 || *total <= 0.0) p = min((i64)(u[t] * n), n - 1);
            }
            s_pick = p;
            cand_idx[t] = p;
        }
    }
    __syncthreads();
    const i64 pick = s_pick;
    for (int f = threadIdx.x; f < D; f += SELB_THREADS) cands[t * D + f] = X[pick * D + f];
}
