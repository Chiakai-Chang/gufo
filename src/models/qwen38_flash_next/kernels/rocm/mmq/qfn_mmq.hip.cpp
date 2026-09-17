// SPDX-License-Identifier: MIT
// HIP adapters for the llama.cpp-derived quantized kernels. See VENDOR.md.
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "qfn_mmq.h"
#include "common.hpp"
#include "mmq.hpp"
#include "mmvq.hpp"
#include "quantize.hpp"
#include "mmid.hpp"
static int g_routed_max_expert_rows = 0;
static int g_routed_tile_cols = 0;
extern "C" int qfn_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "qfn_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_hip_set_device(device);

    const auto & info = ggml_hip_info();
    if (info.device_count == 0) {
        fprintf(stderr, "qfn_mmq_init: no HIP devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "qfn_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    return 0;
}

extern "C" void qfn_mmq_set_routed_max_expert_rows(int rows) {
    g_routed_max_expert_rows = rows > 0 ? rows : 0;
}

extern "C" void qfn_mmq_set_routed_tile_cols(int cols) {
    g_routed_tile_cols = cols > 0 ? cols : 0;
}

extern "C" int qfn_mmq_routed_tile_cols_for_counts(
        const unsigned int *counts, int n_experts) {

    if (!counts || n_experts <= 0) return 0;
    constexpr int kPanel = 16;
    int best_cols = 0;
    long long best_cost = 0;
    for (int w = 16; w <= 80; w += 16) {
        long long cost = 0;
        for (int e = 0; e < n_experts; e++) {
            const unsigned int c = counts[e];
            if (c == 0u) continue;
            cost += (long long)((c + (unsigned int)w - 1u) / (unsigned int)w) *
                    (long long)(w + kPanel);
        }
        if (best_cols == 0 || cost < best_cost) {
            best_cols = w;
            best_cost = cost;
        }
    }
    return best_cols;
}

__global__ static void qfn_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void qfn_mmq_sanitize_f32(float *p, uint64_t n, hipStream_t stream) {
    if (!p || n == 0) return;
    qfn_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ggml_backend_hip_context * get_ctx_for_device(int device) {
    static std::unique_ptr<ggml_backend_hip_context> cached[GGML_HIP_MAX_DEVICES];
    if (device < 0 || device >= GGML_HIP_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = std::make_unique<ggml_backend_hip_context>(device);
    }
    return cached[device].get();
}

template <ggml_type type>
int qfn_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        hipStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 32 != 0) {

        fprintf(stderr, "%s: K=%d must be a multiple of 32\n", tag, K);
        return -1;
    }

    const int dev = ggml_hip_get_device();
    const int cc  = ggml_hip_info().devices[dev].cc;

    ggml_backend_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * y_block_size /
            y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_hip_pool_alloc<char> src1_q8_1_pool(ctx->pool(), nbytes_src1_q8_1);
    char *src1_q8_1 = src1_q8_1_pool.get();

        quantize_mmq_q8_1_hip(
            X_f32, nullptr, (void *)src1_q8_1,
            type, K, (int64_t)K, 0, 0,
            ne10_padded, ne11, ne12, ne13,
            stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k = false;

    mmq_args args = {
        (const char *)W,
        type,
        (const int *)src1_q8_1,
        nullptr,
        nullptr,
        out_f32,
        ne00,    (int64_t)M,    ne11,
        s01,ne11,          s1,
        1,   1,
        0, s12, 0,
        1,    1,
        0, s13, 0,
        use_stream_k,
        ne11,
    };
    const bool inline_sanitize =
        type == GGML_TYPE_Q8_0 && !use_stream_k;
    args.sanitize_output = inline_sanitize;

    mul_mat_q_case<type>(*ctx, args, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, hipGetErrorString(err));
        return -3;
    }
    if (!inline_sanitize) {
        qfn_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    }
    return 0;
}

extern "C" int qfn_mmq_q8_0_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, hipStream_t stream) {
    return qfn_mmq_dense_impl<GGML_TYPE_Q8_0>("qfn_mmq_q8_0_dense", W, X, out, M, N, K, stream);
}

template <ggml_type type>
int qfn_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        hipStream_t    stream,
        const void * W_b = nullptr, float *out_b = nullptr) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 32 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 32\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_hip_get_device();
    const int cc  = ggml_hip_info().devices[dev].cc;

    ggml_backend_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;

    ggml_hip_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_hip_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_hip_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    hipMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
    hipMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);

    const int si1  = n_expert_used;
    const int sis1 = 1;

    ggml_hip_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, (int)ne11, si1, sis1, stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_hip_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;

        quantize_mmq_q8_1_hip(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, K, s11_src, s12_src, s13_src,
            ne10_padded, ne_get_rows, 1, 1,
            stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                tag, hipGetErrorString(err));
        return -3;
    }

    const int64_t s1            = (int64_t)M;

    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k = false;

    mmq_args args = {
        (const char *)W,
        type,
        (const int *)src1_q8_1.get(),
        ids_dst.get(),
        expert_bounds.get(),
        out_f32,
        ne00,
        (int64_t)M,
        ne_get_rows,
        s01,
        ne_get_rows,
        s1,
        (int64_t)n_experts,
        (int64_t)n_experts,
        s02,
        s12_mmq,
        (int64_t)0,
        1,
        1,
        0,
        s13_mmq,
        0,
        use_stream_k,
        W_b ? n_tokens : ne_get_rows,
        nullptr,
        0,
    };

    args.ncols_grid_max =
        g_routed_max_expert_rows > 0 &&
        (int64_t)g_routed_max_expert_rows < ne_get_rows
            ? (int64_t)g_routed_max_expert_rows
            : 0;
    args.mmq_x_request = g_routed_tile_cols;

    mul_mat_q_case<type>(*ctx, args, stream);
    if (W_b) {
        args.x = static_cast<const char *>(W_b);
        args.dst = out_b;
        mul_mat_q_case<type>(*ctx, args, stream);
    }

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, hipGetErrorString(err));
        return -4;
    }

    return 0;
}

template <ggml_type type>
int qfn_mmq_moe_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        hipStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 32 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 32\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_hip_get_device();
    ggml_backend_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 =
        (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_hip_pool_alloc<char> src1_q8_1_pool;
    src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
    char* src1_q8_1_ptr = src1_q8_1_pool.get();

    quantize_row_q8_1_hip(
        X_f32, nullptr, (void *)src1_q8_1_ptr,
        type, K,
        (int64_t)K, (int64_t)K, (int64_t)K * n_tokens,
        ne10_padded, 1, n_tokens, 1,
        stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_hip failed: %s\n",
                tag, hipGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int64_t s2_dst    = (int64_t)n_expert_used * M;

    const int ids_stride = n_expert_used;

    ggml_hip_mm_fusion_args_device fusion = {};

    const int cc      = ggml_hip_info().devices[dev].cc;
    const int col_cap = get_mmvq_mmid_max_batch(type, ggml_hip_highest_compiled_arch(cc));

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        mul_mat_vec_q_switch_type(
            W, type,
            (const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            ids + (size_t)c0 * ids_stride, fusion,
            out_f32 + (int64_t)c0 * s2_dst,
            K, M, ncols,
            (int)s01_row,
            (int)s12_y,
            (int)s2_dst,
            n_experts,
            1,
            n_expert_used,
            (int)s02_chan,
            (int)s11_y,
            (int)s1_dst,
            1, 1,
            0, 0, 0,
            ids_stride, stream);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s (cols %d..%d cap %d)\n",
                    tag, hipGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }
    }

    return 0;
}

extern "C" int qfn_mmq_q8_0_moe_raw(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_impl<GGML_TYPE_Q8_0>("qfn_mmq_q8_0_moe_raw", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            nullptr, nullptr);
}

extern "C" int qfn_mmq_q8_0_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_vec_impl<GGML_TYPE_Q8_0>(
        "qfn_mmq_q8_0_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int qfn_mmq_q4_K_moe_raw(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_impl<GGML_TYPE_Q4_K>("qfn_mmq_q4_K_moe_raw", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            nullptr, nullptr);
}

extern "C" int qfn_mmq_q4_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_vec_impl<GGML_TYPE_Q4_K>(
        "qfn_mmq_q4_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int qfn_mmq_q5_1_moe_raw(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_impl<GGML_TYPE_Q5_1>("qfn_mmq_q5_1_moe_raw", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            nullptr, nullptr);
}

extern "C" int qfn_mmq_q5_1_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_vec_impl<GGML_TYPE_Q5_1>(
        "qfn_mmq_q5_1_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int qfn_mmq_q5_K_moe_raw(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_impl<GGML_TYPE_Q5_K>("qfn_mmq_q5_K_moe_raw", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            nullptr, nullptr);
}

extern "C" int qfn_mmq_q5_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        hipStream_t stream) {
    return qfn_mmq_moe_vec_impl<GGML_TYPE_Q5_K>(
        "qfn_mmq_q5_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int qfn_mmq_q4_K_moe_pair_unique(
    const void * W_a, const void * W_b, const float * X, const int32_t * ids,
    float * out_a, float * out_b, int M, int K, int n_tokens, int n_experts,
    int n_expert_used, hipStream_t stream) {
    if (!W_b || !out_b) return -1;
    return qfn_mmq_moe_impl<GGML_TYPE_Q4_K>("qfn_mmq_q4_K_moe_pair_unique",
        W_a, X, ids, out_a, M, K, n_tokens, n_experts, n_expert_used, stream,
        W_b, out_b);
}

extern "C" size_t qfn_mmq_q8_1_bytes(int N, int K) {
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    return (size_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1;
}

extern "C" int qfn_mmq_quantize_q8_1(
        const float * X_f32, void * X_q8, int N, int K, hipStream_t stream) {
    if (!X_f32 || !X_q8 || N <= 0 || K <= 0 || K % 32 != 0) {
        fprintf(stderr, "qfn_mmq_quantize_q8_1: bad arguments N=%d K=%d\n", N, K);
        return -1;
    }
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    quantize_row_q8_1_hip(
        X_f32, nullptr, X_q8, GGML_TYPE_Q8_0, K,
        (int64_t)K, (int64_t)K * N, (int64_t)K * N,
        ne10_padded, N, 1, 1, stream);
    return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int qfn_mmq_q8_0_dense_vec_preq(
        const void * W, const void * W_gate, const void * X_q8, float * out_f32,
        int M, int N, int K, hipStream_t stream) {
    if (!W || !X_q8 || !out_f32 || M <= 0 || N <= 0 || K <= 0 || K % 32 != 0 ||
        N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "qfn_mmq_q8_0_dense_vec_preq: bad arguments M=%d N=%d K=%d\n",
                M, N, K);
        return -1;
    }
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t s01_row = (int64_t)K / ggml_blck_size(GGML_TYPE_Q8_0);
    const int64_t s11_y   = ne10_padded / QK8_1;
    const int64_t s12_y   = (int64_t)N * s11_y;
    ggml_hip_mm_fusion_args_device fusion = {};
    fusion.gate = W_gate;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    mul_mat_vec_q_switch_type(
        W, GGML_TYPE_Q8_0, X_q8, nullptr, fusion, out_f32,
        K, M, N,
        (int)s01_row, (int)s11_y,
        M, 1, 1, 1, 0, (int)s12_y, 0, 1, 1, 0, 0, 0,
        0, stream);
    return hipGetLastError() == hipSuccess ? 0 : -3;
}
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_1>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
} // namespace qfn_mmq
