#include "backend.h"
#include <hipblaslt/hipblaslt.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ROCM_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[ROCM_QK_K / 16];
    uint8_t qs[ROCM_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} hip_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[ROCM_QK_K / 2];
} hip_block_q4_K;

typedef struct {
    float d;
    int8_t qs[ROCM_QK_K];
    int16_t bsums[ROCM_QK_K / 16];
} hip_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[ROCM_QK_K / 8];
} hip_block_iq2_xxs;

#include "iq2_tables.inc"

#include "detail/ds4_rocm_runtime.hip.hpp"

#include "detail/ds4_rocm_common.hip.hpp"

#include "detail/ds4_rocm_q8.hip.hpp"

#include "detail/ds4_rocm_norm_rope.hip.hpp"

#include "detail/ds4_rocm_fp8_kv.hip.hpp"

#include "detail/ds4_rocm_attention.hip.hpp"

#include "detail/ds4_rocm_hc.hip.hpp"

#include "detail/ds4_rocm_output.hip.hpp"

#include "detail/ds4_rocm_indexer.hip.hpp"

#include "detail/ds4_rocm_embedding_launch.hip.hpp"

#include "detail/ds4_rocm_matmul.hip.hpp"

#include "detail/ds4_rocm_fp8_kv_launch.hip.hpp"

#include "detail/ds4_rocm_compressor.hip.hpp"

#include "detail/ds4_rocm_attention_launch.hip.hpp"

#include "detail/ds4_rocm_shared_expert.hip.hpp"

#include "detail/ds4_rocm_misc_launch.hip.hpp"
#include "detail/ds4_rocm_router.hip.hpp"

#include "detail/ds4_rocm_moe.hip.hpp"

#include "detail/ds4_rocm_moe_launch.hip.hpp"

#include "detail/ds4_rocm_glm.hip.hpp"

#include "detail/ds4_rocm_hc_output_launch.hip.hpp"

#include "detail/ds4_rocm_current_api_compat.hip.hpp"

/* Tensor-parallel gates are Metal-only; stubs keep shared graph code
 * linkable (TP option validation rejects non-Metal backends). */
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    (void)layer; (void)gate;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    (void)fn;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    (void)fn;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)full_in_dim; (void)k_off; (void)k_cnt; (void)out_dim; (void)x;
    (void)x_elem_off;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    (void)out; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups_total; (void)group0; (void)group_cnt; (void)out_dim;
    (void)heads;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add; (void)residual_hc;
    (void)post; (void)comb; (void)n_embd; (void)n_hc;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}
