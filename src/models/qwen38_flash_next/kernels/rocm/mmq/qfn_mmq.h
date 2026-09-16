// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
extern "C" {
int qfn_mmq_init(int device);

void qfn_mmq_set_routed_max_expert_rows(int rows);

void qfn_mmq_set_routed_tile_cols(int cols);

int qfn_mmq_routed_tile_cols_for_counts(const unsigned int *counts,
                                        int n_experts);

int qfn_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    hipStream_t  stream);

int qfn_mmq_q4_K_moe_pair_unique(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_a,
    float         * out_b,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    hipStream_t    stream);

size_t qfn_mmq_q8_1_bytes(int N, int K);

int qfn_mmq_quantize_q8_1(const float * X_f32, void * X_q8, int N, int K,
                          hipStream_t stream);

int qfn_mmq_q8_0_dense_vec_preq(const void * W_q8_0, const void * W_gate,
                                const void * X_q8, float * out_f32, int M,
                                int N, int K, hipStream_t stream);

int qfn_mmq_build_ids_maps(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, hipStream_t stream);

int qfn_mmq_q8_0_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q8_0_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    hipStream_t    stream);

int qfn_mmq_q4_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q4_K_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    hipStream_t    stream);

int qfn_mmq_q5_1_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_1_moe_vec(const void *W, const float *X_f32, const int32_t *ids,
                         float *out_f32, int M, int K, int n_tokens,
                         int n_experts, int n_expert_used, hipStream_t stream);

int qfn_mmq_q5_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_K_moe_vec(const void *W, const float *X_f32, const int32_t *ids,
                         float *out_f32, int M, int K, int n_tokens,
                         int n_experts, int n_expert_used, hipStream_t stream);
}
