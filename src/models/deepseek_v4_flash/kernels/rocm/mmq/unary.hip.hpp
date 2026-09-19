#pragma once
#include "common.hip.hpp"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ds4_ggml_hip_op_abs(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_sgn(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_neg(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_step(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_gelu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_silu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_silu_back(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_gelu_erf(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_gelu_quick(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_tanh(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_relu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_sigmoid(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_hardsigmoid(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_exp(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_hardswish(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_leaky_relu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_sqr(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_sqrt(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_sin(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_cos(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_log(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_expm1(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_softplus(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_elu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_floor(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_ceil(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_round(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_trunc(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_reglu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_geglu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_swiglu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_swiglu_oai(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_geglu_erf(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_geglu_quick(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_xielu(ds4_ggml_hip_context & ctx, ggml_tensor * dst);

void ds4_ggml_hip_op_unary_mul(ds4_ggml_hip_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void ds4_ggml_hip_op_relu_sqr(ds4_ggml_hip_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float ds4_ggml_hip_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ds4_ggml_hip_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ds4_ggml_hip_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
