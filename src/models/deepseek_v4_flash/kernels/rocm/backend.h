#pragma once

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include <rocwmma/rocwmma-version.hpp>
#include <rocwmma/rocwmma.hpp>

namespace cub = hipcub;

static __device__ __forceinline__ int32_t __vcmpne4(uint32_t a, uint32_t b) {
    // For each byte: 0xFF if a != b, 0x00 if a == b
    uint32_t diff = a ^ b;
    // Spread any set bit in each byte to fill the whole byte
    diff |= (diff >> 1); diff |= (diff >> 2); diff |= (diff >> 4);
    diff &= 0x01010101u;
    diff *= 0xFFu; // 0x01 -> 0xFF per byte
    return (int32_t)diff;
}

static __device__ __forceinline__ int32_t __vsub4(int32_t a, int32_t b) {
    // Per-byte subtraction (wrapping, not saturating)
    uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
    // Trick: subtract bytes in parallel avoiding cross-byte borrows
    uint32_t diff = ((ua | 0x80808080u) - (ub & 0x7F7F7F7Fu)) ^ ((ua ^ ~ub) & 0x80808080u);
    return (int32_t)diff;
}

// __dp4a: dot product of 4 signed int8s packed in an int32.
// gfx11-class AMD GPUs expose this as a single v_dot4_i32_i8 instruction;
// using the clang builtin avoids expanding every Q8/Q8_K dot into scalar byte
// multiplies in the resident ROCm implementation.
static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    union ds4_i8x4_bits { int32_t i; char4 v; } av, bv;
    av.i = a;
    bv.i = b;
    return amd_mixed_dot(av.v, bv.v, c, false);
}

// Precise transcendentals for the MoE router top-k scores, immune to -fapprox-func.
// These functions are to be used on paths where small error can be translated to
// some macro effect - like expert selection kernels
extern "C" __device__ __attribute__((pure))  float __ocml_exp_f32(float);
extern "C" __device__ __attribute__((pure))  float __ocml_log1p_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_sqrt_f32(float);

static __device__ __forceinline__ float ds4_precise_expf(float x)   { return __ocml_exp_f32(x); }
static __device__ __forceinline__ float ds4_precise_log1pf(float x) { return __ocml_log1p_f32(x); }
static __device__ __forceinline__ float ds4_precise_sqrtf(float x)  { return __ocml_sqrt_f32(x); }
