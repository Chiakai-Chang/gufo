#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

// Flash Next uses one wave per dense row and per routed token on gfx1151.
template <int ncols_dst, bool has_gate>
__launch_bounds__(32, 1)
static __global__ void mul_mat_vec_q8(
        const void* __restrict__ weights, const void* __restrict__ gate,
        const block_q8_1* __restrict__ input, float* __restrict__ output,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y) {
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * 32 / qi;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int blocks_per_row = ncols_x / QK8_0;
    const int row_offset = row * blocks_per_row;
    const int kqs = vdr * (lane % (qi / vdr));
    float sum[ncols_dst] = {};
    float gate_sum[ncols_dst] = {};
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sum[j] += vec_dot_q8_0_q8_1(weights, input + j * stride_col_y + kbx,
                                      row_offset + kbx, kqs);
            if constexpr (has_gate) {
                gate_sum[j] += vec_dot_q8_0_q8_1(gate, input + j * stride_col_y + kbx,
                                               row_offset + kbx, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sum[j] = warp_reduce_sum<32>(sum[j]);
        if constexpr (has_gate) gate_sum[j] = warp_reduce_sum<32>(gate_sum[j]);
        if (lane == 0) {
            float value = sum[j];
            if constexpr (has_gate) value *= ggml_hip_op_silu_single(gate_sum[j]);
            output[j * nrows_x + row] = value;
        }
    }
}

typedef float (*vec_dot_q_hip_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_hip_t get_vec_dot_q_hip(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        default:                return 1;
    }
}

template<ggml_type type, int c_rows_per_block, bool gated = false>
__launch_bounds__(mmvq_moe_max_batch(type) * (gated ? 64 : 32),
                  1) static __global__
    void mul_mat_vec_q_moe(const void* __restrict__ weights,
                           const void* __restrict__ vy,
                           const int32_t* __restrict__ ids,
                           float* __restrict__ dst, const uint32_t ncols_x,
                           const uint32_t nrows_x, const uint32_t stride_row_x,
                           const uint32_t stride_col_y,
                           const uint32_t stride_col_dst,
                           const uint32_t stride_channel_x,
                           const uint32_t stride_channel_dst,
                           const uint32_t ncols_dst, const uint32_t ids_stride,
                           const void* __restrict__ up_weights = nullptr) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int warp_size = 32;

  constexpr vec_dot_q_hip_t vec_dot_q_hip = get_vec_dot_q_hip(type);

  const bool is_up = gated && threadIdx.y != 0;
  const uint32_t token_idx = gated ? 0 : threadIdx.y;
  const void* vx = is_up ? up_weights : weights;
  const int row0 = c_rows_per_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * warp_size / qi;

  const uint32_t channel_dst = blockIdx.y;

  if (token_idx >= ncols_dst) {
    return;
  }

  // Inactive experts still write zero to every output row. The expert ID
  // and its validity are uniform within a wave.
  const int32_t id_raw = ids[channel_dst + token_idx * ids_stride];
  const bool invalid_id = id_raw < 0;
  const uint32_t channel_x = invalid_id ? 0u : (uint32_t)id_raw;

  const block_q8_1* y = ((const block_q8_1*)vy) + token_idx * stride_col_y;
  uint32_t row_offsets[c_rows_per_block];
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        // A ragged tile reads the last valid row again for its unused lane.
        // Keep the bounds check out of the quantized dot-product loop.
        const uint32_t row = min(uint32_t(row0 + i), nrows_x - 1);
        row_offsets[i] = channel_x*stride_channel_x + row*stride_row_x;
    }

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_hip(vx, &y[kby], row_offsets[i] + kbx, kqs);
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if constexpr (gated) {
      // Separate waves preserve each projection's original dot-product
      // grouping. Combining two accumulators in one wave lets fast-math
      // reassociate their shared input scales.
      __shared__ float values[2][c_rows_per_block];
      if (threadIdx.x < c_rows_per_block) {
        const float value = tmp[threadIdx.x];
        values[is_up][threadIdx.x] = isfinite(value) ? value : 0.0f;
      }
      __syncthreads();
      if (!is_up && threadIdx.x < c_rows_per_block &&
          uint32_t(row0 + threadIdx.x) < nrows_x) {
        const float g = values[0][threadIdx.x];
        const float u = values[1][threadIdx.x];
        dst[channel_dst * stride_channel_dst + row0 + threadIdx.x] =
            (g * (1.0f / (1.0f + __expf(-g)))) * u;
      }
      return;
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        const float value = tmp[threadIdx.x];
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] =
            isfinite(value) ? value : 0.0f;
    }
}

template <int tokens>
static void launch_q8(const void* weights, const void* gate, const block_q8_1* input,
                      float* output, int k, int rows, int input_stride, hipStream_t stream) {
    if (gate) {
        mul_mat_vec_q8<tokens, true><<<rows, 32, 0, stream>>>(
            weights, gate, input, output, k, rows, input_stride);
    } else {
        mul_mat_vec_q8<tokens, false><<<rows, 32, 0, stream>>>(
            weights, nullptr, input, output, k, rows, input_stride);
    }
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                            const block_q8_1* input, float* output,
                            int k, int rows, int tokens, int input_stride,
                            hipStream_t stream) {
    GGML_ASSERT(k % QK8_0 == 0 && rows > 0);
    switch (tokens) {
        case 1: launch_q8<1>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 2: launch_q8<2>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 3: launch_q8<3>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 4: launch_q8<4>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 5: launch_q8<5>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 6: launch_q8<6>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 7: launch_q8<7>(weights, gate, input, output, k, rows, input_stride, stream); break;
        case 8: launch_q8<8>(weights, gate, input, output, k, rows, input_stride, stream); break;
        default: GGML_ABORT("invalid vector batch width");
    }
}

template<ggml_type type, int rows_per_wave = 2>
static void launch_moe(const void* weights, const block_q8_1* input,
                       const int32_t* ids, float* output, int k, int rows,
                       int tokens, int experts_used, int input_stride,
                       hipStream_t stream) {
  GGML_ASSERT(k % ggml_blck_size(type) == 0 && rows > 0);
  GGML_ASSERT(tokens > 0 && tokens <= mmvq_moe_max_batch(type));
  const int row_stride = k / ggml_blck_size(type);
  mul_mat_vec_q_moe<type, rows_per_wave>
      <<<dim3((rows + rows_per_wave - 1) / rows_per_wave, experts_used),
         dim3(32, tokens), 0, stream>>>(
          weights, input, ids, output, k, rows, row_stride, input_stride,
          rows * experts_used, rows * row_stride, rows, tokens, experts_used);
}

void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q5_1:
            launch_moe<GGML_TYPE_Q5_1>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
          // Down projection slots have independent inputs. Four rows reuse
          // each short input across more weights without changing its sum.
          if (k == 640 && experts_used == 1 && tokens > 1) {
            launch_moe<GGML_TYPE_Q8_0, 4>(weights, input, ids, output, k, rows,
                                          tokens, experts_used, input_stride,
                                          stream);
            break;
          }
            launch_moe<GGML_TYPE_Q8_0>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_moe<GGML_TYPE_Q4_K>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_moe<GGML_TYPE_Q5_K>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        default: GGML_ABORT("unsupported vector weight format");
    }
}

void mul_mat_vec_moe_gated_decode(const void* gate, const void* up,
                                  ggml_type type, const block_q8_1* input,
                                  const int32_t* ids, float* output, int k,
                                  int rows, int experts_used, int input_stride,
                                  hipStream_t stream) {
  const int row_stride = k / ggml_blck_size(type);
  const dim3 grid((rows + 1) / 2, experts_used);
  const dim3 block(32, 2);
  if (type == GGML_TYPE_Q4_K) {
    mul_mat_vec_q_moe<GGML_TYPE_Q4_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else {
    GGML_ASSERT(type == GGML_TYPE_Q5_K);
    mul_mat_vec_q_moe<GGML_TYPE_Q5_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  }
}

}  // namespace qfn_mmq
