#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

// MTP shortlists with a private Q4 head, then rescores using the original Q8.
static __global__ void requantize_q8_0_q4_1_kernel(const block_q8_0* source,
                                                   block_q4_1* destination,
                                                   size_t blocks) {
  const size_t block = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= blocks)
    return;
  float values[32];
  float low = INFINITY, high = -INFINITY;
  const float scale = __half2float(source[block].d);
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    values[i] = scale * static_cast<float>(source[block].qs[i]);
    low = fminf(low, values[i]);
    high = fmaxf(high, values[i]);
  }
  const float delta = (high - low) / 15.0f;
  const float inverse = delta != 0.0f ? 1.0f / delta : 0.0f;
  auto& out = destination[block];
  out.dm = __floats2half2_rn(delta, low);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const unsigned a = min(15, int((values[i] - low) * inverse + 0.5f));
    const unsigned b = min(15, int((values[i + 16] - low) * inverse + 0.5f));
    out.qs[i] = a | (b << 4);
  }
}

__launch_bounds__(32) static __global__
    void mul_mat_vec_q4_1(const void* weights, const block_q8_1* input,
                          float* output, int cols) {
  constexpr int vdr = VDR_Q4_1_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / QI4_1;
  const int lane = threadIdx.x;
  const int blocks_per_row = cols / QK4_1;
  const int row_offset = blockIdx.x * blocks_per_row;
  const int part = vdr * (lane % (QI4_1 / vdr));
  float sum = 0.0f;
  for (int kb = lane / (QI4_1 / vdr); kb < blocks_per_row;
       kb += blocks_per_iter) {
    sum += vec_dot_q4_1_q8_1(weights, input + kb, row_offset + kb, part);
  }
  sum = warp_reduce_sum<32>(sum);
  if (lane == 0)
    output[blockIdx.x] = sum;
}

extern "C" int qfn_mmq_requantize_q8_0_q4_1(const void* source,
                                            void* destination, int rows,
                                            int cols, hipStream_t stream) {
  if (!source || !destination || rows <= 0 || cols <= 0 || cols % 32)
    return -1;
  const size_t blocks = size_t(rows) * cols / 32;
  requantize_q8_0_q4_1_kernel<<<(blocks + 255) / 256, 256, 0, stream>>>(
      static_cast<const block_q8_0*>(source),
      static_cast<block_q4_1*>(destination), blocks);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int qfn_mmq_q4_1_dense_vec_preq(const void* weights,
                                           const void* input, float* output,
                                           int rows, int cols,
                                           hipStream_t stream) {
  if (!weights || !input || !output || rows <= 0 || cols <= 0 || cols % 32)
    return -1;
  mul_mat_vec_q4_1<<<rows, 32, 0, stream>>>(
      weights, static_cast<const block_q8_1*>(input), output, cols);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

// Flash Next uses one wave per dense row and per routed token on gfx1151.
template<int ncols_dst, bool has_gate, bool selected = false>
__launch_bounds__(32, 1) static __global__
    void mul_mat_vec_q8(const void* __restrict__ weights,
                        const void* __restrict__ gate,
                        const block_q8_1* __restrict__ input,
                        float* __restrict__ output, const uint32_t ncols_x,
                        const uint32_t nrows_x, const uint32_t stride_col_y,
                        const uint32_t* selected_ids = nullptr) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const int row = selected ? selected_ids[blockIdx.x] : blockIdx.x;
  if constexpr (selected) {
    if (static_cast<uint32_t>(row) >= nrows_x)
      return;
  }
  const int blocks_per_row = ncols_x / QK8_0;
  const int row_offset = row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  float gate_sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
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
    if constexpr (has_gate)
      gate_sum[j] = warp_reduce_sum<32>(gate_sum[j]);
    if (lane == 0) {
      float value = sum[j];
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_sum[j]);
      output[j * nrows_x + row] = value;
    }
  }
}

extern "C" int qfn_mmq_q8_0_selected_vec_preq(
    const void* weights, const void* input, const uint32_t* ids, float* output,
    int count, int rows, int cols, hipStream_t stream) {
  if (!weights || !input || !ids || !output || count <= 0 || rows <= 0 ||
      cols <= 0 || cols % 32)
    return -1;
  mul_mat_vec_q8<1, false, true><<<count, 32, 0, stream>>>(
      weights, nullptr, static_cast<const block_q8_1*>(input), output, cols,
      rows, cols / 32, ids);
  return hipGetLastError() == hipSuccess ? 0 : -2;
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
  const uint32_t token_idx =
      gated ? 0 : blockIdx.z * blockDim.y + threadIdx.y;
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

// Keep one anchor per expert and a slot mask for each token. Duplicate
// experts share their weights; inactive slots still get explicit zeroes.
static __global__ void group_moe_slots(const int32_t* ids, int32_t* groups,
                                       int tokens, int experts_used) {
  const int anchor = blockIdx.x * blockDim.x + threadIdx.x;
  if (anchor >= tokens * experts_used)
    return;
  const int expert = ids[anchor];
  int32_t* group = groups + anchor * (tokens + 1);
  group[0] = -1;
  if (expert < 0) {
    group[0] = -2;
    return;
  }
  for (int i = 0; i < anchor; ++i)
    if (ids[i] == expert)
      return;
  group[0] = expert;
  for (int t = 0; t < tokens; ++t) {
    uint32_t slots = 0;
    for (int j = 0; j < experts_used; ++j)
      if (ids[t * experts_used + j] == expert)
        slots |= uint32_t{1} << j;
    group[t + 1] = static_cast<int32_t>(slots);
  }
}

template<ggml_type type, int tokens>
__launch_bounds__(64) static __global__
    void mul_mat_vec_moe_grouped(const void* __restrict__ gate,
                                 const void* __restrict__ up,
                                 const block_q8_1* __restrict__ input,
                                 const int32_t* __restrict__ groups,
                                 float* __restrict__ output, int k, int rows,
                                 int experts_used, int input_stride) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int blocks_per_iter = vdr * 32 / qi;
  constexpr auto dot = get_vec_dot_q_hip(type);
  const int anchor = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const bool is_up = tid >= 32;
  const int row0 = blockIdx.x * 2;
  const int32_t* group = groups + anchor * (tokens + 1);
  const int expert = group[0];
  if (expert < 0) {
    if (expert == -2 && tid < 2 && row0 + tid < rows)
      output[anchor * rows + row0 + tid] = 0.0f;
    return;
  }

  const void* weights = is_up ? up : gate;
  const int blocks_per_row = k / qk;
  uint32_t slots[tokens];
#pragma unroll
  for (int t = 0; t < tokens; ++t)
    slots[t] = static_cast<uint32_t>(group[t + 1]);
  float sum[tokens][2] = {};
  for (int kb = lane / (qi / vdr); kb < blocks_per_row; kb += blocks_per_iter) {
    const int kqs = vdr * (lane % (qi / vdr));
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (slots[t] == 0)
        continue;
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sum[t][r] +=
            dot(weights, input + t * input_stride + kb * (qk / QK8_1),
                (expert * rows + min(row0 + r, rows - 1)) * blocks_per_row + kb,
                kqs);
    }
  }

  // Gate and up stay in separate waves to retain the scalar projection's
  // sum order, including the treatment of nonfinite quantization scales.
  __shared__ float values[tokens][2][2];
#pragma unroll
  for (int t = 0; t < tokens; ++t) {
    if (slots[t] == 0)
      continue;
#pragma unroll
    for (int r = 0; r < 2; ++r)
      sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
    if (lane < 2)
      values[t][is_up][lane] = isfinite(sum[t][lane]) ? sum[t][lane] : 0.0f;
  }
  __syncthreads();
  if (!is_up && lane < 2 && row0 + lane < rows) {
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (slots[t] == 0)
        continue;
      const float g = values[t][0][lane];
      const float u = values[t][1][lane];
      uint32_t bits = slots[t];
      while (bits) {
        const int slot = __ffs(static_cast<int>(bits)) - 1;
        output[(t * experts_used + slot) * rows + row0 + lane] =
            (g * (1.0f / (1.0f + __expf(-g)))) * u;
        bits &= bits - 1;
      }
    }
  }
}

template<ggml_type type, int tokens = 2>
static void launch_moe_grouped(const void* gate, const void* up,
                               const block_q8_1* input, const int32_t* groups,
                               float* output, int k, int rows, int n_tokens,
                               int experts_used, int input_stride,
                               hipStream_t stream) {
  if (n_tokens == tokens) {
    mul_mat_vec_moe_grouped<type, tokens>
        <<<dim3((rows + 1) / 2, tokens * experts_used), 64, 0, stream>>>(
            gate, up, input, groups, output, k, rows, experts_used,
            input_stride);
  } else if constexpr (tokens < MMVQ_MAX_BATCH_SIZE) {
    launch_moe_grouped<type, tokens + 1>(gate, up, input, groups, output, k,
                                         rows, n_tokens, experts_used,
                                         input_stride, stream);
  } else {
    GGML_ABORT("invalid gated vector batch width");
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
  GGML_ASSERT(tokens > 0);
  const int block_tokens = std::min(tokens, mmvq_moe_max_batch(type));
  const int row_stride = k / ggml_blck_size(type);
  mul_mat_vec_q_moe<type, rows_per_wave>
      <<<dim3((rows + rows_per_wave - 1) / rows_per_wave, experts_used,
              (tokens + block_tokens - 1) / block_tokens),
         dim3(32, block_tokens), 0, stream>>>(
          weights, input, ids, output, k, rows, row_stride, input_stride,
          rows * experts_used, rows * row_stride, rows, tokens, experts_used);
}

void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q5_1:
          // Down projection slots have independent inputs. Wider row tiles
          // reuse each short input without changing its dot-product sum.
          if (k == 640 && experts_used == 1 && tokens > 1) {
            if (tokens > 20) {
              launch_moe<GGML_TYPE_Q5_1, 8>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream);
            } else {
              launch_moe<GGML_TYPE_Q5_1, 4>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream);
            }
            break;
          }
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

void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream) {
  GGML_ASSERT(tokens > 0 && tokens <= MMVQ_MAX_BATCH_SIZE);
  if (tokens > 1) {
    GGML_ASSERT(groups && experts_used <= 32);
    group_moe_slots<<<(tokens * experts_used + 127) / 128, 128, 0, stream>>>(
        ids, groups, tokens, experts_used);
    if (type == GGML_TYPE_Q4_K) {
      launch_moe_grouped<GGML_TYPE_Q4_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else {
      GGML_ASSERT(type == GGML_TYPE_Q5_K);
      launch_moe_grouped<GGML_TYPE_Q5_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    }
    return;
  }
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
