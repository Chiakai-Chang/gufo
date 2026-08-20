#include "src/core/quant/ggml_dequant.hpp"

#include <array>
#include <cmath>
#include <cstring>

namespace strix::quant {

// Standard half-precision float to single-precision float conversion
float Fp16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      float zero = 0.0f;
      std::memcpy(&zero, &sign, sizeof(float));
      return zero;
    }
    const float f =
        (static_cast<float>(mant) / 1024.0f) * std::ldexp(1.0f, -14);
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    const std::uint32_t val = sign | 0x7F800000U | (mant << 13);
    float f = 0.0f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
  }

  const std::uint32_t val = sign | ((exp + 112) << 23) | (mant << 13);
  float f = 0.0f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

#pragma pack(push, 1)
struct block_q4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

struct block_q6_K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

struct block_q3_K {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  std::uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q6_K) == 210);
static_assert(sizeof(block_q3_K) == 110);

namespace {

void GetQ4ScaleMin(std::size_t index, const std::uint8_t* packed,
                   std::uint8_t& scale, std::uint8_t& minimum) noexcept {
  if (index < 4) {
    scale = packed[index] & 0x3FU;
    minimum = packed[index + 4] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                    ((packed[index - 4] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                      ((packed[index] >> 6U) << 4U));
}

std::array<std::int8_t, 16> UnpackQ3Scales(
    const std::uint8_t* packed) noexcept {
  std::array<std::int8_t, 16> scales{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto low =
        index < 8 ? packed[index] & 0x0FU : (packed[index - 8] >> 4U) & 0x0FU;
    const auto high = (packed[8 + (index % 4)] >> (2U * (index / 4))) & 0x03U;
    scales[index] =
        static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
  }
  return scales;
}

float Q4Value(const block_q4_K& block, std::size_t index) noexcept {
  const std::size_t group = index / 32;
  const std::size_t pair = group / 2;
  const bool high_nibble = (group & 1U) != 0;
  const std::size_t lane = index % 32;
  const std::uint8_t packed = block.qs[(pair * 32) + lane];
  const std::uint8_t quant = high_nibble ? packed >> 4U : packed & 0x0FU;
  std::uint8_t scale = 0;
  std::uint8_t minimum = 0;
  GetQ4ScaleMin(group, block.scales, scale, minimum);
  return Fp16ToFloat(block.d) * static_cast<float>(scale) *
             static_cast<float>(quant) -
         Fp16ToFloat(block.dmin) * static_cast<float>(minimum);
}

float Q6Value(const block_q6_K& block, std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t segment = within_half / 32;
  const std::size_t lane = within_half % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }

  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const auto quant =
      static_cast<std::int8_t>(static_cast<int>((high << 4U) | low) - 32);
  return Fp16ToFloat(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

float Q3Value(const block_q3_K& block,
              const std::array<std::int8_t, 16>& scales,
              std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t scale_pair = within_half / 32;
  const std::size_t lane = within_half % 32;
  const std::size_t quant_index = (half * 32) + lane;
  const std::uint8_t shift = static_cast<std::uint8_t>(2U * scale_pair);
  const std::uint8_t high_mask =
      static_cast<std::uint8_t>(1U << ((half * 4) + scale_pair));
  const auto low =
      static_cast<std::int8_t>((block.qs[quant_index] >> shift) & 0x03U);
  const auto quant = static_cast<std::int8_t>(
      static_cast<int>(low) - ((block.hmask[lane] & high_mask) != 0 ? 0 : 4));
  const std::size_t scale_index = (half * 8) + (scale_pair * 2) + (lane / 16);
  return Fp16ToFloat(block.d) * static_cast<float>(scales[scale_index]) *
         static_cast<float>(quant);
}

}  // namespace

std::size_t QuantizedRowBytes(core::GgmlType type,
                              std::size_t elements) noexcept {
  if ((elements % 256U) != 0) {
    return 0;
  }
  const std::size_t blocks = elements / 256U;
  switch (type) {
    case core::GgmlType::kQ3_K:
      return blocks * sizeof(block_q3_K);
    case core::GgmlType::kQ4_K:
      return blocks * sizeof(block_q4_K);
    case core::GgmlType::kQ6_K:
      return blocks * sizeof(block_q6_K);
    default:
      return 0;
  }
}

void DequantizeQ4_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q4Value(blocks[b], i);
    }
  }
}

void DequantizeQ6_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q6Value(blocks[b], i);
    }
  }
}

void DequantizeQ3_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    const auto scales = UnpackQ3Scales(blocks[b].scales);
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q3Value(blocks[b], scales, i);
    }
  }
}

float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q4Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ6_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q6Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ3_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const auto scales = UnpackQ3Scales(blocks[b].scales);
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q3Value(blocks[b], scales, i) * v[i];
    }
  }
  return sum;
}

}  // namespace strix::quant
