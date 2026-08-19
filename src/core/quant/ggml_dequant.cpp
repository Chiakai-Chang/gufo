#include "src/core/quant/ggml_dequant.hpp"

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

void DequantizeQ4_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    const float min = Fp16ToFloat(blocks[b].dmin);
    const auto* scales = blocks[b].scales;
    const auto* qs = blocks[b].qs;

    for (std::size_t i = 0; i < 256; ++i) {
      const std::size_t is = i / 32;
      const std::size_t sc_idx = is < 4 ? is : is + 4;
      const std::uint8_t sc = (scales[sc_idx / 2] >> ((sc_idx % 2) * 4)) & 0x0F;
      const std::uint8_t q = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
      dst[b * 256 + i] =
          d * static_cast<float>(sc) * static_cast<float>(q) - min;
    }
  }
}

void DequantizeQ6_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    const auto* ql = blocks[b].ql;
    const auto* qh = blocks[b].qh;
    const auto* sc = blocks[b].scales;

    for (std::size_t i = 0; i < 256; ++i) {
      const std::size_t is = i / 16;
      const std::int8_t scale = sc[is];
      const std::uint8_t l = (ql[i / 2] >> ((i % 2) * 4)) & 0x0F;
      const std::uint8_t h = (qh[i / 4] >> ((i % 4) * 2)) & 0x03;
      const std::int8_t q = static_cast<std::int8_t>((h << 4) | l) - 32;
      dst[b * 256 + i] = d * static_cast<float>(scale) * static_cast<float>(q);
    }
  }
}

void DequantizeQ3_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    const auto* qs = blocks[b].qs;
    for (std::size_t i = 0; i < 256; ++i) {
      const std::uint8_t q = (qs[i / 4] >> ((i % 4) * 2)) & 0x03;
      dst[b * 256 + i] = d * static_cast<float>(static_cast<int>(q) - 2);
    }
  }
}

float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    const float min = Fp16ToFloat(blocks[b].dmin);
    const auto* scales = blocks[b].scales;
    const auto* qs = blocks[b].qs;
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      const std::size_t is = i / 32;
      const std::size_t sc_idx = is < 4 ? is : is + 4;
      const std::uint8_t sc = (scales[sc_idx / 2] >> ((sc_idx % 2) * 4)) & 0x0F;
      const std::uint8_t q = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
      const float w = d * static_cast<float>(sc) * static_cast<float>(q) - min;
      sum += w * v[i];
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
    const float d = Fp16ToFloat(blocks[b].d);
    const auto* ql = blocks[b].ql;
    const auto* qh = blocks[b].qh;
    const auto* sc = blocks[b].scales;
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      const std::size_t is = i / 16;
      const std::int8_t scale = sc[is];
      const std::uint8_t l = (ql[i / 2] >> ((i % 2) * 4)) & 0x0F;
      const std::uint8_t h = (qh[i / 4] >> ((i % 4) * 2)) & 0x03;
      const std::int8_t q = static_cast<std::int8_t>((h << 4) | l) - 32;
      const float w = d * static_cast<float>(scale) * static_cast<float>(q);
      sum += w * v[i];
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
    const float d = Fp16ToFloat(blocks[b].d);
    const auto* qs = blocks[b].qs;
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      const std::uint8_t q = (qs[i / 4] >> ((i % 4) * 2)) & 0x03;
      const float w = d * static_cast<float>(static_cast<int>(q) - 2);
      sum += w * v[i];
    }
  }
  return sum;
}

}  // namespace strix::quant
