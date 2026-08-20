#include "src/core/quant/ggml_dequant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

constexpr std::size_t kBlockElements = 256;

void ExpectNear(float actual, float expected, std::string_view message,
                float tolerance = 1.0e-5F) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << '\n';
    std::exit(1);
  }
}

std::array<std::uint8_t, 144> MakeQ4Block(
    const std::array<std::uint8_t, 8>& scales,
    const std::array<std::uint8_t, 8>& minima) {
  std::array<std::uint8_t, 144> block{};
  block[0] = 0x00;
  block[1] = 0x3C;
  block[2] = 0x00;
  block[3] = 0x38;
  for (std::size_t index = 0; index < 4; ++index) {
    block[4 + index] = static_cast<std::uint8_t>(
        (scales[index] & 0x3FU) | ((scales[index + 4] >> 4U) << 6U));
    block[8 + index] = static_cast<std::uint8_t>(
        (minima[index] & 0x3FU) | ((minima[index + 4] >> 4U) << 6U));
    block[12 + index] = static_cast<std::uint8_t>(
        (scales[index + 4] & 0x0FU) | ((minima[index + 4] & 0x0FU) << 4U));
  }
  for (std::size_t pair = 0; pair < 4; ++pair) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const auto low = static_cast<std::uint8_t>((lane + pair) & 0x0FU);
      const auto high = static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
      block[16 + (pair * 32) + lane] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return block;
}

std::array<std::uint8_t, 210> MakeQ6Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 210> block{};
  const auto logical_quant = [](std::size_t index) {
    return static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
  };
  for (std::size_t half = 0; half < 2; ++half) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      std::array<std::uint8_t, 4> code{};
      for (std::size_t segment = 0; segment < code.size(); ++segment) {
        code[segment] = static_cast<std::uint8_t>(
            logical_quant((half * 128) + (segment * 32) + lane) + 32);
      }
      block[(half * 64) + lane] = static_cast<std::uint8_t>(
          (code[0] & 0x0FU) | ((code[2] & 0x0FU) << 4U));
      block[(half * 64) + 32 + lane] = static_cast<std::uint8_t>(
          (code[1] & 0x0FU) | ((code[3] & 0x0FU) << 4U));
      block[128 + (half * 32) + lane] = static_cast<std::uint8_t>(
          ((code[0] >> 4U) & 0x03U) | (((code[1] >> 4U) & 0x03U) << 2U) |
          (((code[2] >> 4U) & 0x03U) << 4U) |
          (((code[3] >> 4U) & 0x03U) << 6U));
    }
  }
  std::ranges::copy(scales, block.begin() + 192);
  block[208] = 0x00;
  block[209] = 0x3C;
  return block;
}

std::array<std::uint8_t, 110> MakeQ3Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 110> block{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto encoded = static_cast<std::uint8_t>(scales[index] + 32);
    if (index < 8) {
      block[96 + index] |= encoded & 0x0FU;
    } else {
      block[96 + index - 8] |=
          static_cast<std::uint8_t>((encoded & 0x0FU) << 4U);
    }
    block[104 + (index % 4)] |= static_cast<std::uint8_t>(
        ((encoded >> 4U) & 0x03U) << (2U * (index / 4)));
  }

  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t pair = within_half / 32;
    const std::size_t lane = within_half % 32;
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    const auto low = static_cast<std::uint8_t>(quant < 0 ? quant + 4 : quant);
    block[32 + (half * 32) + lane] |=
        static_cast<std::uint8_t>(low << (2U * pair));
    if (quant >= 0) {
      block[lane] |= static_cast<std::uint8_t>(1U << ((half * 4) + pair));
    }
  }
  block[108] = 0x00;
  block[109] = 0x3C;
  return block;
}

void TestQ4() {
  const std::array<std::uint8_t, 8> scales{1, 7, 15, 31, 32, 47, 55, 63};
  const std::array<std::uint8_t, 8> minima{0, 1, 2, 3, 4, 5, 6, 7};
  const auto block = MakeQ4Block(scales, minima);
  std::array<float, kBlockElements> actual{};
  strix::quant::DequantizeQ4_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t group = index / 32;
    const std::size_t pair = group / 2;
    const std::size_t lane = index % 32;
    const auto quant =
        group % 2 == 0 ? static_cast<std::uint8_t>((lane + pair) & 0x0FU)
                       : static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
    expected[index] = static_cast<float>(scales[group] * quant) -
                      (0.5F * static_cast<float>(minima[group]));
    ExpectNear(actual[index], expected[index], "Q4_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.5F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(strix::quant::DotProductQ4_K(block.data(), vector, vector.size()),
             expected_dot, "Q4_K dot product");
}

void TestQ6() {
  const std::array<std::int8_t, 16> scales{-8, -7, -6, -5, -4, -3, -2, -1,
                                           1,  2,  3,  4,  5,  6,  7,  8};
  const auto block = MakeQ6Block(scales);
  std::array<float, kBlockElements> actual{};
  strix::quant::DequantizeQ6_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t segment = within_half / 32;
    const std::size_t lane = within_half % 32;
    const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q6_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.25F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(strix::quant::DotProductQ6_K(block.data(), vector, vector.size()),
             expected_dot, "Q6_K dot product");
}

void TestQ3() {
  const std::array<std::int8_t, 16> scales{-32, -27, -18, -9, -1, 0,  1,  7,
                                           12,  18,  23,  27, 29, 30, 31, -16};
  const auto block = MakeQ3Block(scales);
  std::array<float, kBlockElements> actual{};
  strix::quant::DequantizeQ3_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t scale_index =
        (half * 8) + ((within_half / 32) * 2) + ((within_half % 32) / 16);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q3_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), 0.125F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(strix::quant::DotProductQ3_K(block.data(), vector, vector.size()),
             expected_dot, "Q3_K dot product");
}

struct ModelSample {
  std::size_t index;
  float value;
};

void TestModelRow(const strix::core::GgufReader& reader, std::string_view name,
                  strix::core::GgmlType expected_type, std::size_t expected_k,
                  std::span<const ModelSample> samples, double expected_dot) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr ||
      tensor->dimensions.empty()) {
    std::cerr << "Assertion failed: missing model tensor " << name << '\n';
    std::exit(1);
  }
  if (tensor->type != expected_type ||
      tensor->dimensions.front() != expected_k) {
    std::cerr << "Assertion failed: model tensor contract " << name << '\n';
    std::exit(1);
  }

  std::vector<float> row(expected_k);
  switch (tensor->type) {
    case strix::core::GgmlType::kQ4_K:
      strix::quant::DequantizeQ4_K(tensor->data, row.data(), row.size());
      break;
    case strix::core::GgmlType::kQ6_K:
      strix::quant::DequantizeQ6_K(tensor->data, row.data(), row.size());
      break;
    default:
      std::cerr << "Assertion failed: unsupported model tensor type " << name
                << '\n';
      std::exit(1);
  }

  for (const auto& sample : samples) {
    ExpectNear(row[sample.index], sample.value, "real GGUF dequant value",
               1.0e-8F);
  }

  std::vector<float> input(expected_k);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<int>(index % 257) - 128) / 64.0F;
  }
  const float actual_dot =
      tensor->type == strix::core::GgmlType::kQ4_K
          ? strix::quant::DotProductQ4_K(tensor->data, input, input.size())
          : strix::quant::DotProductQ6_K(tensor->data, input, input.size());
  ExpectNear(actual_dot, static_cast<float>(expected_dot),
             "real GGUF dot product", 5.0e-4F);
}

void TestModelRows(const std::filesystem::path& model_path) {
  std::string error;
  const auto reader = strix::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "Assertion failed: MTP GGUF opens: " << error << '\n';
    std::exit(1);
  }

  constexpr std::array<ModelSample, 16> kQ4Samples{{
      {0, 0.008269786834716797F},
      {1, -0.019530773162841797F},
      {15, 0.023714542388916016F},
      {31, -0.01335287094116211F},
      {32, 0.009994983673095703F},
      {63, 0.005248546600341797F},
      {64, -0.002658843994140625F},
      {127, 0.013089179992675781F},
      {128, -0.006126880645751953F},
      {255, -0.0026617050170898438F},
      {256, -0.002628028392791748F},
      {511, 0.013987362384796143F},
      {1023, -0.0033451318740844727F},
      {2047, 0.0010213851928710938F},
      {4095, -0.00015413761138916016F},
      {5119, -0.0033974647521972656F},
  }};
  TestModelRow(*reader, "blk.64.nextn.eh_proj.weight",
               strix::core::GgmlType::kQ4_K, 10240, kQ4Samples,
               1.1511284159496427);

  constexpr std::array<ModelSample, 16> kQ6Samples{{
      {0, -0.0027894973754882812F},
      {1, -0.013947486877441406F},
      {15, -0.024175643920898438F},
      {31, 0.009633064270019531F},
      {32, -0.003979682922363281F},
      {63, 0.008864402770996094F},
      {64, 0.0030994415283203125F},
      {127, -0.006855964660644531F},
      {128, -0.008070945739746094F},
      {255, 0.011157989501953125F},
      {256, -0.03636360168457031F},
      {511, 0.024633407592773438F},
      {1023, -0.0008414983749389648F},
      {2047, -0.012941122055053711F},
      {4095, 0.02767181396484375F},
      {5119, -0.012220144271850586F},
  }};
  TestModelRow(*reader, "blk.64.attn_q.weight", strix::core::GgmlType::kQ6_K,
               5120, kQ6Samples, 0.5780322588980198);
}

}  // namespace

int main() {
  TestQ4();
  TestQ6();
  TestQ3();
  if (const char* model = std::getenv("STRIX_MTP_MODEL");
      model != nullptr && std::string_view(model).size() > 0) {
    TestModelRows(model);
  }
  return 0;
}
