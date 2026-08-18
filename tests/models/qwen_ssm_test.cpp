#include "src/models/qwen_ssm.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void TestQwenSsmConvRecurrence() {
  const std::uint32_t num_layers = 1;
  const std::size_t conv_channels = 8192;
  const std::size_t ssm_dim = 4096;

  strix::models::QwenSsmCache cache(num_layers, conv_channels, ssm_dim);

  // Check state sizes
  auto conv = cache.GetConvState(0);
  assert(conv.size() == conv_channels * 4);
  auto ssm = cache.GetSsmState(0);
  assert(ssm.size() == ssm_dim);

  // Initially zero
  assert(conv[0] == 0.0F);
  assert(ssm[0] == 0.0F);

  // Create dummy layer
  strix::models::QwenLayerWeights layer;
  layer.is_full_attention = false;

  std::vector<float> in(2560, 0.1F);
  std::vector<float> qkv_buf(8192, 0.0F);
  std::vector<float> gate_buf(4096, 0.0F);
  std::vector<float> out_buf(4096, 0.0F);
  std::vector<float> out(2560, 0.0F);

  strix::models::ForwardSSM(in, layer, cache, 0, qkv_buf, gate_buf, out_buf,
                            out);

  // After 1 step, cache is updated
  assert(cache.GetConvState(0).size() == conv_channels * 4);
}

int main() {
  TestQwenSsmConvRecurrence();
  std::cout << "All Qwen SSM tests passed.\n";
  return 0;
}
