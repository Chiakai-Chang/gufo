#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_WEIGHT_UPLOAD_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_WEIGHT_UPLOAD_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace gufo::models::qwen38_flash_next::rocm {

/// Bounded disk-to-device pipeline. Copy queues a tensor without waiting for
/// its upload; destinations must remain alive until Finish or destruction.
/// The n-gram table and mapped GGUF payloads are never touched.
class WeightUpload {
public:
  static std::unique_ptr<WeightUpload> Create(
      std::span<const std::filesystem::path> shards, std::string* error);
  ~WeightUpload();
  WeightUpload(const WeightUpload&) = delete;
  WeightUpload& operator=(const WeightUpload&) = delete;

  bool Copy(std::uint32_t shard, std::uint64_t offset, std::size_t size,
            void* device, std::string* error);
  /// Waits for every queued copy, including on failure.
  bool Finish(std::string* error);

private:
  struct State;
  WeightUpload();
  std::unique_ptr<State> state_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif
