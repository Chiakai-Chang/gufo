#include "src/core/gguf_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace strix::core {

namespace {

constexpr std::array<char, 4> kGgufMagic = {'G', 'G', 'U', 'F'};

template<typename T>
bool ReadPod(const std::uint8_t* data, std::size_t size, std::size_t& offset,
             T& out) {
  if (offset + sizeof(T) > size) {
    return false;
  }
  std::memcpy(&out, data + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

bool ReadString(const std::uint8_t* data, std::size_t size, std::size_t& offset,
                std::string_view& out) {
  std::uint64_t len = 0;
  if (!ReadPod(data, size, offset, len)) {
    return false;
  }
  if (offset + len > size) {
    return false;
  }
  out = std::string_view(reinterpret_cast<const char*>(data + offset), len);
  offset += len;
  return true;
}

}  // namespace

GgufReader::~GgufReader() {
  if (owns_mmap_ && mmap_addr_ != nullptr && size_ > 0) {
    munmap(mmap_addr_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

GgufReader::GgufReader(GgufReader&& other) noexcept
    : data_(other.data_),
      mmap_addr_(other.mmap_addr_),
      size_(other.size_),
      fd_(other.fd_),
      owns_mmap_(other.owns_mmap_),
      version_(other.version_),
      alignment_(other.alignment_),
      metadata_(std::move(other.metadata_)),
      tensors_(std::move(other.tensors_)),
      tensor_index_(std::move(other.tensor_index_)) {
  other.data_ = nullptr;
  other.mmap_addr_ = nullptr;
  other.size_ = 0;
  other.fd_ = -1;
  other.owns_mmap_ = false;
}

GgufReader& GgufReader::operator=(GgufReader&& other) noexcept {
  if (this != &other) {
    if (owns_mmap_ && mmap_addr_ != nullptr && size_ > 0) {
      munmap(mmap_addr_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }

    data_ = other.data_;
    mmap_addr_ = other.mmap_addr_;
    size_ = other.size_;
    fd_ = other.fd_;
    owns_mmap_ = other.owns_mmap_;
    version_ = other.version_;
    alignment_ = other.alignment_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);

    other.data_ = nullptr;
    other.mmap_addr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
    other.owns_mmap_ = false;
  }
  return *this;
}

std::unique_ptr<GgufReader> GgufReader::OpenFile(
    const std::filesystem::path& path, std::string* error_msg) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to open file: " + path.string();
    }
    return nullptr;
  }

  struct stat sb{};
  if (fstat(fd, &sb) != 0 || sb.st_size <= 0) {
    close(fd);
    if (error_msg != nullptr) {
      *error_msg = "Invalid or empty file: " + path.string();
    }
    return nullptr;
  }

  const auto size = static_cast<std::size_t>(sb.st_size);
  void* const addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    close(fd);
    if (error_msg != nullptr) {
      *error_msg = "Failed to mmap file: " + path.string();
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->mmap_addr_ = addr;
  reader->data_ = static_cast<const std::uint8_t*>(addr);
  reader->size_ = size;
  reader->fd_ = fd;
  reader->owns_mmap_ = true;

  if (!reader->ParseHeaders(error_msg)) {
    return nullptr;
  }
  return reader;
}

std::unique_ptr<GgufReader> GgufReader::OpenMemory(const void* data,
                                                   std::size_t size,
                                                   std::string* error_msg) {
  if (data == nullptr || size < 24) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid memory buffer for GGUF";
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->mmap_addr_ = nullptr;
  reader->data_ = static_cast<const std::uint8_t*>(data);
  reader->size_ = size;
  reader->fd_ = -1;
  reader->owns_mmap_ = false;

  if (!reader->ParseHeaders(error_msg)) {
    return nullptr;
  }
  return reader;
}

bool GgufReader::ParseHeaders(std::string* error_msg) {
  if (size_ < 24) {
    if (error_msg != nullptr) {
      *error_msg = "File too small for GGUF header";
    }
    return false;
  }

  if (std::memcmp(data_, kGgufMagic.data(), 4) != 0) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid GGUF magic header";
    }
    return false;
  }

  std::size_t offset = 4;
  if (!ReadPod(data_, size_, offset, version_)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read GGUF version";
    }
    return false;
  }

  if (version_ != 2 && version_ != 3) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported GGUF version: " + std::to_string(version_);
    }
    return false;
  }

  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  if (!ReadPod(data_, size_, offset, tensor_count) ||
      !ReadPod(data_, size_, offset, metadata_count)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read tensor/metadata counts";
    }
    return false;
  }

  metadata_.reserve(metadata_count);

  // Parse metadata key-value pairs
  for (std::uint64_t i = 0; i < metadata_count; ++i) {
    std::string_view key;
    if (!ReadString(data_, size_, offset, key)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata key";
      }
      return false;
    }

    std::uint32_t val_type_raw = 0;
    if (!ReadPod(data_, size_, offset, val_type_raw)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata type for key " + std::string(key);
      }
      return false;
    }

    const auto val_type = static_cast<GgufValueType>(val_type_raw);
    GgufMetadataValue meta_val;
    meta_val.type = val_type;

    switch (val_type) {
      case GgufValueType::kUint8: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt8: {
        std::int8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint16: {
        std::uint16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt16: {
        std::int16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint32: {
        std::uint32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt32: {
        std::int32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kFloat32: {
        float v = 0.0F;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<double>(v);
        break;
      }
      case GgufValueType::kBool: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = (v != 0);
        break;
      }
      case GgufValueType::kString: {
        std::string_view s;
        if (!ReadString(data_, size_, offset, s)) {
          return false;
        }
        meta_val.value = s;
        break;
      }
      case GgufValueType::kUint64: {
        std::uint64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kInt64: {
        std::int64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kFloat64: {
        double v = 0.0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kArray: {
        std::uint32_t item_type_raw = 0;
        std::uint64_t array_len = 0;
        if (!ReadPod(data_, size_, offset, item_type_raw) ||
            !ReadPod(data_, size_, offset, array_len)) {
          return false;
        }
        const auto item_type = static_cast<GgufValueType>(item_type_raw);
        if (item_type == GgufValueType::kString) {
          std::vector<std::string_view> str_arr;
          str_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::string_view s;
            if (!ReadString(data_, size_, offset, s)) {
              return false;
            }
            str_arr.push_back(s);
          }
          meta_val.value = str_arr;
        } else if (item_type == GgufValueType::kUint32 ||
                   item_type == GgufValueType::kUint64) {
          std::vector<std::uint64_t> u_arr;
          u_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::uint64_t val = 0;
            if (item_type == GgufValueType::kUint32) {
              std::uint32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else {
              if (!ReadPod(data_, size_, offset, val)) {
                return false;
              }
            }
            u_arr.push_back(val);
          }
          meta_val.value = u_arr;
        } else {
          // Skip other array types cleanly
          std::size_t item_size = 4;
          if (item_type == GgufValueType::kUint8 ||
              item_type == GgufValueType::kInt8 ||
              item_type == GgufValueType::kBool) {
            item_size = 1;
          } else if (item_type == GgufValueType::kUint16 ||
                     item_type == GgufValueType::kInt16) {
            item_size = 2;
          } else if (item_type == GgufValueType::kUint64 ||
                     item_type == GgufValueType::kInt64 ||
                     item_type == GgufValueType::kFloat64) {
            item_size = 8;
          }
          if (offset + (array_len * item_size) > size_) {
            return false;
          }
          offset += (array_len * item_size);
        }
        break;
      }
    }

    if (key == "general.alignment") {
      if (std::holds_alternative<std::uint64_t>(meta_val.value)) {
        alignment_ = std::get<std::uint64_t>(meta_val.value);
      }
    }
    metadata_.emplace(key, std::move(meta_val));
  }

  // Parse tensor infos
  tensors_.reserve(tensor_count);
  tensor_index_.reserve(tensor_count);

  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo info;
    if (!ReadString(data_, size_, offset, info.name)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read tensor name";
      }
      return false;
    }

    std::uint32_t n_dims = 0;
    if (!ReadPod(data_, size_, offset, n_dims) || n_dims > 8) {
      if (error_msg != nullptr) {
        *error_msg = "Invalid tensor dimensions count";
      }
      return false;
    }

    info.dimensions.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadPod(data_, size_, offset, info.dimensions[d])) {
        return false;
      }
    }

    std::uint32_t type_raw = 0;
    if (!ReadPod(data_, size_, offset, type_raw)) {
      return false;
    }
    info.type = static_cast<GgmlType>(type_raw);

    if (!ReadPod(data_, size_, offset, info.offset)) {
      return false;
    }

    tensor_index_[info.name] = tensors_.size();
    tensors_.push_back(std::move(info));
  }

  // Align data payload base to alignment boundary
  const std::size_t data_base = (offset + alignment_ - 1) & ~(alignment_ - 1);
  for (auto& tensor : tensors_) {
    if (data_base + tensor.offset > size_) {
      if (error_msg != nullptr) {
        *error_msg =
            "Tensor offset exceeds file size: " + std::string(tensor.name);
      }
      return false;
    }
    tensor.data = data_ + data_base + tensor.offset;
  }

  return true;
}

const GgufMetadataValue* GgufReader::FindMetadata(
    std::string_view key) const noexcept {
  auto it = metadata_.find(key);
  if (it != metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::optional<std::string_view> GgufReader::GetMetadataString(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::string_view>(meta->value)) {
    return std::get<std::string_view>(meta->value);
  }
  return std::nullopt;
}

std::optional<std::uint32_t> GgufReader::GetMetadataUint32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<std::uint64_t>(meta->value)) {
    return static_cast<std::uint32_t>(std::get<std::uint64_t>(meta->value));
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GgufReader::GetMetadataUint64(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<std::uint64_t>(meta->value)) {
    return std::get<std::uint64_t>(meta->value);
  }
  return std::nullopt;
}

std::optional<float> GgufReader::GetMetadataFloat32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<double>(meta->value)) {
    return static_cast<float>(std::get<double>(meta->value));
  }
  return std::nullopt;
}

std::optional<bool> GgufReader::GetMetadataBool(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<bool>(meta->value)) {
    return std::get<bool>(meta->value);
  }
  return std::nullopt;
}

std::vector<std::string_view> GgufReader::GetMetadataStringArray(
    std::string_view key) const {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::vector<std::string_view>>(meta->value)) {
    return std::get<std::vector<std::string_view>>(meta->value);
  }
  return {};
}

const GgufTensorInfo* GgufReader::FindTensor(
    std::string_view name) const noexcept {
  auto it = tensor_index_.find(name);
  if (it != tensor_index_.end()) {
    return &tensors_[it->second];
  }
  return nullptr;
}

bool GgufReader::HasTensor(std::string_view name) const noexcept {
  return tensor_index_.contains(name);
}

bool GgufReader::HasMtpTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("mtp.") ||
           t.name.find(".mtp.") != std::string_view::npos;
  });
}

bool GgufReader::HasVisionTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("model.visual") ||
           t.name.starts_with("visual.") || t.name.starts_with("v.");
  });
}

std::optional<ModelConfig> GgufReader::ExtractModelConfig(
    std::string* error_msg) const {
  ModelConfig config;

  // 1. Check architecture prefix (e.g. "qwen2", "qwen3", "qwen35", "llama")
  auto arch_str = GetMetadataString("general.architecture").value_or("qwen");
  config.architecture = std::string(arch_str);
  const std::string prefix = config.architecture + ".";

  // 2. Read standard architectural hyperparameters
  if (auto val = GetMetadataUint32(prefix + "block_count")) {
    config.num_layers = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "embedding_length")) {
    config.hidden_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "feed_forward_length")) {
    config.intermediate_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.head_count")) {
    config.num_attention_heads = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.head_count_kv")) {
    config.num_key_value_heads = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.key_length")) {
    config.head_dim = *val;
  } else if (config.num_attention_heads > 0) {
    config.head_dim = config.hidden_size / config.num_attention_heads;
  }
  if (auto val = GetMetadataUint32(prefix + "context_length")) {
    config.context_length = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "full_attention_interval")) {
    config.full_attention_interval = *val;
  }
  if (auto val = GetMetadataFloat32(prefix + "rope.freq_base")) {
    config.rope_theta = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "rope.dimension_count")) {
    config.rotary_dim = *val;
  } else {
    config.rotary_dim = config.head_dim;
  }

  // 3. MTP speculative layers check
  config.mtp_num_layers = HasMtpTensors() ? 1 : 0;

  // 4. Assert text-only contract
  if (HasVisionTensors()) {
    config.is_text_only = false;
    if (error_msg != nullptr) {
      *error_msg =
          "Model contains vision encoder tensors; text-only contract violation";
    }
    return std::nullopt;
  }
  config.is_text_only = true;

  // 5. Model name
  if (auto name = GetMetadataString("general.name")) {
    config.model_name = std::string(*name);
  }

  // 6. Validate configuration against Qwen3.5/3.8 structural rules
  if (!config.IsValidQwen()) {
    if (error_msg != nullptr) {
      *error_msg = "Model config failed Qwen structural validation (head_dim=" +
                   std::to_string(config.head_dim) +
                   ", layers=" + std::to_string(config.num_layers) + ")";
    }
    return std::nullopt;
  }

  return config;
}

}  // namespace strix::core
