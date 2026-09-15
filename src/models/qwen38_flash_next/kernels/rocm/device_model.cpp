#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <initializer_list>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

constexpr std::size_t kStageBytes = 64ULL << 20;
constexpr std::size_t kStageCount = 3;
constexpr std::size_t kDirectAlign = 4096;

/// Streams tensor payloads from the shard files straight into device memory:
/// direct reads into pinned staging buffers, asynchronous copies behind
/// them. The mapped GGUF is never touched, so the page cache stays empty and
/// the copy runs at disk speed instead of page-fault speed.
class Stager {
public:
  Stager(std::vector<std::filesystem::path> shards, std::string* error)
      : error_(error) {
    for (const auto& path : shards) {
      int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
      direct_.push_back(fd >= 0);
      if (fd < 0) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
      }
      if (fd < 0) {
        Fail("cannot open shard " + path.string() + ": " + std::strerror(errno));
      }
      fds_.push_back(fd);
    }
    if (hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess) {
      Fail("upload stream creation failed");
    }
    for (std::size_t i = 0; i < kStageCount; ++i) {
      void* raw = nullptr;
      if (hipHostMalloc(&raw, kStageBytes + kDirectAlign) != hipSuccess ||
          hipEventCreateWithFlags(&events_[i], hipEventDisableTiming) !=
              hipSuccess) {
        Fail("pinned staging allocation failed");
        break;
      }
      raw_[i] = raw;
      const auto address = reinterpret_cast<std::uintptr_t>(raw);
      stage_[i] = reinterpret_cast<std::uint8_t*>(
          (address + kDirectAlign - 1) & ~(kDirectAlign - 1));
      busy_[i] = false;
    }
  }

  ~Stager() {
    if (stream_ != nullptr) {
      (void)hipStreamSynchronize(stream_);
      (void)hipStreamDestroy(stream_);
    }
    for (std::size_t i = 0; i < kStageCount; ++i) {
      if (raw_[i] != nullptr) {
        (void)hipHostFree(raw_[i]);
      }
      if (events_[i] != nullptr) {
        (void)hipEventDestroy(events_[i]);
      }
    }
    for (int fd : fds_) {
      if (fd >= 0) {
        ::close(fd);
      }
    }
  }

  [[nodiscard]] bool ok() const noexcept { return ok_; }

  /// Copies `size` bytes at `offset` of shard `shard` to `device`.
  bool Copy(std::uint32_t shard, std::uint64_t offset, std::size_t size,
            void* device) {
    if (!ok_ || shard >= fds_.size()) {
      return false;
    }
    std::size_t done = 0;
    while (done < size) {
      const std::size_t i = next_++ % kStageCount;
      if (busy_[i] && hipEventSynchronize(events_[i]) != hipSuccess) {
        return Fail("staging event wait failed");
      }
      busy_[i] = false;
      // Direct I/O wants an aligned file offset and length: read the
      // enclosing aligned window and copy from the interior.
      const std::uint64_t at = offset + done;
      const std::uint64_t begin = direct_[shard] ? at & ~(kDirectAlign - 1) : at;
      const std::size_t skip = at - begin;
      std::size_t want = std::min(size - done, kStageBytes - skip);
      std::size_t length = skip + want;
      if (direct_[shard]) {
        length = (length + kDirectAlign - 1) & ~(kDirectAlign - 1);
      }
      std::size_t got = 0;
      while (got < skip + want) {
        const ssize_t n = ::pread(fds_[shard], stage_[i] + got, length - got,
                                  static_cast<off_t>(begin + got));
        if (n < 0 && errno == EINTR) {
          continue;
        }
        if (n <= 0) {
          return Fail("shard read failed: " +
                      std::string(n < 0 ? std::strerror(errno) : "short read"));
        }
        got += static_cast<std::size_t>(n);
      }
      if (hipMemcpyAsync(static_cast<std::uint8_t*>(device) + done,
                         stage_[i] + skip, want, hipMemcpyHostToDevice,
                         stream_) != hipSuccess ||
          hipEventRecord(events_[i], stream_) != hipSuccess) {
        return Fail("device copy failed");
      }
      busy_[i] = true;
      done += want;
    }
    return true;
  }

  bool Finish() {
    return hipStreamSynchronize(stream_) == hipSuccess;
  }

private:
  bool Fail(const std::string& message) {
    if (ok_ && error_ != nullptr) {
      *error_ = message;
    }
    ok_ = false;
    return false;
  }

  std::string* error_;
  bool ok_{true};
  std::vector<int> fds_;
  std::vector<bool> direct_;
  hipStream_t stream_{nullptr};
  void* raw_[kStageCount]{};
  std::uint8_t* stage_[kStageCount]{};
  hipEvent_t events_[kStageCount]{};
  bool busy_[kStageCount]{};
  std::size_t next_{0};
};

struct Uploader {
  Stager& stager;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(t.shard, t.file_offset, size, ptr)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin, nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // The staged copies land on the upload stream; convert behind them.
    if (!stager.Finish()) {
      Fail("upload failed for stacked tensor");
      return d;
    }
    NarrowActivations(static_cast<const float*>(ptr), half, false, count,
                      nullptr);
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    (void)hipDeviceSynchronize();
    // The F32 stack was pushed right after `half`; drop it.
    allocations.erase(allocations.end() - 2);
    (void)hipFree(ptr);
    bytes -= size + kTailMargin;
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    d.nextn_eh_proj = Copy(l.nextn_eh_proj);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const std::filesystem::path& model_path,
    const MtpWeights* mtp, const std::filesystem::path& mtp_path,
    std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  std::vector<std::filesystem::path> shards;
  // Every shard the trunk references; the sidecar is one more file.
  std::uint32_t shard_count = 0;
  for (const auto& l : w.layers) {
    shard_count = std::max(shard_count, l.ffn_down_exps.shard + 1);
    shard_count = std::max(shard_count, l.hc_attn.norm.shard + 1);
  }
  shard_count = std::max(shard_count, w.output.shard + 1);
  for (std::uint32_t i = 0; i < shard_count; ++i) {
    shards.push_back(ShardPath(model_path, i));
  }
  Stager stager(shards, error_msg);
  if (!stager.ok()) {
    return nullptr;
  }
  Uploader up{stager, m->allocations_, m->bytes_, m->max_half_cols_,
              m->max_q8_cols_, error_msg};
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ = w.output.data == w.token_embd.data ? m->token_embd_
                                                   : up.Copy(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    Stager mtp_stager({mtp_path}, error_msg);
    if (!mtp_stager.ok()) {
      return nullptr;
    }
    Uploader mtp_up{mtp_stager, m->allocations_, m->bytes_,
                    m->max_half_cols_, m->max_q8_cols_, error_msg};
    m->mtp_ = mtp_up.Layer(mtp->block);
    m->has_mtp_ = true;
    if (!mtp_up.ok || !mtp_stager.Finish()) {
      return nullptr;
    }
  }
  if (!up.ok || !stager.Finish()) {
    return nullptr;
  }
  (void)hipDeviceSynchronize();
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
