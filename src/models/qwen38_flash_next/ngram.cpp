#include "src/models/qwen38_flash_next/ngram.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <unordered_map>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::size_t kPage = 4096;
/// Synchronous preads at a queue depth of one each: a 512-token chunk
/// gathers 8192 random rows, 62 ms on 8 workers and 27 ms on 32.
constexpr std::size_t kWorkers = 32;

}  // namespace

void HashNgramRows(const Config& c, NgramHistory& history,
                   std::span<const std::int32_t> tokens,
                   std::span<std::uint32_t> rows) {
  const std::uint32_t n = c.ple_ngram_size;
  const std::int64_t eos = c.ple_eos_token;
  std::array<std::uint64_t, Config::kMaxPleNgram> ctx{};
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    // The token's own EOS does not cut its context, only an older one does.
    ctx[0] = static_cast<std::uint64_t>(tokens[i]);
    bool cut = false;
    for (std::uint32_t s = 1; s < n; ++s) {
      const std::int32_t t = cut ? NgramHistory::kNone : history.prev[s - 1];
      cut = cut || t < 0 || t == eos;
      ctx[s] = static_cast<std::uint64_t>(cut ? eos : t);
    }
    std::uint32_t* out = rows.data() + i * c.ple_heads;
    for (std::uint32_t order = 2; order <= n; ++order) {
      std::uint64_t mixed = ctx[0] * c.ple_multipliers[0];
      for (std::uint32_t j = 1; j < order; ++j) {
        mixed ^= ctx[j] * c.ple_multipliers[j];
      }
      const std::uint32_t base = (order - 2) * c.ple_heads_per_ngram;
      for (std::uint32_t g = 0; g < c.ple_heads_per_ngram; ++g) {
        const std::uint32_t h = base + g;
        out[h] = static_cast<std::uint32_t>(mixed % c.ple_head_vocab[h]) +
                 c.ple_head_offsets[h];
      }
    }
    for (std::size_t s = history.prev.size(); s-- > 1;) {
      history.prev[s] = history.prev[s - 1];
    }
    history.prev[0] = tokens[i];
  }
}

NgramTable::~NgramTable() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& w : workers_) {
    w.join();
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

std::unique_ptr<NgramTable> NgramTable::Open(const std::filesystem::path& path,
                                             std::uint64_t file_offset,
                                             std::uint64_t rows,
                                             std::uint32_t row_dim,
                                             core::GgmlType type,
                                             std::string* error_msg) {
  std::unique_ptr<NgramTable> t(new NgramTable());
  const std::size_t block = type == core::GgmlType::kIQ4_NL ? 32 : 1;
  const std::size_t block_bytes = type == core::GgmlType::kIQ4_NL ? 18 : 2;
  if ((type != core::GgmlType::kIQ4_NL && type != core::GgmlType::kBF16) ||
      row_dim == 0 || row_dim % block != 0) {
    if (error_msg != nullptr) {
      *error_msg = "unsupported n-gram table format";
    }
    return nullptr;
  }
  t->type_ = type;
  t->row_dim_ = row_dim;
  t->row_bytes_ = row_dim / block * block_bytes;
  t->rows_ = rows;
  t->base_offset_ = file_offset;
  // Direct I/O bypasses the page cache; the mapping used for the rest of the
  // model must not be used here or every touched row would stay resident.
  t->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  t->direct_ = t->fd_ >= 0;
  if (t->fd_ < 0) {
    t->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  }
  if (t->fd_ < 0) {
    if (error_msg != nullptr) {
      *error_msg = "cannot open n-gram table " + path.string() + ": " +
                   std::strerror(errno);
    }
    return nullptr;
  }
  const std::size_t workers = std::min(
      kWorkers, static_cast<std::size_t>(std::thread::hardware_concurrency()));
  for (std::size_t i = 0; i < std::max<std::size_t>(1, workers); ++i) {
    t->workers_.emplace_back([raw = t.get()] { raw->Worker(); });
  }
  return t;
}

bool NgramTable::ReadOne(std::uint32_t row, float* dst,
                         std::vector<std::uint8_t>& buf) {
  if (row >= rows_) {
    return false;
  }
  const std::uint64_t offset = base_offset_ + row * row_bytes_;
  const std::uint64_t begin = direct_ ? offset & ~(kPage - 1) : offset;
  const std::uint64_t end =
      direct_ ? (offset + row_bytes_ + kPage - 1) & ~(kPage - 1)
              : offset + row_bytes_;
  const std::size_t length = end - begin;
  if (buf.size() < length + kPage) {
    buf.resize(length + kPage);
  }
  // O_DIRECT needs a page-aligned buffer; align inside the vector.
  auto* base =
      reinterpret_cast<std::uintptr_t>(buf.data()) % kPage == 0
          ? buf.data()
          : buf.data() +
                (kPage - reinterpret_cast<std::uintptr_t>(buf.data()) % kPage);
  // The aligned window may run past the end of the file: only the row's own
  // bytes have to arrive.
  const std::size_t needed = (offset - begin) + row_bytes_;
  std::size_t got = 0;
  while (got < needed) {
    const ssize_t n = ::pread(fd_, base + got, length - got, begin + got);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) {
        continue;
      }
      return false;
    }
    got += static_cast<std::size_t>(n);
  }
  const std::uint8_t* src = base + (offset - begin);
  if (type_ == core::GgmlType::kIQ4_NL) {
    gufo::quant::DequantizeIQ4_NL(src, dst, row_dim_);
  } else {
    for (std::uint32_t i = 0; i < row_dim_; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, src + 2 * i, 2);
      const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
      std::memcpy(dst + i, &f, sizeof(float));
    }
  }
  return true;
}

void NgramTable::Worker() {
  std::vector<std::uint8_t> buf;
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || next_job_ < jobs_.size(); });
    if (stop_) {
      return;
    }
    const Job job = jobs_[next_job_++];
    lock.unlock();
    const bool ok = ReadOne(job.row, job.dst, buf);
    lock.lock();
    failed_ = failed_ || !ok;
    if (--pending_ == 0) {
      done_.notify_all();
    }
  }
}

bool NgramTable::Read(std::span<const std::uint32_t> rows,
                      std::span<float> out) {
  if (out.size() < rows.size() * row_dim_) {
    return false;
  }
  // Read each distinct row once, then copy it to its other slots.
  std::unordered_map<std::uint32_t, std::size_t> first;
  first.reserve(rows.size());
  std::vector<std::pair<std::size_t, std::size_t>> copies;
  std::vector<Job> jobs;
  jobs.reserve(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto [it, inserted] = first.emplace(rows[i], i);
    if (inserted) {
      jobs.push_back({rows[i], out.data() + i * row_dim_});
    } else {
      copies.emplace_back(it->second, i);
    }
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    jobs_ = std::move(jobs);
    next_job_ = 0;
    pending_ = jobs_.size();
    failed_ = false;
    wake_.notify_all();
    done_.wait(lock, [&] { return pending_ == 0; });
    jobs_.clear();
    if (failed_) {
      return false;
    }
  }
  for (const auto& [src, dst] : copies) {
    std::copy_n(out.data() + src * row_dim_, row_dim_,
                out.data() + dst * row_dim_);
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
