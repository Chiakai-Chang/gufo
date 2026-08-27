#ifndef GUFO_SERVER_CONTINUATION_CACHE_HPP_
#define GUFO_SERVER_CONTINUATION_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace gufo::server {

using ContinuationToken = std::uint32_t;

/// Model-private continuation state retained by the common serving cache.
///
/// Implementations own all attention, recurrent, position, and graph-bound
/// state. The cache deliberately cannot inspect or copy those bytes.
class ContinuationState {
public:
  ContinuationState() = default;
  virtual ~ContinuationState() = default;

  ContinuationState(const ContinuationState&) = delete;
  ContinuationState& operator=(const ContinuationState&) = delete;
  ContinuationState(ContinuationState&&) = delete;
  ContinuationState& operator=(ContinuationState&&) = delete;

  /// Discards any partially or fully computed continuation.
  virtual void Invalidate() noexcept = 0;
};

/// Bounded exact-prefix cache over opaque model continuation states.
///
/// The first deployment uses one entry. Supporting a bounded entry count here
/// keeps cache policy independent from Qwen, DeepSeek, and future providers.
class ContinuationCache {
public:
  using StateFactory = std::function<std::unique_ptr<ContinuationState>()>;
  using CancellationCheck = std::function<bool()>;

  class Lease {
  public:
    Lease() = default;
    ~Lease();

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
      return cache_ != nullptr;
    }
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept { return cache_hit_; }
    [[nodiscard]] std::size_t cached_tokens() const noexcept {
      return cached_tokens_;
    }

    /// Atomically publishes the state and the exact tokens it represents.
    void Commit(std::vector<ContinuationToken> tokens);

    /// Explicitly discards partial state. The destructor does the same if a
    /// lease is not committed.
    void Invalidate() noexcept;

  private:
    friend class ContinuationCache;

    Lease(ContinuationCache* cache, std::size_t index, bool cache_hit,
          std::size_t cached_tokens) noexcept;

    ContinuationCache* cache_{nullptr};
    std::size_t index_{0};
    bool cache_hit_{false};
    std::size_t cached_tokens_{0};
  };

  ContinuationCache(std::size_t capacity, const StateFactory& factory);
  ~ContinuationCache();

  ContinuationCache(const ContinuationCache&) = delete;
  ContinuationCache& operator=(const ContinuationCache&) = delete;
  ContinuationCache(ContinuationCache&&) = delete;
  ContinuationCache& operator=(ContinuationCache&&) = delete;

  [[nodiscard]] Lease Acquire(std::span<const ContinuationToken> prompt,
                              const CancellationCheck& is_cancelled = {});

  [[nodiscard]] std::size_t capacity() const noexcept;

private:
  struct Entry;

  [[nodiscard]] ContinuationState& StateAt(std::size_t index);
  void Commit(std::size_t index, std::vector<ContinuationToken> tokens);
  void Invalidate(std::size_t index) noexcept;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CONTINUATION_CACHE_HPP_
