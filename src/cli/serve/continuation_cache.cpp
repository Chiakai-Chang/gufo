#include "src/cli/serve/continuation_cache.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace strix::server {

namespace {

constexpr auto kCancellationPollInterval = std::chrono::milliseconds{10};

bool IsPrefix(std::span<const ContinuationToken> prefix,
              std::span<const ContinuationToken> tokens) {
  return prefix.size() <= tokens.size() &&
         std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

}  // namespace

struct ContinuationCache::Entry {
  explicit Entry(std::unique_ptr<ContinuationState> model_state)
      : state(std::move(model_state)) {}

  std::unique_ptr<ContinuationState> state;
  std::vector<ContinuationToken> tokens;
  std::uint64_t last_used{0};
  bool available{true};
  bool valid{false};
  bool dirty{false};
};

struct ContinuationCache::Impl {
  std::vector<std::unique_ptr<Entry>> entries;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::uint64_t clock{0};
};

ContinuationCache::Lease::Lease(ContinuationCache* cache, std::size_t index,
                                bool cache_hit,
                                std::size_t cached_tokens) noexcept
    : cache_(cache),
      index_(index),
      cache_hit_(cache_hit),
      cached_tokens_(cached_tokens) {}

ContinuationCache::Lease::~Lease() {
  Invalidate();
}

ContinuationCache::Lease::Lease(Lease&& other) noexcept
    : cache_(std::exchange(other.cache_, nullptr)),
      index_(std::exchange(other.index_, 0)),
      cache_hit_(std::exchange(other.cache_hit_, false)),
      cached_tokens_(std::exchange(other.cached_tokens_, 0)) {}

ContinuationCache::Lease& ContinuationCache::Lease::operator=(
    Lease&& other) noexcept {
  if (this != &other) {
    Invalidate();
    cache_ = std::exchange(other.cache_, nullptr);
    index_ = std::exchange(other.index_, 0);
    cache_hit_ = std::exchange(other.cache_hit_, false);
    cached_tokens_ = std::exchange(other.cached_tokens_, 0);
  }
  return *this;
}

ContinuationState& ContinuationCache::Lease::state() const {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  return cache_->StateAt(index_);
}

void ContinuationCache::Lease::Commit(std::vector<ContinuationToken> tokens) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  cache_->Commit(index_, std::move(tokens));
  cache_ = nullptr;
}

void ContinuationCache::Lease::Invalidate() noexcept {
  if (cache_ != nullptr) {
    cache_->Invalidate(index_);
    cache_ = nullptr;
  }
}

ContinuationCache::ContinuationCache(std::size_t capacity,
                                     const StateFactory& factory)
    : impl_(std::make_unique<Impl>()) {
  if (capacity == 0) {
    throw std::invalid_argument(
        "continuation cache capacity must be at least one");
  }
  if (!factory) {
    throw std::invalid_argument(
        "continuation cache state factory must be callable");
  }

  impl_->entries.reserve(capacity);
  for (std::size_t index = 0; index < capacity; ++index) {
    auto state = factory();
    if (state == nullptr) {
      throw std::runtime_error(
          "continuation cache state factory returned null");
    }
    impl_->entries.push_back(std::make_unique<Entry>(std::move(state)));
  }
}

ContinuationCache::~ContinuationCache() = default;

ContinuationCache::Lease ContinuationCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const CancellationCheck& is_cancelled) {
  while (true) {
    std::unique_lock<std::mutex> lock(impl_->mutex);

    std::size_t selected = impl_->entries.size();
    std::size_t cached_tokens = 0;
    for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
      const auto& entry = *impl_->entries[index];
      if (!entry.available || !entry.valid || !IsPrefix(entry.tokens, prompt)) {
        continue;
      }
      if (selected == impl_->entries.size() ||
          entry.tokens.size() > cached_tokens) {
        selected = index;
        cached_tokens = entry.tokens.size();
      }
    }

    const bool cache_hit = selected != impl_->entries.size();
    if (!cache_hit) {
      std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto& entry = *impl_->entries[index];
        if (entry.available && entry.last_used < oldest) {
          selected = index;
          oldest = entry.last_used;
        }
      }
    }

    if (selected != impl_->entries.size()) {
      auto& entry = *impl_->entries[selected];
      entry.available = false;
      const bool needs_invalidation = entry.dirty && !cache_hit;
      entry.valid = false;
      entry.tokens.clear();
      entry.dirty = true;
      entry.last_used = ++impl_->clock;
      lock.unlock();

      if (needs_invalidation) {
        entry.state->Invalidate();
      }
      return Lease(this, selected, cache_hit, cached_tokens);
    }

    lock.unlock();
    if (is_cancelled && is_cancelled()) {
      return {};
    }
    lock.lock();
    impl_->condition.wait_for(lock, kCancellationPollInterval);
  }
}

std::size_t ContinuationCache::capacity() const noexcept {
  return impl_->entries.size();
}

ContinuationState& ContinuationCache::StateAt(std::size_t index) {
  return *impl_->entries.at(index)->state;
}

void ContinuationCache::Commit(std::size_t index,
                               std::vector<ContinuationToken> tokens) {
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& entry = *impl_->entries.at(index);
    entry.tokens = std::move(tokens);
    entry.valid = !entry.tokens.empty();
    entry.dirty = true;
    entry.available = true;
    entry.last_used = ++impl_->clock;
  }
  impl_->condition.notify_one();
}

void ContinuationCache::Invalidate(std::size_t index) noexcept {
  auto& entry = *impl_->entries[index];
  if (entry.dirty) {
    entry.state->Invalidate();
  }
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    entry.tokens.clear();
    entry.valid = false;
    entry.dirty = false;
    entry.available = true;
    entry.last_used = ++impl_->clock;
  }
  impl_->condition.notify_one();
}

}  // namespace strix::server
