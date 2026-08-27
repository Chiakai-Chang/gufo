#include "src/cli/serve/continuation_cache.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace gufo::server {

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
  std::shared_ptr<const ContinuationSnapshot> snapshot;
  std::vector<ContinuationToken> tokens;
  std::uint64_t state_last_used{0};
  std::uint64_t snapshot_last_used{0};
  bool available{true};
  bool valid{false};
  bool dirty{false};
};

struct ContinuationCache::Impl {
  std::vector<std::unique_ptr<Entry>> entries;
  SnapshotSupport snapshot_support;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::uint64_t clock{0};

  [[nodiscard]] bool snapshot_mode() const noexcept {
    return static_cast<bool>(snapshot_support.restore);
  }
};

ContinuationCache::Lease::Lease(ContinuationCache* cache, std::size_t index,
                                bool cache_hit, std::size_t cached_tokens,
                                std::size_t source_index,
                                std::size_t restored_snapshot_bytes,
                                double restore_ms) noexcept
    : cache_(cache),
      index_(index),
      cache_hit_(cache_hit),
      cached_tokens_(cached_tokens),
      source_index_(source_index),
      restored_snapshot_bytes_(restored_snapshot_bytes),
      restore_ms_(restore_ms) {}

ContinuationCache::Lease::~Lease() {
  Invalidate();
}

ContinuationCache::Lease::Lease(Lease&& other) noexcept
    : cache_(std::exchange(other.cache_, nullptr)),
      index_(std::exchange(other.index_, 0)),
      cache_hit_(std::exchange(other.cache_hit_, false)),
      cached_tokens_(std::exchange(other.cached_tokens_, 0)),
      source_index_(std::exchange(other.source_index_, 0)),
      restored_snapshot_bytes_(
          std::exchange(other.restored_snapshot_bytes_, 0)),
      restore_ms_(std::exchange(other.restore_ms_, 0.0)) {}

ContinuationCache::Lease& ContinuationCache::Lease::operator=(
    Lease&& other) noexcept {
  if (this != &other) {
    Invalidate();
    cache_ = std::exchange(other.cache_, nullptr);
    index_ = std::exchange(other.index_, 0);
    cache_hit_ = std::exchange(other.cache_hit_, false);
    cached_tokens_ = std::exchange(other.cached_tokens_, 0);
    source_index_ = std::exchange(other.source_index_, 0);
    restored_snapshot_bytes_ = std::exchange(other.restored_snapshot_bytes_, 0);
    restore_ms_ = std::exchange(other.restore_ms_, 0.0);
  }
  return *this;
}

ContinuationState& ContinuationCache::Lease::state() const {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  return cache_->StateAt(index_);
}

void ContinuationCache::Lease::Commit(
    std::vector<ContinuationToken> tokens,
    std::unique_ptr<ContinuationSnapshot> snapshot) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  cache_->Commit(index_, source_index_, std::move(tokens), std::move(snapshot));
  cache_ = nullptr;
}

void ContinuationCache::Lease::Invalidate() noexcept {
  if (cache_ != nullptr) {
    cache_->Invalidate(index_);
    cache_ = nullptr;
  }
}

ContinuationCache::ContinuationCache(std::size_t capacity,
                                     const StateFactory& factory,
                                     SnapshotSupport snapshot_support)
    : impl_(std::make_unique<Impl>()) {
  if (capacity == 0) {
    throw std::invalid_argument(
        "continuation cache capacity must be at least one");
  }
  if (!factory) {
    throw std::invalid_argument(
        "continuation cache state factory must be callable");
  }
  impl_->snapshot_support = std::move(snapshot_support);

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

    const std::size_t no_entry = impl_->entries.size();
    std::size_t source = no_entry;
    std::size_t cached_tokens = 0;
    for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
      const auto& entry = *impl_->entries[index];
      if ((!impl_->snapshot_mode() && !entry.available) || !entry.valid ||
          !IsPrefix(entry.tokens, prompt)) {
        continue;
      }
      if (source == no_entry || entry.tokens.size() > cached_tokens) {
        source = index;
        cached_tokens = entry.tokens.size();
      }
    }

    const bool cache_hit = source != no_entry;
    std::size_t selected = source;
    if (impl_->snapshot_mode() || !cache_hit) {
      selected = no_entry;
      std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto& entry = *impl_->entries[index];
        if (entry.available && entry.state_last_used < oldest) {
          selected = index;
          oldest = entry.state_last_used;
        }
      }
    }

    if (selected != no_entry) {
      auto& entry = *impl_->entries[selected];
      entry.available = false;
      const bool needs_invalidation = entry.dirty && !cache_hit;
      entry.dirty = true;
      entry.state_last_used = ++impl_->clock;

      std::shared_ptr<const ContinuationSnapshot> snapshot;
      if (impl_->snapshot_mode()) {
        if (cache_hit) {
          auto& source_entry = *impl_->entries[source];
          snapshot = source_entry.snapshot;
          source_entry.snapshot_last_used = ++impl_->clock;
        }
      } else {
        entry.valid = false;
        entry.tokens.clear();
      }
      lock.unlock();

      std::size_t restored_snapshot_bytes = 0;
      double restore_ms = 0.0;
      try {
        if (cache_hit && impl_->snapshot_mode()) {
          if (snapshot == nullptr) {
            throw std::runtime_error(
                "continuation cache snapshot entry is empty");
          }
          restored_snapshot_bytes = snapshot->PayloadBytes();
          const auto restore_start = std::chrono::steady_clock::now();
          impl_->snapshot_support.restore(*entry.state, *snapshot);
          restore_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - restore_start)
                           .count();
        } else if (needs_invalidation) {
          entry.state->Invalidate();
        }
      } catch (...) {
        entry.state->Invalidate();
        {
          const std::lock_guard<std::mutex> failure_lock(impl_->mutex);
          entry.available = true;
          entry.dirty = false;
          entry.state_last_used = ++impl_->clock;
        }
        impl_->condition.notify_one();
        throw;
      }
      return Lease(this, selected, cache_hit, cached_tokens, source,
                   restored_snapshot_bytes, restore_ms);
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

void ContinuationCache::Commit(std::size_t index, std::size_t source_index,
                               std::vector<ContinuationToken> tokens,
                               std::unique_ptr<ContinuationSnapshot> snapshot) {
  if (impl_->snapshot_mode() && snapshot == nullptr) {
    throw std::invalid_argument(
        "snapshot continuation commit requires an immutable snapshot");
  }

  std::shared_ptr<const ContinuationSnapshot> retained_snapshot(
      std::move(snapshot));
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& state_entry = *impl_->entries.at(index);
    if (impl_->snapshot_mode()) {
      const std::size_t no_entry = impl_->entries.size();
      std::size_t target = no_entry;
      for (std::size_t candidate = 0; candidate < impl_->entries.size();
           ++candidate) {
        const auto& entry = *impl_->entries[candidate];
        if (entry.valid && entry.tokens == tokens) {
          target = candidate;
          break;
        }
      }
      if (target == no_entry) {
        for (std::size_t candidate = 0; candidate < impl_->entries.size();
             ++candidate) {
          if (!impl_->entries[candidate]->valid) {
            target = candidate;
            break;
          }
        }
      }
      if (target == no_entry) {
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t candidate = 0; candidate < impl_->entries.size();
             ++candidate) {
          if (impl_->entries.size() > 1 && candidate == source_index) {
            continue;
          }
          const auto& entry = *impl_->entries[candidate];
          if (entry.snapshot_last_used < oldest) {
            target = candidate;
            oldest = entry.snapshot_last_used;
          }
        }
      }
      if (target == no_entry) {
        target = source_index < impl_->entries.size() ? source_index : index;
      }

      auto& snapshot_entry = *impl_->entries[target];
      snapshot_entry.tokens = std::move(tokens);
      snapshot_entry.snapshot = std::move(retained_snapshot);
      snapshot_entry.valid = !snapshot_entry.tokens.empty();
      snapshot_entry.snapshot_last_used = ++impl_->clock;
    } else {
      state_entry.tokens = std::move(tokens);
      state_entry.valid = !state_entry.tokens.empty();
    }
    state_entry.dirty = true;
    state_entry.available = true;
    state_entry.state_last_used = ++impl_->clock;
  }
  impl_->condition.notify_all();
}

void ContinuationCache::Invalidate(std::size_t index) noexcept {
  auto& entry = *impl_->entries[index];
  entry.state->Invalidate();
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->snapshot_mode()) {
      entry.tokens.clear();
      entry.valid = false;
    }
    entry.dirty = false;
    entry.available = true;
    entry.state_last_used = ++impl_->clock;
  }
  impl_->condition.notify_one();
}

}  // namespace gufo::server
