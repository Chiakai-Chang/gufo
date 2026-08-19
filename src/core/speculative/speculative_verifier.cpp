#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace strix::speculative {

SpeculativeVerifier::SpeculativeVerifier(
    hip::QwenGpuExecutor& target_executor,
    std::unique_ptr<IDraftBackend> draft_backend, SpeculativeOptions options)
    : target_executor_(target_executor),
      draft_backend_(std::move(draft_backend)),
      options_(options),
      current_draft_length_(options_.initial_draft_tokens) {}

void SpeculativeVerifier::Reset() noexcept {
  stats_ = {};
  rolling_acceptance_.clear();
  current_draft_length_ = options_.initial_draft_tokens;
  if (draft_backend_ != nullptr) {
    draft_backend_->Reset();
  }
}

void SpeculativeVerifier::UpdateAdaptiveDraftLength(std::size_t accepted,
                                                    std::size_t drafted) {
  if (!options_.enable_adaptive_draft_length || drafted == 0) {
    return;
  }

  const float rate = static_cast<float>(accepted) / static_cast<float>(drafted);
  rolling_acceptance_.push_back(rate);
  if (rolling_acceptance_.size() > options_.rolling_window) {
    rolling_acceptance_.pop_front();
  }

  const float avg_rate = std::accumulate(rolling_acceptance_.begin(),
                                         rolling_acceptance_.end(), 0.0F) /
                         static_cast<float>(rolling_acceptance_.size());

  if (avg_rate > options_.target_acceptance_rate &&
      current_draft_length_ < options_.max_draft_tokens) {
    ++current_draft_length_;
  } else if (avg_rate < (options_.target_acceptance_rate * 0.5F) &&
             current_draft_length_ > options_.min_draft_tokens) {
    --current_draft_length_;
  }
}

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifyStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id) {
  if (draft_backend_ == nullptr || current_draft_length_ == 0) {
    const auto next = target_executor_.ForwardToken(current_token, cur_pos);
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    const bool hit_eos = (next == eos_id || next == 151643U ||
                          next == 248044U || next == 248046U);
    return {.emitted_tokens = {next},
            .accepted_count = 0,
            .draft_count = 0,
            .next_token = next,
            .hit_eos = hit_eos};
  }

  // 1. Propose draft tokens
  const auto proposal =
      draft_backend_->Propose(current_sequence, cur_pos, current_draft_length_);
  if (proposal.tokens.empty()) {
    const auto next = target_executor_.ForwardToken(current_token, cur_pos);
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    const bool hit_eos = (next == eos_id || next == 151643U ||
                          next == 248044U || next == 248046U);
    return {.emitted_tokens = {next},
            .accepted_count = 0,
            .draft_count = 0,
            .next_token = next,
            .hit_eos = hit_eos};
  }

  const std::size_t num_draft = proposal.tokens.size();

  // 2. Transactional state checkpoint at cur_pos
  target_executor_.SaveState(cur_pos);

  // 3. Execute target forward verification pass
  std::vector<tokenization::TokenId> target_predictions;
  target_predictions.reserve(num_draft + 1);

  tokenization::TokenId in_tok = current_token;
  std::uint32_t eval_pos = cur_pos;

  for (std::size_t i = 0; i < num_draft; ++i) {
    const auto target_pred = target_executor_.ForwardToken(in_tok, eval_pos);
    target_predictions.push_back(target_pred);
    if (target_pred != proposal.tokens[i]) {
      break;
    }
    in_tok = proposal.tokens[i];
    ++eval_pos;
  }

  // 4. Determine acceptance prefix
  std::size_t accepted_count = 0;
  while (accepted_count < target_predictions.size() &&
         accepted_count < num_draft &&
         target_predictions[accepted_count] ==
             proposal.tokens[accepted_count]) {
    ++accepted_count;
  }

  // Authoritative correction token from target model
  const tokenization::TokenId correction_token =
      target_predictions[accepted_count];

  StepResult result;
  result.draft_count = num_draft;
  result.accepted_count = accepted_count;

  // Build emitted tokens: accepted draft tokens + correction token
  for (std::size_t i = 0; i < accepted_count; ++i) {
    result.emitted_tokens.push_back(proposal.tokens[i]);
  }
  result.emitted_tokens.push_back(correction_token);
  result.next_token = correction_token;

  // Check EOS in emitted tokens
  for (const auto tok : result.emitted_tokens) {
    if (tok == eos_id || tok == 151643U || tok == 248044U || tok == 248046U) {
      result.hit_eos = true;
      break;
    }
  }

  // 5. Transactional state commit / rollback
  if (accepted_count < num_draft) {
    // Rollback speculative state beyond accepted tokens
    target_executor_.RestoreState();

    // Replay accepted tokens plus correction token
    tokenization::TokenId replay_in = current_token;
    std::uint32_t replay_pos = cur_pos;
    for (std::size_t i = 0; i < accepted_count; ++i) {
      (void)target_executor_.ForwardToken(replay_in, replay_pos, false);
      replay_in = proposal.tokens[i];
      ++replay_pos;
    }
    // Commit correction token
    (void)target_executor_.ForwardToken(replay_in, replay_pos, true);
  }

  // 6. Update stats and notify backend
  stats_.total_draft_tokens += num_draft;
  stats_.total_accepted_tokens += accepted_count;
  stats_.total_verification_steps += 1;
  stats_.total_emitted_tokens += result.emitted_tokens.size();

  UpdateAdaptiveDraftLength(accepted_count, num_draft);

  draft_backend_->AcceptFeedback(std::span<const tokenization::TokenId>(
                                     proposal.tokens.data(), accepted_count),
                                 correction_token);

  return result;
}

std::vector<tokenization::TokenId> SpeculativeVerifier::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }

  Reset();
  target_executor_.Reset();

  // 1. Prefill prompt
  tokenization::TokenId next_token =
      target_executor_.ForwardPromptBatch(prompt_tokens);

  std::vector<tokenization::TokenId> current_sequence(prompt_tokens.begin(),
                                                      prompt_tokens.end());
  std::uint32_t cur_pos = static_cast<std::uint32_t>(prompt_tokens.size());
  const auto eos_id = target_executor_.GetTokenizer().GetEosTokenId();

  // 2. Speculative Decode Generation Loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (next_token == eos_id || next_token == 151643U ||
        next_token == 248044U || next_token == 248046U) {
      break;
    }

    const auto step_res =
        VerifyStep(current_sequence, cur_pos, next_token, eos_id);

    bool should_stop = false;
    for (const auto tok : step_res.emitted_tokens) {
      output_tokens.push_back(tok);
      current_sequence.push_back(tok);
      ++cur_pos;

      if (on_token) {
        const auto piece = target_executor_.GetTokenizer().DecodeToken(tok);
        if (!on_token(tok, piece)) {
          should_stop = true;
          break;
        }
      }

      if (tok == eos_id || tok == 151643U || tok == 248044U || tok == 248046U ||
          output_tokens.size() >= options.max_new_tokens) {
        should_stop = true;
        break;
      }
    }

    if (should_stop || step_res.hit_eos) {
      break;
    }

    next_token = step_res.next_token;
  }

  return output_tokens;
}

}  // namespace strix::speculative
#endif  // defined(ENGINE_ENABLE_HIP)
