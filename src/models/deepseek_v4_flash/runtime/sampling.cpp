#include "native_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define DS4_SAMPLE_NEG_INF (-1.0e30f)

struct ds4_sample_candidate {
    int id;
    float logit;
    float probability;
};

static uint64_t rng_next(uint64_t *state) {
    uint64_t value = *state;
    if (value == 0) value = UINT64_C(0x9e3779b97f4a7c15);
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(0x2545f4914f6cdd1d);
}

static float rng_f32(uint64_t *state) {
    const uint64_t value = rng_next(state);
    return static_cast<float>((value >> 40) & UINT64_C(0xffffff)) /
           16777216.0f;
}

int ds4_sample_argmax(const float *logits, uint32_t vocabulary_size) {
    int best = 0;
    float best_value = DS4_SAMPLE_NEG_INF;
    for (uint32_t i = 0; i < vocabulary_size; ++i) {
        const float value = logits[i];
        if (value > best_value) {
            best_value = value;
            best = static_cast<int>(i);
        }
    }
    return best;
}

static int sample_full_vocabulary(const float *logits,
                                  uint32_t vocabulary_size,
                                  float temperature,
                                  float top_p,
                                  float min_p,
                                  uint64_t *rng_state) {
    float max_logit = DS4_SAMPLE_NEG_INF;
    int best = 0;
    uint32_t finite_count = 0;
    for (uint32_t i = 0; i < vocabulary_size; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) continue;
        ++finite_count;
        if (value > max_logit) {
            max_logit = value;
            best = static_cast<int>(i);
        }
    }
    if (finite_count == 0) {
        return ds4_sample_argmax(logits, vocabulary_size);
    }

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float minimum_relative_probability = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < vocabulary_size; ++i) {
            const float value = logits[i];
            if (!std::isfinite(value)) continue;
            const float probability = expf((value - max_logit) / temperature);
            if (probability < minimum_relative_probability) continue;
            sum += probability;
        }
        if (sum <= 0.0f || !std::isfinite(sum)) return best;

        float sample = rng_f32(rng_state) * sum;
        for (uint32_t i = 0; i < vocabulary_size; ++i) {
            const float value = logits[i];
            if (!std::isfinite(value)) continue;
            const float probability = expf((value - max_logit) / temperature);
            if (probability < minimum_relative_probability) continue;
            sample -= probability;
            if (sample <= 0.0f) return static_cast<int>(i);
        }
        return best;
    }

    std::vector<ds4_sample_candidate> candidates;
    candidates.reserve(finite_count);
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocabulary_size; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) continue;
        const float probability = expf((value - max_logit) / temperature);
        candidates.push_back(
            ds4_sample_candidate{static_cast<int>(i), value, probability});
        sum += probability;
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        return best;
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &left, const auto &right) {
        return left.logit > right.logit;
    });
    const float minimum_probability =
        (candidates[0].probability / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered_count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const float probability = candidates[i].probability / sum;
        if (i > 0 && probability < minimum_probability) break;
        filtered_sum += candidates[i].probability;
        ++filtered_count;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered_count == 0) {
        return best;
    }

    float sample = rng_f32(rng_state) * filtered_sum;
    for (uint32_t i = 0; i < filtered_count; ++i) {
        sample -= candidates[i].probability;
        if (sample <= 0.0f) {
            return candidates[i].id;
        }
    }
    return candidates[filtered_count - 1].id;
}

int ds4_sample_top_p_min_p(const float *logits,
                           uint32_t vocabulary_size,
                           float temperature,
                           int top_k,
                           float top_p,
                           float min_p,
                           uint64_t *rng_state) {
    if (temperature <= 0.0f) {
        return ds4_sample_argmax(logits, vocabulary_size);
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) {
        return sample_full_vocabulary(logits,
                                      vocabulary_size,
                                      temperature,
                                      top_p,
                                      min_p,
                                      rng_state);
    }
    if (top_k > 1024) top_k = 1024;
    if (static_cast<uint32_t>(top_k) > vocabulary_size) {
        top_k = static_cast<int>(vocabulary_size);
    }

    int ids[1024];
    float values[1024];
    int count = 0;
    for (uint32_t i = 0; i < vocabulary_size; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) continue;
        if (count == top_k && value <= values[count - 1]) continue;
        int position = count < top_k ? count++ : count - 1;
        while (position > 0 && values[position - 1] < value) {
            values[position] = values[position - 1];
            ids[position] = ids[position - 1];
            --position;
        }
        values[position] = value;
        ids[position] = static_cast<int>(i);
    }
    if (count == 0) return ds4_sample_argmax(logits, vocabulary_size);

    float probabilities[1024];
    const float max_logit = values[0];
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        probabilities[i] = expf((values[i] - max_logit) / temperature);
        sum += probabilities[i];
    }
    if (sum <= 0.0f || !std::isfinite(sum)) return ids[0];

    const float minimum_probability = (probabilities[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered_count = 0;
    for (int i = 0; i < count; ++i) {
        const float probability = probabilities[i] / sum;
        if (i > 0 && probability < minimum_probability) break;
        filtered_sum += probabilities[i];
        ++filtered_count;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered_count <= 0) return ids[0];

    float sample = rng_f32(rng_state) * filtered_sum;
    for (int i = 0; i < filtered_count; ++i) {
        sample -= probabilities[i];
        if (sample <= 0.0f) return ids[i];
    }
    return ids[filtered_count - 1];
}
