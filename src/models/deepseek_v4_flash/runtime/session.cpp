#include "native_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <new>

#define DS4_SESSION_NEG_INF (-1.0e30f)

struct ds4_session {
    ds4_engine *engine;
    ds4_rocm_graph *graph;
    ds4_tokens checkpoint;
    std::unique_ptr<float[]> logits;
    ds4_session_cancel_fn cancel;
    void *cancel_user_data;
    uint32_t prefill_capacity;
    int context_size;
    bool checkpoint_valid;
};

static void set_error(char *error,
                      size_t error_capacity,
                      const char *message) {
    if (error && error_capacity != 0) {
        snprintf(error, error_capacity, "%s", message);
    }
}

static bool session_cancelled(const ds4_session *session) {
    return session && session->cancel &&
           session->cancel(session->cancel_user_data);
}

int ds4_session_create(ds4_session **out,
                       ds4_engine *engine,
                       int context_size) {
    if (!out || !engine || context_size <= 0) return 1;

    auto session = std::unique_ptr<ds4_session>(
        new (std::nothrow) ds4_session{});
    if (!session) return 1;

    session->engine = engine;
    session->context_size = context_size;
    session->prefill_capacity =
        ds4_rocm_graph_prefill_capacity(context_size);
    session->graph = ds4_rocm_graph_create(engine,
                                           context_size,
                                           session->prefill_capacity);
    if (!session->graph) {
        return 1;
    }

    const int vocabulary_size = ds4_engine_vocab_size(engine);
    if (vocabulary_size <= 0 ||
        (size_t)vocabulary_size > SIZE_MAX / sizeof(session->logits[0])) {
        ds4_rocm_graph_destroy(session->graph);
        return 1;
    }
    session->logits.reset(new (std::nothrow) float[vocabulary_size]);
    if (!session->logits) {
        ds4_rocm_graph_destroy(session->graph);
        return 1;
    }

    *out = session.release();
    return 0;
}

void ds4_session_free(ds4_session *session) {
    if (!session) return;
    ds4_rocm_graph_destroy(session->graph);
    ds4_tokens_free(&session->checkpoint);
    delete session;
}

void ds4_session_set_cancel(ds4_session *session,
                            ds4_session_cancel_fn callback,
                            void *user_data) {
    if (!session) return;
    session->cancel = callback;
    session->cancel_user_data = user_data;
}

void ds4_session_invalidate(ds4_session *session) {
    if (!session) return;
    session->checkpoint_valid = false;
}

int ds4_session_sync(ds4_session *session,
                     const ds4_tokens *prompt,
                     char *error,
                     size_t error_capacity) {
    if (!session || !prompt || prompt->len <= 0 ||
        prompt->len >= session->context_size) {
        set_error(error, error_capacity, "prompt exceeds context");
        return 1;
    }
    if (session_cancelled(session)) {
        set_error(error, error_capacity, "prefill cancelled");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }

    if (session->checkpoint_valid &&
        prompt->len >= session->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &session->checkpoint)) {
        const int suffix = prompt->len - session->checkpoint.len;
        const uint32_t resume_minimum =
            ds4_rocm_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_minimum) {
            const bool ok = ds4_rocm_graph_prefill_range(
                session->graph,
                session->engine,
                prompt,
                (uint32_t)session->checkpoint.len,
                (uint32_t)suffix,
                session->logits.get());
            if (!ok) {
                set_error(error,
                          error_capacity,
                          "ROCm resumed prefill failed while extending checkpoint");
                session->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&session->checkpoint, prompt);
            session->checkpoint_valid = true;
            return 0;
        }

        for (int i = session->checkpoint.len; i < prompt->len; ++i) {
            if (session_cancelled(session)) {
                set_error(error, error_capacity, "prefill cancelled");
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!ds4_rocm_graph_eval(session->graph,
                                     session->engine,
                                     prompt->v[i],
                                     (uint32_t)session->checkpoint.len,
                                     session->logits.get())) {
                set_error(error,
                          error_capacity,
                          "ROCm decode failed while extending checkpoint");
                session->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_push(&session->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    session->checkpoint_valid = false;
    if (!ds4_rocm_graph_reset(session->graph)) {
        set_error(error, error_capacity, "ROCm prefill state reset failed");
        return 1;
    }
    if (!ds4_rocm_graph_prefill(session->graph,
                                session->engine,
                                prompt,
                                session->logits.get())) {
        set_error(error, error_capacity, "ROCm prefill failed");
        return 1;
    }
    ds4_tokens_copy(&session->checkpoint, prompt);
    session->checkpoint_valid = true;
    return 0;
}

int ds4_session_argmax(const ds4_session *session) {
    if (!session || !session->logits) return -1;
    return ds4_sample_argmax(session->logits.get(),
                             (uint32_t)ds4_engine_vocab_size(session->engine));
}

int ds4_session_argmax_excluding(const ds4_session *session,
                                 int excluded_token) {
    if (!session || !session->logits) return -1;

    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    int best = -1;
    float best_logit = DS4_SESSION_NEG_INF;
    for (int i = 0; i < vocabulary_size; ++i) {
        if (i == excluded_token) continue;
        const float value = session->logits[i];
        if (best < 0 || value > best_logit) {
            best = i;
            best_logit = value;
        }
    }
    return best;
}

int ds4_session_sample(const ds4_session *session,
                       float temperature,
                       int top_k,
                       float top_p,
                       float min_p,
                       uint64_t *rng_state) {
    if (!session || !session->logits || !rng_state) return -1;
    return ds4_sample_top_p_min_p(
        session->logits.get(),
        (uint32_t)ds4_engine_vocab_size(session->engine),
        temperature,
        top_k,
        top_p,
        min_p,
        rng_state);
}

int ds4_session_copy_logits(const ds4_session *session,
                            float *out,
                            int capacity) {
    if (!session || !out) return 0;
    const int vocabulary_size = ds4_engine_vocab_size(session->engine);
    if (capacity < vocabulary_size) return 0;
    memcpy(out,
           session->logits.get(),
           (size_t)vocabulary_size * sizeof(out[0]));
    return vocabulary_size;
}

int ds4_session_eval(ds4_session *session,
                     int token,
                     char *error,
                     size_t error_capacity) {
    if (!session) {
        set_error(error, error_capacity, "invalid session");
        return 1;
    }
    if (session_cancelled(session)) {
        set_error(error, error_capacity, "decode cancelled");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (!ds4_rocm_graph_eval(session->graph,
                             session->engine,
                             token,
                             (uint32_t)session->checkpoint.len,
                             session->logits.get())) {
        set_error(error, error_capacity, "ROCm decode failed");
        session->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_push(&session->checkpoint, token);
    session->checkpoint_valid = true;
    return 0;
}

uint64_t ds4_session_payload_bytes(const ds4_session *session) {
    if (!session || !session->checkpoint_valid) return 0;
    return ds4_rocm_graph_snapshot_bytes(session->graph,
                                         &session->checkpoint);
}

int ds4_session_save_snapshot(const ds4_session *session,
                              ds4_session_snapshot *snapshot,
                              char *error,
                              size_t error_capacity) {
    if (!session || !session->checkpoint_valid) {
        set_error(error, error_capacity, "session has no valid checkpoint");
        return 1;
    }
    return ds4_rocm_graph_save_snapshot(session->graph,
                                        &session->checkpoint,
                                        session->logits.get(),
                                        session->prefill_capacity,
                                        session->context_size,
                                        snapshot,
                                        error,
                                        error_capacity);
}

int ds4_session_load_snapshot(ds4_session *session,
                              const ds4_session_snapshot *snapshot,
                              char *error,
                              size_t error_capacity) {
    if (!session) {
        set_error(error, error_capacity, "invalid session snapshot load");
        return 1;
    }
    session->checkpoint_valid = false;
    const int result =
        ds4_rocm_graph_load_snapshot(session->graph,
                                     &session->checkpoint,
                                     session->logits.get(),
                                     session->prefill_capacity,
                                     session->context_size,
                                     snapshot,
                                     error,
                                     error_capacity);
    session->checkpoint_valid = result == 0;
    return result;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->ptr);
    memset(snapshot, 0, sizeof(*snapshot));
}

int ds4_session_pos(const ds4_session *session) {
    return session ? session->checkpoint.len : 0;
}

int ds4_session_ctx(const ds4_session *session) {
    return session ? session->context_size : 0;
}
