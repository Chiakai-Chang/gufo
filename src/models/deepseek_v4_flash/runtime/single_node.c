#include "single_node.h"

#include <stdio.h>
#include <string.h>

static int unsupported(char *err, size_t errlen) {
    if (err != NULL && errlen > 0) {
        snprintf(err, errlen,
                 "this DeepSeek V4 Flash build supports one local ROCm device");
    }
    return 0;
}

int ds4_dist_session_create(ds4_dist_session **out,
                            ds4_engine *engine,
                            const ds4_distributed_options *opt,
                            ds4_session *owner,
                            int ctx_size,
                            char *err,
                            size_t errlen) {
    (void)engine;
    (void)opt;
    (void)owner;
    (void)ctx_size;
    if (out != NULL) *out = NULL;
    return unsupported(err, errlen);
}

void ds4_dist_session_free(ds4_dist_session *session) {
    (void)session;
}

int ds4_dist_session_route_ready(ds4_dist_session *session,
                                 char *err,
                                 size_t errlen) {
    (void)session;
    unsupported(err, errlen);
    return -1;
}

int ds4_dist_session_sync(ds4_dist_session *session,
                          ds4_session *owner,
                          const ds4_tokens *checkpoint,
                          const ds4_tokens *prompt,
                          float *logits,
                          char *err,
                          size_t errlen) {
    (void)session;
    (void)owner;
    (void)checkpoint;
    (void)prompt;
    (void)logits;
    unsupported(err, errlen);
    return 1;
}

int ds4_dist_session_eval(ds4_dist_session *session,
                          ds4_session *owner,
                          const ds4_tokens *checkpoint,
                          int token,
                          float *logits,
                          char *err,
                          size_t errlen) {
    (void)session;
    (void)owner;
    (void)checkpoint;
    (void)token;
    (void)logits;
    unsupported(err, errlen);
    return 1;
}

int ds4_dist_session_save_payload(ds4_dist_session *session,
                                  ds4_session *owner,
                                  FILE *file,
                                  char *err,
                                  size_t errlen) {
    (void)session;
    (void)owner;
    (void)file;
    unsupported(err, errlen);
    return 1;
}

int ds4_dist_session_load_payload(ds4_dist_session *session,
                                  ds4_session *owner,
                                  FILE *file,
                                  uint64_t payload_bytes,
                                  char *err,
                                  size_t errlen) {
    (void)session;
    (void)owner;
    (void)file;
    (void)payload_bytes;
    unsupported(err, errlen);
    return 1;
}

bool ds4_tp_failed(const ds4_tp *tp) {
    (void)tp;
    return true;
}

#define DS4_TP_UNSUPPORTED(name, signature, unused) \
    int name signature {                            \
        unused                                      \
        return 0;                                   \
    }

DS4_TP_UNSUPPORTED(ds4_tp_send_session_create,
                   (ds4_tp *tp, uint64_t session_id, int ctx_size),
                   (void)tp; (void)session_id; (void)ctx_size;)
DS4_TP_UNSUPPORTED(ds4_tp_send_session_destroy,
                   (ds4_tp *tp, uint64_t session_id),
                   (void)tp; (void)session_id;)
DS4_TP_UNSUPPORTED(ds4_tp_send_sync,
                   (ds4_tp *tp, uint64_t session_id,
                    const int *tokens, uint32_t n_tokens),
                   (void)tp; (void)session_id; (void)tokens; (void)n_tokens;)
DS4_TP_UNSUPPORTED(ds4_tp_send_eval,
                   (ds4_tp *tp, uint64_t session_id,
                    uint64_t sequence, int token),
                   (void)tp; (void)session_id; (void)sequence; (void)token;)
DS4_TP_UNSUPPORTED(ds4_tp_send_rewind,
                   (ds4_tp *tp, uint64_t session_id, int position),
                   (void)tp; (void)session_id; (void)position;)
DS4_TP_UNSUPPORTED(ds4_tp_send_invalidate,
                   (ds4_tp *tp, uint64_t session_id),
                   (void)tp; (void)session_id;)
DS4_TP_UNSUPPORTED(ds4_tp_send_eval_batch,
                   (ds4_tp *tp, const ds4_tp_batch_item *items,
                    uint32_t count),
                   (void)tp; (void)items; (void)count;)
DS4_TP_UNSUPPORTED(ds4_tp_send_mixed_batch,
                   (ds4_tp *tp, uint64_t prefill_session_id,
                    const int *prompt, uint32_t prompt_count,
                    const ds4_tp_batch_item *items, uint32_t count),
                   (void)tp; (void)prefill_session_id; (void)prompt;
                   (void)prompt_count; (void)items; (void)count;)
DS4_TP_UNSUPPORTED(ds4_tp_send_logits_half,
                   (ds4_tp *tp, const float *half, uint32_t count),
                   (void)tp; (void)half; (void)count;)
DS4_TP_UNSUPPORTED(ds4_tp_recv_logits_half,
                   (ds4_tp *tp, float *half, uint32_t count),
                   (void)tp; (void)half; (void)count;)
DS4_TP_UNSUPPORTED(ds4_tp_send_verify,
                   (ds4_tp *tp, uint64_t session_id,
                    const int *drafts, uint32_t count),
                   (void)tp; (void)session_id; (void)drafts; (void)count;)
DS4_TP_UNSUPPORTED(ds4_tp_send_verify_commit,
                   (ds4_tp *tp, int32_t full_accept, int32_t replay_count),
                   (void)tp; (void)full_accept; (void)replay_count;)
DS4_TP_UNSUPPORTED(ds4_tp_recv_verify_commit,
                   (ds4_tp *tp, int32_t *full_accept, int32_t *replay_count),
                   (void)tp; (void)full_accept; (void)replay_count;)

#undef DS4_TP_UNSUPPORTED

int ds4_tp_wait_command_ack(ds4_tp *tp,
                            uint64_t session_id,
                            const char *operation,
                            char *err,
                            size_t errlen) {
    (void)tp;
    (void)session_id;
    (void)operation;
    return unsupported(err, errlen);
}

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes) {
    (void)bytes;
    (void)per_expert_bytes;
    return 0;
}

bool ds4_ssd_auto_cache_plan(uint64_t recommended_bytes,
                             uint64_t non_routed_bytes,
                             uint64_t per_expert_bytes,
                             uint64_t max_model_experts,
                             ds4_ssd_cache_plan *out) {
    (void)recommended_bytes;
    (void)non_routed_bytes;
    (void)per_expert_bytes;
    (void)max_model_experts;
    if (out != NULL) memset(out, 0, sizeof(*out));
    return false;
}

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock, uint64_t bytes) {
    (void)bytes;
    if (lock != NULL) memset(lock, 0, sizeof(*lock));
    return false;
}

void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock) {
    if (lock != NULL) memset(lock, 0, sizeof(*lock));
}

int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *config,
                                int *device_for_entry) {
    (void)entry_bytes;
    (void)n_entries;
    (void)config;
    (void)device_for_entry;
    return 1;
}

void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus) {
    (void)out;
    (void)device_for_entry;
    (void)n_entries;
    (void)n_layers;
    (void)entry_bytes;
    (void)gpu_used_bytes;
    (void)gpu_budget_bytes;
    (void)n_gpus;
}
