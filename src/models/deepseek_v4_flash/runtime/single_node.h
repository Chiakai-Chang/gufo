#ifndef STRIX_DEEPSEEK_V4_FLASH_SINGLE_NODE_H
#define STRIX_DEEPSEEK_V4_FLASH_SINGLE_NODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "model.h"

typedef struct ds4_dist_session ds4_dist_session;
typedef struct ds4_tp ds4_tp;

#define DS4_LAYER_PACK_MAX_GPUS 16
#define DS4_LAYER_PACK_CPU (-1)

typedef struct {
    size_t gpu_budget_bytes[DS4_LAYER_PACK_MAX_GPUS];
    int n_gpus;
} ds4_layer_pack_config;

enum {
    DS4_TP_GATE_ATTN = 0,
    DS4_TP_GATE_FFN = 1,
    DS4_TP_GATES_PER_LAYER = 2,
    DS4_TP_BATCH_MAX_ROWS = 8,
};

typedef struct {
    uint64_t session_id;
    int32_t token;
    uint32_t reserved;
} ds4_tp_batch_item;

int ds4_dist_session_create(ds4_dist_session **out,
                            ds4_engine *engine,
                            const ds4_distributed_options *opt,
                            ds4_session *owner,
                            int ctx_size,
                            char *err,
                            size_t errlen);
void ds4_dist_session_free(ds4_dist_session *session);
int ds4_dist_session_route_ready(ds4_dist_session *session,
                                 char *err,
                                 size_t errlen);
int ds4_dist_session_sync(ds4_dist_session *session,
                          ds4_session *owner,
                          const ds4_tokens *checkpoint,
                          const ds4_tokens *prompt,
                          float *logits,
                          char *err,
                          size_t errlen);
int ds4_dist_session_eval(ds4_dist_session *session,
                          ds4_session *owner,
                          const ds4_tokens *checkpoint,
                          int token,
                          float *logits,
                          char *err,
                          size_t errlen);
int ds4_dist_session_save_payload(ds4_dist_session *session,
                                  ds4_session *owner,
                                  FILE *file,
                                  char *err,
                                  size_t errlen);
int ds4_dist_session_load_payload(ds4_dist_session *session,
                                  ds4_session *owner,
                                  FILE *file,
                                  uint64_t payload_bytes,
                                  char *err,
                                  size_t errlen);

bool ds4_tp_failed(const ds4_tp *tp);
int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size);
int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id);
int ds4_tp_send_sync(ds4_tp *tp,
                     uint64_t session_id,
                     const int *tokens,
                     uint32_t n_tokens);
int ds4_tp_send_eval(ds4_tp *tp,
                     uint64_t session_id,
                     uint64_t sequence,
                     int token);
int ds4_tp_send_rewind(ds4_tp *tp, uint64_t session_id, int position);
int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id);
int ds4_tp_send_eval_batch(ds4_tp *tp,
                           const ds4_tp_batch_item *items,
                           uint32_t count);
int ds4_tp_send_mixed_batch(ds4_tp *tp,
                            uint64_t prefill_session_id,
                            const int *prompt,
                            uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count);
int ds4_tp_wait_command_ack(ds4_tp *tp,
                            uint64_t session_id,
                            const char *operation,
                            char *err,
                            size_t errlen);
int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count);
int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count);
int ds4_tp_send_verify(ds4_tp *tp,
                       uint64_t session_id,
                       const int *drafts,
                       uint32_t count);
int ds4_tp_send_verify_commit(ds4_tp *tp,
                              int32_t full_accept,
                              int32_t replay_count);
int ds4_tp_recv_verify_commit(ds4_tp *tp,
                              int32_t *full_accept,
                              int32_t *replay_count);

typedef struct {
    void *ptr;
    uint64_t bytes;
} ds4_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} ds4_ssd_cache_plan;

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool ds4_ssd_auto_cache_plan(uint64_t recommended_bytes,
                             uint64_t non_routed_bytes,
                             uint64_t per_expert_bytes,
                             uint64_t max_model_experts,
                             ds4_ssd_cache_plan *out);
bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock, uint64_t bytes);
void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock);

int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *config,
                                int *device_for_entry);
void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus);

#endif
