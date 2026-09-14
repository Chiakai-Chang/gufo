#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
// HIP fallback for the NVIDIA-specific D2R launchers in qfn_mmq_d2r.cu.
// Generic MMQ remains available; callers will take its raw-layout path.

#include "qfn_mmq_d2r.cuh"

bool qfn_mmq_q2_K_moe_d2r_available(int) {
    return false;
}

bool qfn_mmq_iq2_xxs_moe_d2r_available(int) {
    return false;
}

bool qfn_mmq_q8_0_dense_d2r_available(int) {
    return false;
}

size_t qfn_mmq_q2_K_moe_d2r_scratch_bytes(int64_t, int) {
    return 0;
}

size_t qfn_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t, int) {
    return 0;
}

size_t qfn_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(int64_t, int) {
    return 0;
}

int qfn_mmq_q2_K_moe_d2r_launch(
        const void *, int64_t, const void *, const int32_t *, const int32_t *,
        float *, int, int, int64_t, int, void *, size_t, cudaStream_t) {
    return -1;
}

int qfn_mmq_q8_0_dense_d2r_launch(
        const void *, const void *, float *, int, int, int, cudaStream_t) {
    return -1;
}

int qfn_mmq_iq2_xxs_moe_d2r_pair_launch(
        const void *, const void *, int64_t, const void *, const int32_t *,
        const int32_t *, float *, float *, int, int, int64_t, int, void *,
        size_t, cudaStream_t) {
    return -1;
}

int qfn_mmq_iq2_xxs_moe_d2r_fused_launch(
        const void *, const void *, int64_t, const void *, const int32_t *, int,
        const int32_t *, const int32_t *, const float *, void *, int, int,
        int64_t, int, float, void *, size_t, cudaStream_t) {
    return -1;
}

}  // namespace qfn_mmq
