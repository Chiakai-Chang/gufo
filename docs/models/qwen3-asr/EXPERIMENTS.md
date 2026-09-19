# Qwen3-ASR experiments

| Experiment | Decision / evidence |
| --- | --- |
| Fused LDS-staged decode GEMV | Retained; fixed reductions, official IDs unchanged, near measured DRAM ceiling. |
| QKV/gate-up dispatch grouping | Retained; shares inputs/weights without changing each row's dot product. |
| Audio output stays on GPU | Retained; host receives only finite-status scalar; exact transcript. |
| Overlapped audio/text loading | Retained; independent model-owned resources. |
| Model-specific hipBLASLt selection | Rank 6 retained for strongest prefill-boundary agreement, even though ranks 7/8 were slightly faster. Requalify after library changes. |
| Mapped decoder weights | Rejected: lower loading latency but 58% slower resident requests. |
| Official 104-token audio window | Rejected: encoder quality failure and nonfinite long fixture; parity gap remains open. |
| Fused SwiGLU / non-temporal weight loads | Rejected: slower complete request. |
| Handwritten BF16 prefill WMMA | Rejected: slower than hipBLASLt on these shapes. |
| Matrix-core attention port | Not implemented: small fraction of measured request; quantify benefit first. |

Next: cold-weight GEMM plan selection, launch overhead and long-form quality.
No precision reduction is qualified by these experiments.
