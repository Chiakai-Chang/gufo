# MiniMax H3 BF16 Residency Baseline

Status: initial gfx1151 loader decision, 2026-08-21

## Contract

The production loader accepts only the committed source manifest with SHA-256
`8776014efafac996761041c0e3df740b41667275ed12a1e021cf8b49faf9b009`.
Inspection parses the two safetensors indexes and every shard header with
bounded `pread`; it maps no tensor payload and performs no HIP allocation.

The immutable BF16/F32 inventory is divided into non-overlapping execution
phases:

| Phase | Tensor bytes | GiB |
| --- | ---: | ---: |
| Qwen prompt encoder | 66,714,780,128 | 62.13 |
| AdaLN/time precompute | 26,142,079,488 | 24.35 |
| DiT core and heads | 40,138,350,656 | 37.38 |
| VisualVAE | 10,415,484,128 | 9.70 |
| AudioVAE | 605,306,340 | 0.56 |

A `PhaseSession` owns its stream, page-aligned file mappings, read-only HIP
registrations or device-copy allocation, persistent arena, and scratch arena.
Destruction and every failed load unwind those resources in reverse order.
Only one phase session may be promoted by the orchestrator at a time, so the
five phase peaks do not sum.

## Initial gfx1151 Measurement

Machine: the supported 128 GB Strix Halo development host, ROCm 7.2.3,
integrated `gfx1151`. Artifact: pinned MiniMax H3 FL2VA BF16 checkpoint.
The focused AudioVAE residency case covers 605,306,340 tensor bytes plus two
16 MiB arenas.

```sh
STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_SOURCE_MANIFEST=src/models/minimax_h3/\
MINIMAX_H3_FL2VA_BF16.source-manifest.json \
./build-h3-166-hip/minimax_h3_runtime_hip_test
```

Observed process runs:

| Cache state/order | Mapped read-only | Device copy |
| --- | ---: | ---: |
| First cold-ish run | 226.9 ms | 98.2 ms |
| Immediate warm run | 73.3 ms | 97.0 ms |

Metadata inspection took 10.6–14.0 ms and read about 376 KB of safetensors
headers. It read zero tensor payload bytes and mapped or allocated zero model
bytes.

The initial default is explicit device copy. Its load time was stable across
the two runs, its compute-access behavior is conventional, and no H3 kernel
profile exists yet to justify direct system-memory reads. Read-only
`hipHostRegisterMapped | hipHostRegisterReadOnly` is retained as an explicit
diagnostic mode: it succeeded on the target and avoids the 605 MB device copy,
but its cold load was sensitive to page residency.

Issue #175 must revisit this default using alternating, thermally balanced
complete generations. A metadata or load-only microbenchmark is not an
end-to-end performance claim.

## Regression Gates

- Duplicate JSON keys, unsafe shard paths, unknown indexed tensors, bad dtypes,
  shape/byte mismatches, overlaps, missing shards, and wrong source-manifest
  bytes fail before payload mapping.
- Failure injection covers target validation, stream creation, mapping,
  prefault, registration/copy, synchronization, and arena allocation.
- Cancellation before publication releases all resources.
- Entering timed execution performs no allocation or first-touch operation.
- The real-checkpoint HIP test remains opt-in through explicit model and
  manifest environment variables; normal CI never downloads weights.
