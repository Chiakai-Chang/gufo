# MiniMax H3 Upstream Provenance

## Model checkpoint

```text
repository: MiniMaxAI/MiniMax-H3
revision:   42ed227ee7df40d41602854ae760620d6eb651fe
partition:  FL2VA
```

The checkpoint is an external operator-supplied artifact and is not
distributed by Strix-Halo.cpp. Its public license metadata and acquisition
boundary are recorded in `docs/MINIMAX_H3.md` and
`THIRD_PARTY_NOTICES.md`.

## Native implementation reference

```text
repository: antirez/h3.c
revision:   8974cc055ea9c02fcd14cc27dfda3e1027c05153
license:    MIT
copyright:  Copyright (c) 2026 Salvatore Sanfilippo
```

The pinned source supplies the initial independent implementation contract for:

- safetensors tensor names and component boundaries;
- tokenizer and Qwen3-VL layer-50 prompt encoding;
- packed text/video/audio sequence layout;
- sigma schedule, RNG, Euler integration, and core reuse;
- the 50-block Omni Transformer;
- VisualVAE and AudioVAE decoding;
- selected-frame development and audiovisual mux behavior;
- exact, fast, and aggressive performance references.

Metal, MPSGraph, Objective-C runtime integration, command-line presentation,
and Apple-specific resource management are not imported into the Linux
runtime.

## TensorOps ancestry

The pinned h3.c `THIRD_PARTY_NOTICES.md` states that the rectangular Morton
decoder and dynamic symmetric INT8/Metal TensorOps scheduling design in
`h3_shaders.metal` are adapted from ccv's `NAMatMulKernel` and
`NAInt8MatMulKernel` under BSD-3-Clause:

```text
Copyright (c) 2010, Liu Liu
All rights reserved.
```

If source expression or a recognizably adapted implementation from those
sections is brought into the gfx1151 backend, the BSD-3-Clause notice and
conditions must remain with the affected source and binary distribution.

## Adaptation record

Every copied or adapted file must record:

- upstream path and pinned revision;
- local destination;
- whether it was copied, translated, or independently reimplemented;
- material local changes;
- applicable MIT and, where relevant, BSD-3-Clause notices.

The model-private H3 implementation lives under `src/models/minimax_h3/`.
Shared engine infrastructure must not gain H3 tensor-name guesses or
model-specific numerical dispatch.

Current prompt-boundary adaptations:

| Upstream source | Local destination | Method and material changes |
| --- | --- | --- |
| `h3_tokenizer.m` | `tokenizer.cpp`, `tokenizer.hpp` | Semantic translation to C++20; Foundation storage replaced by strict project JSON and ICU NFC/category APIs; stronger duplicate/range/schema rejection |
| `h3_text_encoder.c` | `prompt_encoder.hip`, `prompt_encoder.hpp` | Layer/tensor contract adapted; Metal resource management replaced by device-copy streaming, next-layer prefetch, hipBLASLt, phase lease, cancellation, and telemetry |
| text kernels in `h3_shaders.metal` | `prompt_encoder_ops.cuh` | Operation-boundary translation to HIP; BF16 rounding and FP32 reductions retained; no Metal/MPSGraph or ccv TensorOps code imported |
| `tests/test_tokenizer.c`, `tests/test_real_prompt.c` | MiniMax H3 tokenizer and prompt-encoder tests | Released corpus retained; synthetic malformed cases, analytic HIP fixtures, content-hashed external Transformers boundaries, repetition, and leak checks added |
