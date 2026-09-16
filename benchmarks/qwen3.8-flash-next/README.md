# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Artifact:
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.
Target weights occupy about 77 GiB of GPU memory; the 26.8 GiB n-gram table
stays on disk. Context, scratch and MTP add memory. NVMe loading takes about
15 seconds; 256 MiB of upload staging is released before inference.

## Performance

Nix release, C1, pp2048, tg128, seed 1, one prefill warm-up and one measured
repetition (2026-09-16 UTC). Rates are tok/s. Depth precedes the measured operation;
fixed-length generation continues past EOS. MTP scores the full vocabulary
and proposes up to seven drafts per cycle.

| Depth | AR pp2048 | MTP pp2048 |
| ---: | ---: | ---: |
| 0 | 1259.33 | 1270.26 |
| 4096 | 1146.71 | 1115.88 |

| Sampling | Depth | AR tg128 | MTP tg128 | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0 | 24.19 | TODO | TODO |
| Greedy | 4096 | 23.31 | TODO | TODO |
| Temperature 0.7 | 0 | TODO | 24.91 | 34.9% |
| Temperature 0.7 | 4096 | TODO | 47.20 | 92.4% |
| Temperature 1.0, top-p 0.95 | 0 | TODO | TODO | TODO |
| Temperature 1.0, top-p 0.95 | 4096 | TODO | TODO | TODO |

Depths 8192/12288/16384 and concurrent throughput: TODO. Serving interleaves
sessions but does not batch model work; `gufo bench` supports C1 for this model.

```sh
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v
./result/bin/gufo bench --model "$MODEL" -p 2048 -n 128 \
  -d 0,4096 --temperature 0 --seed 1 -v \
  --speculative mtp --mtp-model "$MTP"
```

The model uses 2048-token internal chunks, selected by a 512–4096 sweep.
Resident readers gather n-gram rows while layer 0 runs, including graph replay.
HTTP accepts arbitrary prompt lengths and yields between those chunks;
spare session slots do not reduce the idle chunk size. Dense projection plans
are deterministic and depend on exact shapes and the pinned HIP library.
The [projection sweep](tools/projection_plans.hip) uses `tools/bench/build.sh`.

`prompt`, `chat` and `serve llm` share MTP and sampling options. Drafts are
greedy; verification uses the target sampler with its filters, penalties and
RNG. Chains support 1–7 drafts. Adaptive MTP: TODO. Shorter chains favored
short prefixes; longer chains favored the repetitive 4K case. Fixed seven-draft
chains can be slower than AR when acceptance is low.

## Quality checks

Tests live in `tests/models/qwen38_flash_next/`:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_tests
nix develop -c ctest --test-dir build/gpu-test -L qwen38_flash_next --output-on-failure
build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test \
  --model "$MODEL" --mtp-model "$MTP"
```

- Model replay checks AR/MTP tokens, full logits, RNG and positions at short
  and 4K contexts, including fresh loads, request order, draft widths and resets.
- The shared 23-case Qwen sampler matrix runs through serving with AR/MTP,
  replay, C2 and short output budgets. CPU/GPU tests cover combined filters;
  verifier tests cover accepted/rejected prefixes, stops and RNG rollback.
- Operator tests cover attention, DeltaNet, hyper-connections, routing and
  projections against numerical references. F16 plans require FP64 agreement
  and bitwise replay, including ragged sizes and fallback shapes.
- Upload tests check bytes, guards, shard/chunk boundaries and invalid inputs.
  N-gram tests check asynchronous reads, duplicates, failed I/O and teardown.
  Wiring changes also require actual prompt, chat and sampled HTTP requests.

Run the affected operator test first, then model replay for retained changes.
Prefill changes also require `gufo bench --validate-prefill N` and CPU/reference
probes. The same artifact, build, hardware and request setup must reproduce
outputs. Batched prefill differs numerically from sequential decoding;
AR/MTP agreement does not establish unquantized upstream equivalence.
Full-model upstream quality qualification remains TODO.
