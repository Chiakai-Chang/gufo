# Qwen3.8-Flash-Next on Strix Halo

Linux x86-64, AMD `gfx1151`, 128 GB unified memory. Artifact:
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` (four shards).
MTP: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from the same revision.
The target occupies about 77 GiB of GPU memory; its 26.8 GiB n-gram table
is read from disk. Context, scratch and the optional MTP block add memory.
Loading takes **14.5 s** for the target or **15.1 s** with MTP on the test
machine's NVMe drive. Sixteen disk readers share 256 MiB of temporary staging,
released before inference; weights retain their device-resident encoding.

## Performance

Use `nix build` binaries. C1, pp2048, tg128, batch 2048, one prefill warm-up and one
measured repetition (three for MTP at depth 0; 2026-09-16). Depth is the number of tokens prepared
before the measured operation. Fixed-length TG continues past EOS.
TODO entries have no current qualified measurement.

| Depth | AR pp2048 (tok/s) | AR tg128 (tok/s) | MTP pp2048 (tok/s) | MTP tg128 (tok/s) | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1271.79 | 22.63 | 1248.54 | 25.84 | 31.2% |
| 4096 | 1161.97 | 21.82 | 1122.28 | 47.36 | 81.2% |
| 8192 | TODO | TODO | TODO | TODO | TODO |
| 12288 | TODO | TODO | TODO | TODO | TODO |
| 16384 | TODO | TODO | TODO | TODO | TODO |

Sampling: temperature 0.7, seed 1; MTP scores the first 65536 vocabulary
entries and proposes up to seven tokens. Target verification always scores
the full vocabulary. Greedy and temperature 1.0/top-p 0.95 throughput: TODO.
Concurrent throughput: TODO; serving interleaves independent sessions but
does not batch their model work. `gufo bench` supports C1 only for this model.

```sh
./result/bin/gufo bench --model "$MODEL" -b 2048 -p 2048 -n 128 \
  -d 0,4096 --temperature 0.7 --seed 1 -v
./result/bin/gufo bench --model "$MODEL" -b 2048 -p 2048 -n 128 \
  -d 0,4096 --temperature 0.7 --seed 1 -v \
  --speculative mtp --mtp-model "$MTP" --draft-tokens 7 --draft-vocab 65536
./result/bin/gufo prompt --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --temperature 0.7 --seed 1 'Explain sparse attention.'
```

The same model and MTP options work in `chat` and `serve llm`. MTP uses a
fixed chain of 1–7 draft tokens. Draft proposals are deterministic; each
verification row uses the request's target sampler, including filters and
penalties. Rejection leaves the RNG at that target draw, so the next step
samples the same frontier. Stops and output budgets consume no later draws.

## Quality checks

Tests live in `tests/models/qwen38_flash_next/`. Run through Nix:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_tests
nix develop -c ctest --test-dir build/gpu-test -L qwen38_flash_next --output-on-failure
build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test \
  --model "$MODEL" --mtp-model "$MTP"
```

- CPU sampler checks cover 23 strategy combinations, accepted/rejected
  prefixes, penalties, stop tokens and RNG rollback.
- Model replay compares AR and MTP token IDs, full frontier logits, RNG and
  positions at short and 4096-token contexts. It repeats MTP with and without
  interleaved sessions to check hidden-state ownership and prefix resets.
  Both three- and seven-token draft limits pass. The sampled release tg128
  runs above also produce identical AR/MTP token hashes at both depths.
- Operator tests cover attention, DeltaNet, hyper-connections, expert routing,
  quantized projections and epilogues against numerical references. Dense
  and routed vector tests require identical results across verification
  widths; paired expert projections must match separate projections.
- CLI parser checks retain every sampling field and MTP option. Exercise
  prompt, chat and sampled HTTP requests after changing their wiring.
- Upload checks compare every byte and destination guard across shards,
  chunk boundaries, unaligned ranges and EOF; truncated files and invalid
  ranges must fail. Loading changes must retain the release token hashes.

For a kernel change, run its operator test first, then model replay if it
affects decode or verification. Prefill changes also require
`gufo bench --validate-prefill 2048` and comparison with the CPU/reference
probes. Prefill uses different arithmetic from sequential decoding; AR/MTP
agreement after the same prefill does not establish agreement with an
unquantized upstream model. That full-model reference qualification is TODO.
