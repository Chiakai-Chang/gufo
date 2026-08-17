---
id: M005-C003
title: "Implement transactional checksum-validating text artifact loading"
milestone: M005
status: planned
dependencies: [M003-C001, M003-C002, M003-C004, M003-C011, M003-C013, M005-C001]
---

# M005-C003: Implement transactional checksum-validating text artifact loading

## Dependencies

- [M003-C001](../003-test-oracles-model-contracts/001-checked-tensor-descriptor.md)
- [M003-C002](../003-test-oracles-model-contracts/002-safetensors-inspection-hardening.md)
- [M003-C004](../003-test-oracles-model-contracts/004-qwen35-08b-descriptor.md)
- [M003-C011](../003-test-oracles-model-contracts/011-candidate-shq-byte-vectors.md)
- [M003-C013](../003-test-oracles-model-contracts/013-milestone2-contract-gate.md)
- [M005-C001](001-gpu-only-strix-server-target.md)

## Required Context

- M4 requires a safetensors-derived artifact and fail-closed loading. Source and quantization tooling already emits manifests and candidate SHQ planes, but no independent native loader exists.

## Goal

Reserve the complete 0.8B GPU-only load budget, validate manifest identity, bounds, offsets, alignment, dtypes, tokenizer hashes, text capability, and checksums, warm final immutable allocations, then atomically publish the model.

## Non-Goals

- Artifact conversion
- SHQ v1 freeze
- Runtime repacking
- NPU views
- Hot multi-model swapping

## Expected Paths

- read: `tools/strix/manifest.py`
- read: `tools/strix/safetensors.py`
- read: `docs/MEMORY_MANAGEMENT.md`
- read: `docs/PROJECT_STATUS.md`
- write: `src/core/artifact/text_artifact.*`
- write: `src/core/model_loader.*`
- write: `src/core/memory/immutable_weight_storage.*`
- write: `tests/fixtures/formats/qwen35_08b_*`
- write: `tests/model_loader_test.*`

## Definition of Done

- [ ] A valid 0.8B text-only artifact publishes only after all required tensors and tokenizer assets validate.
- [ ] Truncation, overlap, overflow, wrong model kind, vision-only capability, hash mismatch, and injected failure at every load stage leave no published model or leaked reservation.
- [ ] Weights use final aligned immutable storage and no runtime repack occurs.
- [ ] Load diagnostics identify source revision, artifact hash, encoding, and resident bytes.

## Development Loop

```sh
nix develop -c cmake --preset hip-test
nix develop -c cmake --build --preset hip-test
git add -A
nix build
nix develop -c ctest --preset hip-test --output-on-failure -R 'artifact_(valid|malformed|checksum)|model_load_rollback'
nix develop -c ctest --preset test --output-on-failure -R 'm005_c003_artifact_fixtures'
```

## Hardware Validation

```sh
nix develop -c ctest --preset test --output-on-failure -R 'm005_c003_artifact_fixtures'
```

## Output Artifacts

- Malformed artifact fixture corpus
- Transactional load-stage fault matrix JSON
- Valid 0.8B load inventory and memory report

## Stop Conditions

- Stop if source revision, tokenizer hashes, or complete destination budget cannot be verified before publication.
- Stop if a loader failure leaves resident GPU bytes or a visible alias.

## ROADMAP Traceability

- tasks: M4.1 transactional loading and checksums
- exitCriteria: Safetensors-derived artifact can be loaded
- exitCriteria: No general allocation prerequisite
- docs: docs/MEMORY_MANAGEMENT.md#Weight-Loading
- docs: docs/TESTING.md#T0:-Static-and-format-tests

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
