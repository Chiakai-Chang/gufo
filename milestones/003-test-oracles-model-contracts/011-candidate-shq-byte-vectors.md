---
id: M003-C011
title: "Commit byte-exact candidate SHQ4/SHQ6/SHQ8 conformance vectors"
milestone: M003
status: planned
dependencies: [M003-C001, M003-C008]
---

# M003-C011: Commit byte-exact candidate SHQ4/SHQ6/SHQ8 conformance vectors

## Dependencies

- [M003-C001](001-checked-tensor-descriptor.md)
- [M003-C008](008-high-precision-cpu-operator-oracles.md)

## Required Context

- tools/strix/shq.py and tools/strix/conformance.py already quantize, pack, and dequantize all three tiers. Milestone 2 needs committed language-neutral vectors, while PROJECT_STATUS explicitly forbids freezing v1 before later C++/HIP/AIE/malformed gates.

## Goal

Generate, review, commit, and validate small canonical candidate-contract fixtures for SHQ4-T16, SHQ6-T16, and SHQ8-T16, including exact packed planes and CPU expected values.

## Non-Goals

- Do not freeze or label the format SHQ-T16 v1.
- Do not implement HIP/AIE kernels or a production SHQ container.
- Do not alter quantization recipes based on the held-out evaluation suite.

## Expected Paths

- `tools/strix/shq.py`
- `tools/strix/conformance.py`
- `tests/tools/test_shq_conformance.py`
- `tests/fixtures/formats/shq-candidate/`
- `docs/QUANTIZATION.md`

## Definition of Done

- [ ] Fixtures cover every U4 and signed nibble, every signed-6 boundary and 4-per-3-byte packing, every signed-8 boundary, low/high order, G32/G64 where applicable, offsets/alignment, BF16 RNE ties-even, saturation, all-zero groups, non-finite rejection, and K/N tails.
- [ ] Dynamic A8 rounding/saturation, exact INT32 U4Z correction/group sums, and FP32 epilogue expected outputs are included.
- [ ] Each vector records candidate format ID, logical/padded shape, group size, plane lengths/checksums, quantizer mode/version, source values, packed bytes, and expected dequantized/accumulated values.
- [ ] Python implementation round-trips all fixtures byte-exactly; checked descriptor validation accepts valid ranges and rejects malformed/truncated/overlapping metadata.
- [ ] Documentation continues to call SHQ-T16 a candidate and lists C++ parser, HIP, promoted AIE, and malformed-artifact gates still required before v1 freeze.

## Development Loop

```sh
nix develop -c python3 -m unittest -v tests.tools.test_shq_conformance
nix develop -c python3 -c "import sys; sys.path.insert(0,'tools'); from strix.conformance import run_all; run_all()"
```

## Output Artifacts

- Committed language-neutral SHQ candidate vectors
- Vector checksums and generation provenance

## Stop Conditions

- Stop if any fixture is generated nondeterministically or regenerated automatically during comparison.
- Stop if SHQ-T16 is presented as frozen v1.
- Stop if vector expected output is produced only by the same code path under test without an analytic cross-check.

## ROADMAP Traceability

- M2 task 9
- M2 exit 3
- First Backlog item 9
- PROJECT_STATUS.md SHQ Format Stability

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
