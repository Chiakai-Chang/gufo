---
id: M003-C012
title: "Commit the capability suite, provenance, and extraction fixtures"
milestone: M003
status: planned
dependencies: [M001-C008]
---

# M003-C012: Commit the capability suite, provenance, and extraction fixtures

## Dependencies

- [M001-C008](../001-repository-toolchain/008-single-pr-test-entrypoint.md)

## Required Context

- docs/EVAL.md defines 75 committed GPQA/SuperGPQA/AIME cases plus 17 opt-in uncommitted COMPSEC cases, pure extractors, provenance, and self-tests. No runtime generation is required in Milestone 2.

## Goal

Materialize the redistributable capability cases, audited provenance, pure text answer extractors, fixture-driven scorer tests, and suite hashing while keeping COMPSEC opt-in content out of the default commit.

## Non-Goals

- Do not implement model generation, interactive UI, or strix-eval runtime integration.
- Do not report official benchmark scores.
- Do not commit private/raw COMPSEC content or tune quantization against this held-out suite.

## Expected Paths

- `tests/quality/gpqa.json`
- `tests/quality/supergpqa.json`
- `tests/quality/aime2025.json`
- `tests/quality/provenance.json`
- `tests/quality/extractor-fixtures.json`
- `tools/strix/eval_extract.py`
- `tests/tools/test_eval_extract.py`
- `.gitignore`
- `docs/EVAL.md`
- `docs/LICENSING.md`

## Definition of Done

- [ ] Exactly 25 audited cases from each committed suite are present and interleavable in the documented first-75 progressive order; case IDs and suite hash are stable.
- [ ] Every row names upstream dataset, immutable revision/content hash, row ID, license, answer audit, and replacement reason where applicable.
- [ ] MCQ, integer, and linespec extractors are pure text-in/verdict-out and pass ambiguity, wrapped-answer, separators/currency, ranges, no-answer, and adversarial phrasing fixtures.
- [ ] COMPSEC path is ignored/opt-in by default and absence yields SKIPPED rather than corrupting the committed suite hash.
- [ ] A license/provenance validator fails missing or incompatible records; calibration, teacher, and capability suite hashes are checked for declared disjoint identities.

## Development Loop

```sh
nix develop -c python3 -m unittest -v tests.tools.test_eval_extract
nix develop -c env PYTHONPATH=tools python3 -m strix.eval_extract --self-test-extractors
```

## Output Artifacts

- 75 committed capability cases
- Provenance/license manifest
- Extractor golden fixtures
- Stable suite hash

## Stop Conditions

- Stop and quarantine any case with an unverifiable key, missing figure, ambiguous prompt, or unresolved redistribution status.
- Stop if capability prompts overlap calibration or quantizer-search material.
- Stop before runtime generation or baseline score creation.

## ROADMAP Traceability

- M2 task 10
- EVAL.md Suite Set
- EVAL.md Provenance
- TESTING.md T5

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
