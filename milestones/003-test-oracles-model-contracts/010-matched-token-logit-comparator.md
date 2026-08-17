---
id: M003-C010
title: "Implement validated matched-token full-logit comparison"
milestone: M003
status: planned
dependencies: [M003-C009]
---

# M003-C010: Implement validated matched-token full-logit comparison

## Dependencies

- [M003-C009](009-versioned-teacher-logit-capture.md)

## Required Context

- tools/strix/quality.py and tools/strix-bench.py provide reusable KL/perplexity/top-k calculations. Comparison must first prove artifact compatibility and then report the full required metric set.

## Goal

Add a machine-readable comparison runner over versioned teacher/candidate artifacts that enforces identical token history, positions, vocabulary identity, and declared numerical contracts before calculating quality metrics.

## Non-Goals

- Do not free-run either model.
- Do not define universal release thresholds.
- Do not treat matching top-1 text as sufficient.

## Expected Paths

- `tools/strix/quality.py`
- `tools/strix-compare.py`
- `tools/strix-bench.py`
- `tests/tools/test_logit_compare.py`
- `tests/fixtures/logits/tiny-candidate/`
- `docs/BENCHMARKS.md`

## Definition of Done

- [ ] Runner rejects mismatched schema, model/source, tokenizer/template, vocabulary/order, token stream, positions, suite, shape, dtype contract, checksum, and non-finite inputs before comparison.
- [ ] Report includes mean/median/p95/p99/p99.9/max KL, NLL/perplexity, top-1, top-5 and configurable top-k overlap, teacher-top-1 candidate probability/rank displacement, normalized-log-probability max/RMS error, raw-logit diagnostics, and non-finite counts.
- [ ] Stable log-softmax/KL avoids log(0) NaNs and has analytic identical/constant-shift/extreme-logit tests.
- [ ] CLI emits JSON and exits nonzero on validation or a supplied declared gate failure; existing strix-bench delegates comparison math rather than duplicating it.

## Development Loop

```sh
nix develop -c python3 -m unittest -v tests.tools.test_logit_compare
nix develop -c bash -lc 'tools/strix-compare.py --help >/dev/null && tools/strix-bench.py --help >/dev/null'
```

## Output Artifacts

- Tiny matched teacher/candidate fixture pair
- Machine-readable comparison report golden

## Stop Conditions

- Stop if mismatched vocabularies can be compared positionally.
- Stop if non-finite logits are averaged away.
- Stop if a tolerance is introduced without a declared numerical contract.

## ROADMAP Traceability

- M2 task 8
- M2 exit 4
- TESTING.md Metrics
- BENCHMARKS.md matched-token methodology

## Agent Handoff

When complete, report:

- Files changed.
- Exact commands run and their results.
- Produced artifact paths and hashes where applicable.
- Any skipped hardware checks and why.
- Residual risks or follow-up cards without expanding this card's scope.
