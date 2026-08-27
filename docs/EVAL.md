# Capability Evaluation

## Purpose

This document defines the capability evaluation layer: free-running,
answer-graded evaluation against a fixed suite of knowledge, reasoning, and
security-localization questions. It fills the "task suites" and "task gates"
slots referenced by TESTING.md and QUANTIZATION.md.

Teacher-forced metrics (BENCHMARKS.md) measure intrinsic distributional error.
They cannot see a quantized or routing change that keeps matched-token KL low
while breaking multi-step reasoning, long-horizon generation, or chat-template
rendering. Capability evaluation catches that class of regression.

This suite is a regression harness, not a leaderboard runner. Results must
never be reported as official GPQA, SuperGPQA, AIME, or COMPSEC scores. The
embedded cases are a small fixed subset chosen to make local regression
testing useful and inspectable.

A bad target is worse than a merely hard target. Every suite row is either
verified correct or replaced; rows are not locally re-keyed.

## Scope and Boundaries

This document owns:

- Free-running generation for evaluation.
- Answer extraction and grading.
- Suite selection, curation, and audit records.
- Trace format and offline regrade.
- Task-gate policy.

It does not own:

- Teacher-forced quality metrics: BENCHMARKS.md.
- Threshold baselining, budgets, and release gates: TESTING.md, "Quality
  Thresholds".
- Sampling semantics, RNG, and stop conditions: TOKENIZATION.md.
- Context capacity and KV budgeting: KV_CACHE.md.
- Runtime request APIs: SERVER.md and CLI.md.
- Throughput and latency objectives: PERFORMANCE.md.

## Suite Set

The initial suite is a 92-item set in progressive order: early cases are
useful smoke tests, later cases are hard enough that a strong reasoning model
still misses some.

| Suite | Role | Answer form | License | Commit policy |
| --- | --- | --- | --- | --- |
| GPQA Diamond slice | Graduate-level science, multi-step | Multiple choice | CC BY 4.0 | Committed |
| SuperGPQA slice | Specialist knowledge, domain transfer | Multiple choice | ODC-BY | Committed, audited |
| AIME 2025 slice | Competition mathematics | Integer | MIT-licensed mirror | Committed |
| COMPSEC slice | Single-function C/C++ vulnerability localization | Line or line set | Derived from public CVE writeups | Opt-in, uncommitted |

Initial composition: the first 75 cases are 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 cases interleaved. The final 17 are an audited
COMPSEC slice.

### Provenance

Provenance records identify the upstream source for every row and must name
the dataset, its revision or content hash, the upstream row identifier, the
license, and, for replaced rows, the replacement reason. Canonical upstream
references:

- GPQA Diamond, released under CC BY 4.0.
- SuperGPQA, released under ODC-BY; mostly original data plus a limited
  amount of transformed third-party data.
- AIME 2025, from an MIT-licensed mirror.
- COMPSEC, derived from public CVE writeups; no proprietary content is
  reproduced.

Provenance records are reviewed as a licensing matter, not only as test
data (LICENSING.md).

### Curation Rules

- SuperGPQA is curated rather than blind. Upstream rows with wrong keys,
  missing figures, or underspecified prompts are replaced with cleaner rows
  from the same suite.
- COMPSEC cases are reduced single-function vulnerability-localization
  questions. The model is asked for the single best source line, or the
  smallest exact line set only when the defect cannot be localized to one
  line. CVE anchors and private rationales are not rendered to the model.
- Suite rows must be disjoint from imatrix calibration prompts, quantizer
  search prompts, and the teacher-forced capture corpus. Tuning a model or a
  quantizer against the held-out suite invalidates it as a release gate
  (TESTING.md).
- Adding or replacing rows changes the suite hash and requires re-baselining
  the affected gates.

## Case Format

Cases are machine-readable JSON under the layout declared by TESTING.md,
"Initial Repository Layout":

```text
tests/quality/
  gpqa.json
  supergpqa.json
  aime2025.json
  compsec.json        # opt-in, gitignored by default
  provenance.json     # dataset revisions, hashes, licenses, audit records
```

Each case carries:

```text
id, source, domain, kind, title, question, choices[], answer, provenance
```

`kind` is one of `mcq`, `integer`, or `linespec`. The scorer is pure
text-in, verdict-out: it must run without loading a model or touching the
runtime.

## Generation Contract

- Greedy is the canonical gate policy. Greedy token history is deterministic
  (ROADMAP.md), which is what makes token-count gates meaningful.
- Temperature or seeded-sampling runs are advisory. They must never gate a
  release.
- The default run enables thinking mode where the model supports it, uses a
  default generation budget of 16000 tokens, and applies a soft/hard
  think-close budget cutoff so a model that never closes its thinking block
  still produces a visible, gradeable answer. Budgets and the soft-limit rank
  are declared parameters, not hard-coded constants.
- Prompt rendering must use the canonical chat template from TOKENIZATION.md.
  Rendering regressions (template drift, special-token handling, think-block
  framing) are a primary failure class this suite exists to catch.
- The evaluator sizes the context from the largest selected prompt plus the
  generation budget and refuses runs that would exceed the context capacity
  of the target configuration (KV_CACHE.md). Runs are refused, never
  silently truncated.
- The report must record the close kind for every case: `none`, `natural`,
  `soft`, or `hard`.
- Cases whose answer was produced under a budget close are graded normally
  but tracked separately. Reports present both the raw pass rate and the
  pass rate excluding budget-closed cases.

## Command Surface

The proposed interface is a single evaluation tool:

```text
gufo-eval -m <model> [options]

  --plain               Disable the interactive UI; print the report.
  --cases <n>           Run only the first n cases.
  --tokens <n>          Per-case generation budget.
  --greedy              Canonical deterministic policy (default for gates).
  --temp <f>            Advisory sampling; never gates a release.
  --seed <n>            Advisory sampling seed.
  --trace <path>        Write the per-case trace artifact.
  --regrade-trace <path> Replay the current scorer against a prior trace
                        without loading the model or regenerating tokens.
  --self-test-extractors Run extractor self-tests only; no model required.
```

Interactive mode renders a two-pane view: a case list that follows the
selection cursor, and the current generation. Pause, quit-and-report,
select, and run-selected-next are available. Interactive mode is a
convenience; `--plain` output is the canonical form for CI and for traces.

`--regrade-trace` reports which cases changed, the old and new picked
answers, and a pass/fail summary. It is the required tool for auditing
evaluator changes.

Proposed command interfaces are contracts for later implementation; they do
not imply that the tools already exist.

## Grading

Extractors run before comparison. One extractor per `kind`:

- `mcq`: picks the answer letter. The pick must survive wrapped phrasing such
  as "D, not B"; extraction happens at word boundaries, and the final
  unambiguous pick wins.
- `integer`: normalizes the candidate and expected answers (strips currency
  and thousands separators, "the answer is" framing) and compares the final
  integer expression.
- `linespec`: normalizes a line number or range. A small range is accepted
  only when its lines are equivalent locations for the same defect; the
  accepted range set is part of the audit record.

Rules:

- Extractors are unit-tested with committed fixtures (T0/T1 in TESTING.md).
  `--self-test-extractors` runs without a model and is part of the release QA
  checklist.
- Any extractor change must be followed by a regrade of the most recent
  baseline trace. Cases that flip require a per-case justification or a
  revert of the extractor change.
- Verdicts per case: `PASSED`, `FAILED`, `SKIPPED` (unsupported by the
  configuration), or `STOPPED` (refused by context or budget). Runtime states
  such as pending, prefill, and thinking are trace states, not verdicts.

## Trace and Offline Regrade

Every run writes a trace artifact: one record per case with:

```text
case id, verdict, prompt tokens, generated tokens,
picked answer, correct answer,
close kind, close token index, remaining budget at close, close rank,
route fingerprint (model revision, quantized artifact, backend route)
```

Traces are artifacts. They are never committed except small hand-checkable
fixtures.

## Deterministic Drift Gate

A small fixed gate detects generation drift from any change that touches
sampling, kernels, routing, or rendering:

```text
gufo-eval --plain --cases 4 --tokens 2048 --greedy
```

The baseline table records expected verdict and exact generated-token count
per case, for example:

| Case | Expected verdict | Expected generated tokens | Picked / correct |
| ---: | --- | ---: | --- |
| 1 | PASSED | 2048 | B / B |
| 2 | PASSED | 438 | C / C |
| 3 | PASSED | 666 | 70 / 70 |
| 4 | FAILED | 2048 | A / C |

Under the greedy contract the token counts must match exactly. A drift in
token count is a regression even when the verdict is unchanged. The first
four cases are committed fixtures of the full suite, not a separate set.

## Reporting Rules

Every report records, at minimum:

- Model revision and quantized artifact identity.
- Suite hash and per-suite case counts.
- Sampling policy, budgets, and close-kind counts.
- Route fingerprint and hardware context.
- Pass rate per suite and overall, plus the rate excluding budget-closed
  cases.
- Regrade metadata when a trace is replayed.

Reports are machine-readable JSON with a human summary. Numbers from this
suite describe regression behavior of a specific artifact on a specific
route; they are not portable quality claims.

## Gate Policy

- Phase 1 (advisory): capability evaluation runs in the release suite and
  reports only. No release is blocked on it. This is the operating mode until
  greedy decoding and the first quantized release exist.
- Phase 2 (gated): after baselining per TESTING.md "Quality Thresholds"
  (teacher against itself, first accepted quantized release, repeat-run and
  machine-to-machine noise), each model declares per-suite floors and
  regression budgets. A candidate that violates a floor or budget fails the
  task gate.
- PR placement: drift gate plus extractor self-tests run on every change
  that touches generation. The full suite runs in the release suite, not on
  every pull request.

## Failure Workflow

When a case flips between a baseline and a candidate:

1. Preserve both traces and the suite hash.
2. Regrade both traces with the current scorer to rule out an evaluator
   change.
3. If the scorer changed and the flip is unexplained, revert the scorer.
4. Otherwise minimize to the divergent prompt, token range, or route and add
   a targeted fixture where practical.
5. A verdict flip with no identified cause is a gate failure, not a pass
   with a note.

When a committed case is found to be a bad target:

- Quarantine it with an owner, failure record, and expiry (TESTING.md).
- Replace it with an audited row and record the replacement in provenance.
- Re-baseline affected gates with the new suite hash.

## Non-Goals

This suite does not cover long-form writing quality, judge-model scoring,
multi-turn or agentic behavior, safety evaluation beyond the COMPSEC slice,
or any throughput or latency objective.
