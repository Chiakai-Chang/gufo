# Autonomous MVP Milestone Cards

These card specifications decompose `docs/ROADMAP.md` Milestones 0–4 into
atomic, dependency-ordered work suitable for autonomous AI coding agents.
They define the first native MVP: text-only deterministic greedy terminal
generation for Qwen3.5-0.8B on `gfx1151`, with reusable checked descriptors for
Qwen3.8-27B. Production Qwen3.8-27B validation is a later milestone.

The Markdown files are durable implementation contracts. Live status,
assignment, priority, and review state belong in the GitHub Project.

## Inventory

| Directory | Roadmap milestone | Cards |
| --- | ---: | ---: |
| [M001: Milestone 0: Repository and Toolchain](001-repository-toolchain/README.md) | 0 | 8 |
| [M002: Hardware Diagnostics and Baselines](002-hardware-diagnostics-baselines/README.md) | 1 | 7 |
| [M003: Test Oracles and Model Contracts](003-test-oracles-model-contracts/README.md) | 2 | 13 |
| [M004: Early GPU and NPU Format Feasibility](004-gpu-npu-format-feasibility/README.md) | 3 | 12 |
| [M005: Qwen GPU-Only Vertical Slice](005-qwen-gpu-only-vertical-slice/README.md) | 4 | 22 |
| **Total** | | **62** |

## Agent Execution Contract

1. Work on one card per branch and pull request.
2. Read every path in **Required context** before editing.
3. Confirm every dependency is merged; do not infer missing prerequisite APIs.
4. Implement only the stated goal. Treat non-goals as hard scope boundaries.
5. Add the tests in the same card as the behavior they validate.
6. Use Nix for builds and development commands. Per `AGENTS.md`, stage newly
   created files before `nix build` so they enter the Nix source closure.
7. Never weaken an oracle threshold in the same card that fails that threshold.
8. Keep model weights, full logits, profiler traces, and large benchmark output
   under gitignored `artifacts/`; commit only bounded fixtures and metadata.
9. Hardware tests may skip when hardware is unavailable only when the card says
   so. Required-hardware mode must fail clearly instead of silently skipping.
10. On any stop condition, stop implementation and report the blocker with
    evidence. Do not invent an architectural decision.

## Global Product Constraints

- Linux x86-64 on AMD Strix Halo only.
- No Windows, macOS, or CUDA work.
- Qwen3.5-0.8B is the rapid-iteration MVP model.
- Qwen3.8-27B is the first production model, but full production validation is
  outside these five milestones.
- The first artifact is text-only and excludes the vision encoder.
- SHQ remains a candidate contract until all documented native conformance gates
  pass; cards must not claim portable v1 stability prematurely.
- GPU-only greedy generation is the protected MVP path.
- Milestone 3 NPU and shared-allocation experiments are evidence gathering and
  must not block the GPU MVP when they fail or are slower.
- Preserve existing CLI/executable naming; command topology changes require a
  separate explicit decision.

## Card Schema

Every card contains dependencies, required context, one goal, non-goals,
expected paths, a checkable Definition of Done, exact development-loop commands,
output artifacts, stop conditions, and ROADMAP traceability.
