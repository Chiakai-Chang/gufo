---
name: optimize-kernel
description: Use when working an optimization card (gh issue opt-*) on Strix Halo: assess a baseline, introduce a change, verify quality against the baseline, and retain or reject with evidence.
metadata:
  origin: strix-halo.cpp
---

# Optimize Kernel (Agent Skill)

Workflow for GPU kernel optimization cards (e.g. gh issue #114 "opt-c010-qk-rope-kv").
Rejection with evidence is a valid card completion. Never promote a fusion that
degrades end-to-end layer time or spills scratch.

## Task scratch file

Keep an evidence file at repo root (e.g. `task-on-going.md`) for the duration
of the card; it is the only artifact that survives context. Record baselines,
profile findings, tool gotchas, and ideas as they are discovered. DELETE it
before pushing - it only makes sense during the optimization.

## 1. Assess the baseline (before any change)

1. Clean stage state: `git add .` (Nix sees only tracked files).
2. Probe hardware + print baseline revision: `./result/bin/strix`.
3. Build:
   - Fast iteration: `nix develop -c cmake --build --preset hip-test`
   - Canonical: `nix build` (produces `result/`).
   Build often. Run a build (fast iteration first) after EVERY step that
   edits kernels or launchers; never batch several uncommitted kernel edits
   and only then compile - accumulating broken code between builds turns one
   small brace mistake into a cascade of confusing secondary errors.
4. Benchmark with `./result/bin/strix-server bench`. Do NOT pass
   `--repetitions` (single run per case is the contract; alternate
   baseline/candidate runs).
   - Decode: `--n-prompt 0 --n-gen 128`
   - Prefill: `--n-prompt 2048 --n-gen 0`
   - Depth scaling: `--n-prompt 2048 --n-depth 0,4096,8192,16384`
     plus `--n-gen 128 --n-depth 0,4096,8192,16384` for tg. Use powers of
     2; for larger contexts keep the same pp value with deeper blocks.
     Read `src/server/bench_cli.cpp` to interpret values before trusting
     them.
5. Logits comparison (correctness before performance counts):
   - `--validate-prefill 1024 --n-prompt 1024 --n-gen 0` compares batched vs
     sequential logits over the full vocabulary (top-1 parity, RMSE, cosine).
     Record the envelope; a fused kernel must not regress it.
6. Profiling (separate pass; tracing changes timing so never profile the
   headline run):
   ```
   nix develop -c rocprofv3 \
     --kernel-trace --marker-trace --scratch-memory-trace --stats --summary \
     --output-directory /tmp/strix-profile \
     -- ./result/bin/strix-server bench --model "$MODEL" --n-prompt 0 --n-gen 128
   ```
   From the report collect per-kernel: launch count, elapsed time, memory
   traffic, and resource table (VGPR, LDS, scratch, occupancy, waves/SIMD).
   Headline latency always comes from an unprofiled run.

Record the baseline artifact (fingerprint, revision, raw samples, register
allocation, scratch, occupancy) before touching code.

## 2. Introduce the change

- One fused kernel / one route at a time; keep the unfused path runnable as
  the independent reference (comparison and revert).
- Reuse existing kernel patterns in `src/core/hip/` (and llama.cpp CUDA
  kernels as reference sources). When a change targets one model, copy-paste
  the pattern into that model's own implementation; do not modularize shared
  code. Only the model being optimized may change.
- Kernel must sustain >= 4 waves per SIMD and no scratch spilling to be
  retained per card contract; check `__launch_bounds__` and resource usage.
- When editing `__global__` kernels, re-balance braces before compiling:
  count `{`/`}` per line skipping `//` and `#` lines. A stray `}` closing
  the namespace early will surface as `use of undeclared identifier
  '<KernelName>'` at the launch site (secondary cascade, not the real bug);
  a missing `}` surfaces as `function definition is not allowed here` or
  `extraneous closing brace`. Fix the FIRST structural error and rebuild
  before chasing later ones.
- Register tests under the card's CTest label:
  `ctest -L 'opt-c010-qk-rope-kv'`.
- Record new ideas in `task-on-going.md` as they are discovered (and the
  model's `benchmarks/<model>/README.md` `# TODOs` dotted list at the end).

## 3. Verify against the baseline

1. Correctness gate first:
   ```
   nix develop -c cmake --build --preset hip-test
   nix develop -c ctest --preset hip-test --output-on-failure --no-tests=error -L 'opt-c010-qk-rope-kv'
   ```
   Fused output must match the unfused reference under the arithmetic
   contract (CPU oracle vs GPU kernel).
2. Logits: same `--validate-prefill` as baseline; require finite logits,
   identical top-1, no material envelope regression.
3. End-to-end with the exact same bench matrix as the baseline (pp, tg, and
   depth scaling); report medians, tail, and raw samples.
4. rocprofv3 with the same command as baseline; compare launch count,
   eliminated memory traffic, layer latency, VGPR/LDS/scratch, occupancy,
   waves/SIMD, and raw repetitions against the recorded baseline.
5. Full gates ONLY for the retained candidate AND only before pushing:
   - `nix build .#checks.x86_64-linux.pr`
   - `nix flake check`
6. Decision: retain only if end-to-end evidence improves under the acceptance
   contract; otherwise revert the fusion with evidence. Document the winning
   route (or the rejection) in the model's `benchmarks/<model>/README.md`,
   updating existing sections only; new ideas go in the `# TODOs` dotted list.

## Commands that worked (append as you validate)

- Baseline bench (no reps): see section 1.4.
- rocprofv3 invocation: see section 1.6.
- rocprof sqlite aggregate (decode stats), devshell python3:
  ```python
  import sqlite3
  c = sqlite3.connect('/tmp/<prof>/homelab/*.db')
  kd = [t for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if 'kernel_dispatch' in t][0]
  ks = [t for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if 'kernel_symbol' in t][0]
  q = f"SELECT ks.kernel_name, COUNT(*), SUM(kd.end-kd.start)/1000.0, AVG(kd.end-kd.start)/1000.0, "\
      f"MAX(ks.arch_vgpr_count), MAX(ks.sgpr_count), MAX(ks.group_segment_size), "\
      f"MAX(ks.private_segment_size), MAX(kd.workgroup_size_x), MAX(kd.grid_size_x) "\
      f"FROM {kd} kd JOIN {ks} ks ON kd.kernel_id=ks.id GROUP BY ks.kernel_name ORDER BY 3 DESC"
  for r in c.execute(q): print(r)
  ```