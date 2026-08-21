# MiniMax H3 issue 169 evidence

This directory records one correctness/profiling evidence run for the
production-shape BF16 block-0 baseline on the supported integrated `gfx1151`
Radeon 8060S. It is not a performance claim.

The profiled test used the operator-owned 528-row oracle described by
`tests/fixtures/minimax_h3/dit-block0-oracle-v1.json`. The test executes the
block twice to prove byte-repeatability; session construction also executes
four paired GEMMs to reject a changed or nondeterministic rocBLAS solution.
Those setup calls are therefore present in the raw aggregate counts.

Files:

- `profile-kernel-stats.csv`: rocprofiler-sdk aggregate kernel dispatch data.
- `isa-summary.txt`: compiler/resource identity for the model-private HIP
  kernels and the SHA-256 of the complete generated assembly retained during
  development.

The complete kernel trace and generated assembly are deliberately not tracked.
Their development SHA-256 values are recorded in `isa-summary.txt`.

Reproduction:

```sh
STRIX_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
STRIX_H3_DIT_GOLDEN=/var/llms/huggingface/strix-h3-oracles/dit-block0-528-v1 \
nix develop -c rocprofv3 --kernel-trace --stats -- \
  ./build-h3-169/minimax_h3_dit_hip_test --real
```
