# Development

Use Nix on Linux x86-64 gfx1151. Direct host builds are unsupported.

| Directory | Ownership |
| --- | --- |
| `src/cli` | Executable commands and HTTP adapters |
| `src/core` | Shared runtime, sampling, formats and diagnostics |
| `src/models/<model>` | Model implementation and kernels |
| `src/eval` | HTTP capability evaluation |
| `tests` | Maintained fixtures and focused checks |
| `tools` | Offline evaluation, profiling and microbenchmarks |
| `docs/models/<model>` | Usage, benchmark, evaluation and experiment docs |
| `.devops/nix` | Production package and server configuration |

Stage new files before Nix builds; the flake includes tracked source only.
Production performance comes from `nix build` / `result/bin`, not test binaries.

```sh
nix build
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <affected-target>
nix develop -c ctest --preset gpu-fast -R <affected-check> --output-on-failure
```

`gpu-test` uses RelWithDebInfo with assertions. `cpu-test` provides host checks;
`cpu-sanitizer` enables ASan/UBSan. `gpu-full` selects the complete GPU tree.
Keep slow external-model checks explicit. See [testing](TESTING.md),
[performance tooling](PERFORMANCE.md) and [model quality contracts](models/README.md).

The production package includes only the HIP backend. Experimental accelerator
work remains on `perf/qwen27b-dflash2-npu`; offline quantization research remains
in PR #234. Neither toolchain is required to install or run main.
