# AGENTS.md

## Supported Platform

Linux x86-64 on AMD Strix Halo (`gfx1151` GPU and XDNA2 NPU) is the only
supported production target.

## Build (Nix)

Build with Nix only. Direct host builds and Makefiles are unsupported.

```sh
nix build           # build
./result/bin/strix  # run
nix develop         # dev shell
```

- `git add` before `nix build` — Nix sees only tracked files.

## Development

- Use `gh` CLI to retrieve and update issues content. Verify if installed and configured with `gh auth status`, and use it unless the user explicitly says otherwise.
