# AGENTS.md

## Build (Nix)

Build with Nix only. No Makefile.

```sh
nix build           # build
./result/bin/strix  # run
nix develop         # dev shell
```

- `git add` before `nix build` — Nix sees only tracked files.
