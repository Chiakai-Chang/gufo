# Offline conversion toolchain (tools/)

Python-only, torch-free serving. Never a transitive dependency of the server.
Run inside `nix develop` (flake adds torch, transformers, safetensors,
huggingface-hub, numpy, scipy, zstandard).

All model weights, logit dumps, and quantized artifacts live under `artifacts/`
which is gitignored — nothing committed.

## Commands

```bash
# 1. download safetensors + tokenizer to artifacts/source
# 2. validate safetensors, write source manifest
tools/strix-inspect.py --source artifacts/source --revision <sha> \
  --out artifacts/work/source-manifest.json

# 3. capture full-precision teacher logits + perplexity (matched-token)
tools/strix-capture.py --source artifacts/source \
  --suite tools/suites/teacher.json --out artifacts/teacher

# 4. quantize LM linear projections to SHQ4-T16 U4Z G64
tools/strix-quantize.py --source artifacts/source \
  --out artifacts/quant --plan artifacts/work/quantization-plan.json

# 5. benchmark: candidate-vs-teacher quality + prefill/decode speed
tools/strix-bench.py --source artifacts/source --quant artifacts/quant \
  --suite tools/suites/teacher.json --teacher-artifact artifacts/teacher
```

## Modules

- `strix/safetensors.py` — safetensors validation + lazy read (Source Contract)
- `strix/shq.py` — SHQ4-T16 / SHQ8-T16 quantize/pack/dequant (normative contract)
- `strix/conformance.py` — byte-exact conformance vectors (T1)
- `strix/manifest.py` — source-manifest / quantization-plan writers
- `strix/model.py` — Qwen3.5 teacher/candidate load + logit extraction
- `strix/quality.py` — KL, perplexity, top-k agreement
- `strix-capture.py` — teacher logit artifact (chunked zstd)
- `strix-quantize.py` — deterministic conversion
- `strix-bench.py` — correctness-linked benchmark

## Conformance tests

```bash
python3 -c "import sys; sys.path.insert(0,'tools'); from strix.conformance import run_all; run_all()"
```

## Layout contract (SHQ4-T16)

See docs/QUANTIZATION.md. Byte layout: `qweight[n_tile][k_group][k16][lane=16][k_pair=8]`,
scales/zeros BF16 + packed UINT4 per (tile,group,lane). U4Z dequant:
`s * (q - z)`, scale rounded to BF16 RNE-ties-even before code selection.
Deterministic v1 scale search is per-channel-per-group min/max range.
