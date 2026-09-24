# Vendored llama.cpp Jinja engine

Copied from the ROCmFPX fork of llama.cpp (`pwilkin-strix-halo-sync`, commit `d08094170`):
`common/jinja/*`, `common/json.{h,cpp}`, `common/unicode.{h,cpp}` and `vendor/nlohmann/json{,_fwd}.hpp`.
MIT licensed (llama.cpp: see LICENSE.llama.cpp; nlohmann/json: MIT, header notice retained).

Local change: `json.cpp` uses `assert` instead of `GGML_ASSERT` so it builds without ggml.
Used by `src/models/qwen/jinja_chat.cpp` for `--chat-template-file`.
