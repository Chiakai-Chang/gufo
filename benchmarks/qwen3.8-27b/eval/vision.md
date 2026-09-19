# Qwen3.8 image input

Qwen3.8-27B Q4/Q8 and Flash-Next use their matching **BF16 vision projector**.
Place `mmproj-BF16.gguf` beside the target GGUF or its quantization directory,
or pass `--mmproj PATH`. Vision weights upload only for an image request.
Text generation keeps its existing execution path.

```sh
./result/bin/gufo prompt --model "$MODEL" --image photo.jpg \
  --prompt "Describe this image." --temperature 0
./result/bin/gufo chat --model "$MODEL" --image photo.jpg --temperature 0
./result/bin/gufo serve llm --model "$MODEL" --served-model-name vision-test \
  --sessions 2 --context 4096
```

Repeat `--image` for multiple images. In chat, these belong to the first user
turn and remain in its history. HTTP accepts ordered text and `image_url`
parts in user messages, including multiple images and subsequent turns:

Image numbering follows the official template: off by default. Enable
`Picture N:` prefixes with `--add-vision-id` or HTTP
`chat_template_kwargs: {"add_vision_id": true}`. Numbering continues across
conversation turns; fully rendered message content is trimmed before image
offsets are passed to the encoder.

```python
import base64, json, urllib.request

image = base64.b64encode(open("photo.jpg", "rb").read()).decode()
body = {
    "model": "vision-test", "temperature": 0, "max_tokens": 64,
    "messages": [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64," + image}},
        {"type": "text", "text": "Describe this image."}
    ]}]
}
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request).read().decode())
```

PNG/JPEG data URLs and HTTPS images are supported; `detail` is `auto`.
Limits: 20 MiB encoded, 32 megapixels decoded, 16 images per message.
Images use the official dynamic resolution policy; their expanded tokens
count toward the context limit. Unsupported or invalid input returns an error.
The synthetic `bench` command remains a text benchmark.

DFlash2 on 27B and MTP on Flash-Next use the same target image state as AR.
Native 27B MTP also receives the shifted image embeddings. Request-local
mRoPE positions continue through decoding, verification and rollback.
RAM and disk cache identities include processed pixels, image placement,
preprocessing version and projector contents. Snapshots retain image geometry;
complete prefix hits do not rerun the vision encoder. Adding or replacing an
image can safely miss the prefix cache.

## Quality checks

Shared operators live in `src/models/qwen/vision`; each language runtime owns
its embedding, attention, speculative and snapshot integration.

Qualified: Q4/Q8 AR and DFlash2, Flash-Next AR/MTP, native 27B MTP CLI,
cold and sampled C4, HTTP C2/streaming, multiple images, and RAM/disk replay.
Controls also cover spatial shape recognition, a 2,304-token image crossing
prefill chunks, and Flash-Next image decoding/cache replay at 10,495 tokens.
Image throughput sweeps remain **TODO**.

Known gap: the Q8_K_XL continued-image fixture produces different greedy text
from a fresh full prefill. This reproduces on `bc4c4a5` before the cache cleanup.
Replaying the same prompt/decode history matches the live cache exactly.
AR and DFlash2 can also retain different final decode frontiers, leaving different
prefill suffixes on continuation.
The full vision check retains its cold-versus-live equality gate; `--disk-only`
checks persistence and history replay without claiming that stronger parity.

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen27b_vision_test qwen_vision_serving_test
nix develop -c python3 tools/qwen27b/vision_check.py \
  --directory /tmp/qwen-images \
  --probe build/gpu-test/tests/models/qwen27b/qwen27b_vision_test
build/gpu-test/tests/models/qwen27b/qwen_vision_serving_test \
  "$MODEL" "$DRAFT" /tmp/qwen-images
# Use "-" for AR only. Use --disk-only for a focused persistence check.
nix develop -c python3 tools/qwen27b/vision_check.py \
  --directory /tmp/qwen-images --http-url http://127.0.0.1:8080 \
  --model vision-test
nix develop -c python3 tools/qwen27b/vision_check.py \
  --directory /tmp/qwen-images --binary result/bin/gufo \
  --target "$MODEL" --draft "$DRAFT" --speculative dflash2
# For Flash-Next or native 27B MTP, use --speculative mtp.
```

Use a fresh server for the HTTP check. The native check covers cold C4,
greedy AR/speculative token IDs, proposal accounting, seeded sampled C4,
multiple images, exact prefix replay and disk restoration after restart.
The CPU check compares pixels against Pillow/Torchvision, including resizing,
EXIF orientation, palette, alpha, grayscale, gamma metadata and CMYK JPEG.
Run parser/cache/codec tests under the `cpu-sanitizer` preset when changing
input handling or persistence.

Flash-Next's maintained continuation fixture requires **identical full logits**
for bulk prefill, incremental prefill and restored cache state. Its router and
recurrent-gate projections keep one accumulation order when rows move between
chunks. Prompt projections and normalization retain their arithmetic even for
one-token tails. Prefill retains F16 SSM output activations, and attention
accumulates each query's final partial tile using only visible keys. Tests cover
unaligned boundaries, full-model continuations through 4096 tokens, and FP64 operator
controls.
This does not establish bit-identical output between prefill and token-at-a-time
kernels, which use different arithmetic routes.

For encoder arithmetic, compare the complete graph and isolated layers with
the original Transformers operators using the **same BF16 GGUF weights**:

```sh
build/gpu-test/tests/models/qwen27b/qwen27b_vision_test \
  "$MMPROJ" /tmp/qwen-images/shapes.png /tmp/qwen-trace 5120
# Flash-Next's output width is 2560.
nix develop -c python3 tools/qwen27b/vision_reference.py \
  --source "$TRANSFORMERS_CHECKOUT" --mmproj "$MMPROJ" \
  --image /tmp/qwen-images/shapes.png --trace /tmp/qwen-trace \
  --output /tmp/qwen-vision-reference.json
```

Transformers pin: `3713bd839e580d07e4b70f2c89e986cb3c0e8ddf`.
The tool verifies source hashes, exact pixels, every vision layer, isolated
layers 0/8/26, and output error against a full FP32 control. Both projectors
pass these checks on the focused fixtures. This qualifies operators and
bindings; it does not independently qualify GGUF conversion or the quantized
language model. Keep traces outside the repository.
When changing vision arithmetic or preprocessing, bump the image execution
version in `src/models/qwen/vision/prompt.cpp` to invalidate persisted prefixes.

Audited formulas include patch/merge order, learned position interpolation,
axial vision RoPE, normalization and GELU variants, language mRoPE, Flash
indexer positions, and shifted MTP inputs. Secondary implementation controls:
llama.cpp `18a04f09c24616898792bcfaa17f3550bdc78912` and
vLLM `63d9ad0a3a435cdf3a44495028b10f390a38f960`.
