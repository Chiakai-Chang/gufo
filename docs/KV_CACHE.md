# KV Cache and Session Persistence

Status: design draft, 2026-08-11

## Purpose

The KV subsystem provides:

- Efficient paged attention for active sessions.
- Continuous batching without relocating whole sessions.
- Prefix sharing with copy-on-write ownership.
- Transactional provisional state for speculative decoding.
- Optional dump and restore of cold sessions.

Strix Halo has unusually large unified memory, but limited external memory
bandwidth relative to its capacity. KV design must optimize both capacity and
bytes read per generated token.

## Capacity Model

For a conventional transformer:

```text
KV bytes per token =
    2
  * number_of_layers
  * number_of_kv_heads
  * head_dimension
  * bytes_per_element
```

The factor 2 represents K and V.

The compiled-in model implementation publishes exact cache requirements because
hybrid attention, recurrent layers, compressed latent attention, and
model-specific state may not fit the conventional formula.

Memory budgeting reserves space for:

- Model weights.
- GPU and NPU programs.
- Persistent scratch and graphs.
- Active KV pages.
- Speculative provisional pages.
- Recurrent and hidden state.
- OS and desktop requirements.
- Restore staging.

Do not allocate all remaining memory to KV. The allocator keeps a configurable
emergency and backend-recovery reserve.

## KV Precision

Initial policy:

| Precision | Use |
| --- | --- |
| BF16 | Quality reference and default sensitive path |
| FP16 | Optional model-compatible path |
| INT8 | Capacity and long-context option after quality validation |
| INT4 | Experimental only |

KV quantization is evaluated separately from weight quantization. Required
tests include long-context logits, retrieval, attention distributions, and
generation stability.

Per-head or per-block scales must be laid out for fused attention consumption.
A smaller cache that requires an expensive standalone dequantization pass may
lose end-to-end performance.

## Page Design

Use fixed-size logical token pages. Candidate initial sizes:

```text
16 or 32 tokens per page
```

The final size is selected per model from:

- Attention kernel access patterns.
- Internal fragmentation.
- Prefix-sharing granularity.
- Speculative copy-on-write cost.
- Page-table lookup overhead.
- Dump and restore chunking.

Each page has:

```text
PageId
model/session layout ID
layer or layer-group
token start and valid token count
K/V allocation offsets
dtype and scale metadata
reference count
generation/version
state: FREE, LIVE, PROVISIONAL, COLD, RESTORING
```

Page IDs remain stable across scheduler compaction. Device kernels receive a
page table rather than assuming physically contiguous session storage.

## Physical Layout

The model implementation defines the physical K/V tensor order. General
requirements:

- Contiguous vector access for the selected attention kernel.
- GQA/MQA-aware head grouping.
- Separate single-token and batched attention views only when lossless.
- Aligned page starts.
- No per-step host pointer rebuilding.
- GPU/NPU visibility for routes that consume shared KV.

GPU decode is the initial owner of ordinary KV attention. NPU programs may read
KV for prefill or verification only after shared access and cache ordering are
proven.

## Allocation

Use a page pool created at model load:

- No per-token device allocation.
- Lock-free or low-contention free-page management.
- Per-session page vectors in stable host memory.
- Device-visible page tables updated in batches.
- Separate provisional-page reserve for speculative verification.

Admission reserves a minimum number of pages. Additional pages are acquired as
the context grows. Failure to grow is handled before executing a transition
that would require the unavailable page.

## Prefix Cache

Prefix entries are keyed by:

```text
model revision
quantized artifact ID
tokenizer and chat-template revision
exact token sequence
position/rope configuration
KV dtype and layout
model-specific recurrent-state version
```

Use a radix tree or hashed token chunks to find the longest reusable prefix.

Shared pages are immutable. A session extending a partially shared final page
uses copy-on-write.

Eviction considers:

- Reference count.
- Last access.
- Recompute cost.
- Page count.
- Current memory pressure.
- Persisted snapshot availability.

Prefix caching must be disabled for model configurations whose state cannot be
captured and restored exactly.

## Speculative State

Verification writes provisional rows and pages. The transaction records:

- Root state.
- Candidate topology.
- Per-row provisional KV spans.
- Recurrent and hidden-state snapshots.
- Accepted row.
- Commit and rollback plan.

On commit:

- Promote only the accepted prefix.
- Copy or relink the selected recurrent state.
- Release rejected pages.
- Update positions and page-table generations atomically.

On failure or cancellation, all provisional pages are reclaimed without
changing canonical session state.

## Disk Dump and Restore

Disk persistence is for cold sessions, application suspend/resume, and avoiding
expensive repeated prefill. SSD is not a viable live extension of the token
decode cache.

### Snapshot contents

- Manifest and format version.
- Model and quantization hashes.
- Tokenizer and chat-template hashes.
- Token IDs and logical positions.
- RoPE and context parameters.
- KV dtype, layout ID, page size, and valid lengths.
- K/V page payloads.
- Model-specific recurrent, convolution, or hidden state.
- Speculative-provider state only after a committed boundary.
- Per-chunk checksums.

Never dump an in-flight provisional transaction.

### Compatibility

A snapshot restores only when all required compatibility IDs match. By default
it is not portable across:

- Different model revisions.
- Different quantized weight artifacts.
- Different tokenizer or chat template.
- Different KV dtype or physical layout.
- Different RoPE scaling.
- Model implementations with incompatible recurrent state.

Backend portability may be allowed when GPU and NPU use the same defined KV
layout and both implementations pass restore tests.

### File organization

```text
session-id/
  manifest.json
  tokens.bin
  state.bin
  kv-0000.bin
  kv-0001.bin
  ...
```

Write chunks to temporary names, fsync according to configured durability, then
atomically publish the final manifest. The manifest is the commit record.

### Restore

1. Read and validate the manifest.
2. Reserve required pages and state slots.
3. Read chunks asynchronously into bounded staging buffers.
4. Verify checksums before publication.
5. Convert layout only through an explicitly versioned converter.
6. Publish the complete page table and state at one commit point.
7. Make the session schedulable.

Partial restores are not visible to inference.

### Compression

General-purpose compression may have limited value on BF16/INT8 KV payloads.
Benchmark:

- No compression.
- Fast lossless compression.
- Sparse or model-specific compression where applicable.

Lossy conversion, such as BF16 KV to INT8 for disk, creates a different
snapshot format and requires its own quality evaluation.

## Security and Privacy

Snapshots contain prompt-derived model state and token IDs. Treat them as
sensitive user data.

- Persistence is opt-in.
- Store under restrictive file permissions.
- Support application-provided encryption at rest.
- Do not expose arbitrary snapshot paths through the public OpenAI API.
- Sanitize session IDs before creating paths.
- Permit secure deletion policy where required.

Administrative session APIs should use a separate authenticated namespace.

## Background I/O Policy

Dump and restore use bounded CPU and storage queues. They yield when:

- Interactive decode latency is above target.
- Unified memory bandwidth is saturated.
- Restore staging would reduce the emergency reserve.
- The SSD queue is congested.

Where supported, use direct or asynchronous I/O only after proving that
alignment and buffer pinning do not degrade GPU/NPU access.

## Tests

- Capacity and bytes-per-token calculations.
- Page allocation, release, and generation reuse.
- Copy-on-write prefix extension.
- Prefix radix lookup and hash collision handling.
- Variable context lengths in one physical batch.
- Speculative full, partial, and zero-token commit.
- Cancellation and backend failure rollback.
- Snapshot round trip with exact greedy continuation.
- Corrupted manifest and payload rejection.
- Compatibility mismatch rejection.
- Atomic publication under process interruption.
- Concurrent dump, restore, decode, and eviction.
- Memory accounting returns to baseline after session close.
- Long-context quality for every promoted KV dtype.
