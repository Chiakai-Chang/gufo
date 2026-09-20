# Qwen3.8 Flash-Next experiments

| Experiment | Decision / evidence |
| --- | --- |
| Full-width MTP RMSNorm and split projection | Retained after independent CPU stage audit; one 10240-wide normalization, embedding projection shared across HC branches. |
| Full Q8 vocabulary head | Retained; private Q4 shortlist removed. Sampled top-64 proposals use exact target verification. |
| Batched MTP transformer and heads | Retained; independent body/head comparisons, private KV/recurrent/rollback/RNG state. |
| Batched decode mixer projections | Retained; exact C2/C4/C6/C8 logits, acceptance and RNG; mixed C4 throughput improvement repeated. |
| Q4 shared-expert weight reuse | Retained with a separate compact kernel for single-request experts; faster repetitive C1/C4/C8, no mixed-work regression, exact projection/model replay. |
| Register-cached vision softmax | Retained for bounded row sizes; unchanged reduction order, byte-identical Flash-Next/Qwen27B embeddings, no additional allocation. |
| Ratio-four predictor QSA, FP32 ranking queries | Retained with sparse state, rewind and deep selector checks. |
| Greedy batch cost controller | Retained only for all-greedy C>1; separate occupancy/context bins, stable plain controls, no transition timings. Sampled replay uses fixed cost curves. |
| One-row MTP prefill lag | Retained; 320 KiB kept hidden state/session, avoids replaying a final prefill chunk. |
| Lazy rollback and shared scratch | Retained; depth grows on demand, reset releases it; seven-draft cap about 788 MiB/session. |
| Live final frontier and async prompt snapshots | Retained; immutable branch snapshot, worker capture, bounded persistence outside the lookup lock. |
| Chunk-equivalent projections/attention | Retained with exact continued-image/cache/full-logit gates; one-token tails keep prefill arithmetic. |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |

Separate d32K pp2048 profiling attributes 29.1% of kernel time to MoE, 34.9%
to dense projections and 12.6% to attention/indexing. A C4 mixed MTP trace is
80.7% GPU-busy; MoE/MMQ is 51.3% and dense projections 36.1% of kernel time.
These are profile observations, not unprofiled throughput measurements.

Next: improve prefill at depth and target/draft batch projection reuse while
preserving [quality](EVALUATION.md). The 1700 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).
