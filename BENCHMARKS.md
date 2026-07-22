# Measured benchmarks — Laguna-S-2.1-INT4 + DFlash, 4x RTX 3090, vLLM v0.25.1

All decode numbers are client-side wall clock (curl → usage.completion_tokens / elapsed),
400-token code generations at temp 0.3 unless noted. KV pools from engine boot logs.
fp8 KV active in ALL rows (checkpoint kv_cache_scheme auto-activates it; see README #3).

## Decode ladder (chronological)

| config | run 1 | run 2 | run 3 | accept len | draft accept |
|---|---|---|---|---|---|
| eager .85 k15 @32K | 22.8* | 22.9* | 25.7* | 3.32 | 15.4% |
| (same, code prompt) | 37.5 | — | — | 3.32 | 15.4% |
| eager .85 k7 @32K | 39.8 | 43.2 | 45.8 | 3.13 | 30.4% |
| graphs .88 k7 @32K | 85.3 | 157.6 | 147.7 | 3.18 | — |
| graphs .88 k7 @80K (SPEED) | 70.5 | 244.9 | 113.2 | ~3.2 | — |
| + explicit fp8 flag (no-op) | 110.2 | 249.0 | 146.4 | — | — |
| no spec, FULL graphs, @80K | 86.2 | 88.4 | 90.2 | n/a | n/a |
| **seqs 4, batch 2K, gmu .87 @176K** | **263.1**† | — | — | — | — |

*mixed-content chat runs (temp 0.7, 512 tok) — everything else is code prompts.
HTML generation on SPEED build: 108.9 cold, ~200 user-reported warm.

†500-token HTML prompt, 1.901 s client-wall. The 176K build also passed a fresh
160,043-token prompt plus 27 generated tokens in 59.45 s.

## KV pools (engine `GPU KV cache size`)

| config | ctx | pool tokens | concurrency |
|---|---|---|---|
| eager .85 (any k) | 32K | 143,126 | 4.37x |
| graphs .88 k7 | 32K | 85,775 | 2.62x |
| graphs .88 k7 | 80K | 98,827 | 1.21x |
| graphs .88 no-spec | 80K | 142,220 | 1.74x |
| Kai's eager .90 k15 (crashed on gen) | 100K | 294,183 | 2.87x |
| computed: profiler-reserve reclaimed | 80K+ | ~234,900 | — |
| graphs .88 k7, seqs 4 | 128K | 193,917 | 1.48x |
| graphs .88 k7, seqs 4 (prefill OOM) | 192K | 198,273 | 1.01x |
| **graphs .87 k7, seqs 4, batch 2K** | **176K** | **212,112** | **1.18x** |

## Failure catalog (equally load-bearing)

| attempt | failure | root cause |
|---|---|---|
| gmu 0.90 eager + DFlash | first generation → engine death | 394 MB draft buffer vs ~100 MB free |
| graphs @ gmu 0.85 | boot: 0.25 GiB KV < 0.36 needed | graph profiler reserve inside gmu |
| k15 | 60% of draft work wasted | per-position acceptance dies after ~5 |
| `--kv-cache-dtype fp8` | byte-identical pool | fp8 already on (checkpoint scheme) |
| 196K, seqs 4, batch 4K, gmu .88 | fresh ~180K prompt → engine death | KV admission passed, but Marlin MoE workspace needed 24 MiB with only 2-14 MiB free |
| `int4_per_token_head`, 256K, seqs 4 | engine initialization failed | Laguna cache shape rejected: `shape '[64, 2, 16, 2, 68]' is invalid for input of size 540672` |

## Reference points
- Community (X, 2026-07-21): 86 t/s c1 base no-spec, pool "425,799 at bf16", power-capped
  cards. Our no-spec run matches the 86. The pool gap is memory budget, not allocator magic.
- vLLM upstream DFlash+fp8 validation (4090/Ada): accept len 8.23, GSM8K 0.844.

## Round 2 (2026-07-22, after @hampsonw shared his flags)

| config | ctx | pool | speed | notes |
|---|---|---|---|---|
| reclaim env @ .88 + DFlash | 80K | 231,943 | CRASH | illegal memory access, 1st gen |
| reclaim env @ .85 + DFlash | 80K | 157,945 | CRASH | OOM at graph capture — the profiler reserve is load-bearing |
| big-ctx (no spec, FULL graphs, .90) | 128K | 224,858 | 87.3 / 89.2 | flat, content-independent |
| Will-exact (no ctx cap, seqs 8, image defaults gmu .92) | 256K | 390,566 (1.49x) | 88.9 | reproduces the 425K post ± version noise |
| **DFlash k7 + graphs + seqs 8 @ .88** | **128K** | **160,335 (1.22x)** | **149.9 / 259.7** | **champion** — max-num-seqs 16→8 freed the workspace |

Pool-formula insight (verified in v0.25.1 source): printed pool = f(memory, max-model-len);
sliding layers charge ~289 blocks/request regardless of ctx, so bigger max-model-len prints
bigger pools from the same GiB. Cross-config pool comparisons are meaningless without
matching max-model-len.

## Round 3 (2026-07-22, workspace + long-prompt gates)

| config | ctx | pool | measured result | verdict |
|---|---:|---:|---|---|
| DFlash k7, seqs 4, batch 4K, gmu .88 | 128K | 193,917 (1.48x) | 300-token generation passed; graph estimate 0.43 GiB | seqs 8→4 recovers 0.31 GiB |
| DFlash k7, seqs 4, batch 4K, gmu .88 | 192K | 198,273 (1.01x) | short generation passed; fresh ~180K prompt OOMed | boot pool is not sufficient validation |
| **DFlash k7, seqs 4, batch 2K, gmu .87** | **176K** | **212,112 (1.18x)** | **160,043 prompt + 27 output passed; HTML 263.1 tok/s** | **new champion** |

The 196K failure was a real CUDA OOM in `moe_wna16_marlin_gemm` / `aten::new_empty`:
24 MiB requested with 2-14 MiB physically free, despite ~444 MiB reserved but unusable.
The stable build adds explicit runtime margin and halves the largest prefill chunk.

Next KV experiment: `--kv-cache-dtype turboquant_k8v4`, following Will's note that
the key and value caches do not need the same precision (8-bit K / 4-bit V). This is
an experiment target, not a validated configuration.
