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

*mixed-content chat runs (temp 0.7, 512 tok) — everything else is code prompts.
HTML generation on SPEED build: 108.9 cold, ~200 user-reported warm.

## KV pools (engine `GPU KV cache size`)

| config | ctx | pool tokens | concurrency |
|---|---|---|---|
| eager .85 (any k) | 32K | 143,126 | 4.37x |
| graphs .88 k7 | 32K | 85,775 | 2.62x |
| graphs .88 k7 | 80K | 98,827 | 1.21x |
| graphs .88 no-spec | 80K | 142,220 | 1.74x |
| Kai's eager .90 k15 (crashed on gen) | 100K | 294,183 | 2.87x |
| computed: profiler-reserve reclaimed | 80K+ | ~234,900 | — |

## Failure catalog (equally load-bearing)

| attempt | failure | root cause |
|---|---|---|
| gmu 0.90 eager + DFlash | first generation → engine death | 394 MB draft buffer vs ~100 MB free |
| graphs @ gmu 0.85 | boot: 0.25 GiB KV < 0.36 needed | graph profiler reserve inside gmu |
| k15 | 60% of draft work wasted | per-position acceptance dies after ~5 |
| `--kv-cache-dtype fp8` | byte-identical pool | fp8 already on (checkpoint scheme) |

## Reference points
- Community (X, 2026-07-21): 86 t/s c1 base no-spec, pool "425,799 at bf16", power-capped
  cards. Our no-spec run matches the 86. The pool gap is memory budget, not allocator magic.
- vLLM upstream DFlash+fp8 validation (4090/Ada): accept len 8.23, GSM8K 0.844.
