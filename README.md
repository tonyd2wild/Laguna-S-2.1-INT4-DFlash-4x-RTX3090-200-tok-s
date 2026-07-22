# Laguna-S-2.1 INT4 + DFlash on 4x RTX 3090 — a levers menu (private, WIP)

poolside **Laguna-S-2.1-INT4** (~8B-active WNA16 MoE, 15 shards ~77 GB, native 256K ctx,
hybrid attention: **12 full + 36 sliding(w=512) layers**) with the **DFlash-INT4** speculator,
on a quad-RTX-3090 box (24 GB each, NVLink pairs 0↔1 NV2 / 2↔3 NV1, PCIe between pairs, TP4),
vLLM v0.25.1.

This repo is a **menu of measured configs** — pick your lever. Every number below was measured
on this box (client-wall tok/s, 400-token code gens unless noted).

## The levers table (all measured 2026-07-22)

| build | flags delta | ctx | KV pool | decode tok/s | notes |
|---|---|---|---|---|---|
| baseline (Kai handoff) | eager, gmu .90, k15 | 100K | 294K | **0 — crashed on 1st gen** | DFlash 394 MB buffer OOM |
| first serve | eager, gmu .85, k15 | 32K | 143,126 | 37.5 | accept 3.32, draft accept 15% |
| k7 | eager, gmu .85, k7 | 32K | 143,126 | 39.8-45.8 | accept 3.13, draft accept 30% |
| graphs | +CUDA graphs, gmu .88, k7 | 32K | 85,775 | 85-158 | PIECEWISE (spec forces it) |
| former SPEED | graphs, gmu .88, k7 | 80K | 98,827 | 113-245 (HTML ~200 warm) | superseded |
| context (no spec) | drop DFlash, FULL graphs | 80K | 142,220 | 86-90 | matches community 86 t/s |
| reclaim @ 0.88 (greedy) | estimate-off env | 80K | 231,943 | **CRASH** | illegal access at 1st gen |
| reclaim @ 0.85 | estimate-off env | 80K | 157,945 | **CRASH** | OOM at graph capture |
| big-ctx mode | no DFlash, FULL graphs, gmu .90 | 128K | 224,858 | 87-89 | measured, works |
| Will-reference | his exact flags (no ctx cap, seqs 8, defaults→gmu .92) | 256K | **390,566** | 88.9 | reproduces his 425K claim |
| round-2 champion | DFlash k7, graphs, seqs 8, gmu .88 | 128K | 160,335 | 150-260 | superseded by seqs 4 + 2K chunks |
| seqs-4 probe | DFlash k7, graphs, seqs 4, gmu .88, batch 4K | 128K | 193,917 | generation-gated | graph reserve 0.74→0.43 GiB |
| 196K edge | DFlash k7, graphs, seqs 4, gmu .88, batch 4K | 192K | 198,273 | **CRASH @ 180K prefill** | KV admitted it; Marlin workspace OOM |
| **CHAMPION** | DFlash k7, graphs, **seqs 4**, **batch 2K**, **gmu .87** | **176K** | **212,112** | **263.1** | 160,043-token prompt + generation passed |

## The 425K mystery (solved, with credit to @hampsonw)

A community post showed "425,799 tokens at bf16" on the same model/hardware class. Panic
ensued. Resolution, measured: (1) the printed pool **scales with max-model-len** in hybrid
models (sliding layers amortize over the denominator) — his 256K default vs our capped
runs; (2) his lean workspace (`--max-num-seqs 8`, defaults) frees ~1 GiB vs our config;
(3) "at bf16" was a mislabel — this checkpoint bakes fp8 KV in (see below). His exact flags
on our box: 390,566 @ 256K. No magic, no missing memory — and his `max-num-seqs` tip is
what pushed the DFlash build to 128K. Good exchange.

## Current champion: speed + context together

```text
--max-model-len 180224 --gpu-memory-utilization 0.87 \
--max-num-seqs 4 --max-num-batched-tokens 2048 \
--speculative-config '{"model":"/root/.cache/huggingface/Laguna-S-2.1-DFlash-INT4","num_speculative_tokens":7,"method":"dflash"}'
```

Measured boot: **212,112-token KV pool (1.18x)**. Measured HTML decode: **500 tokens in
1.901 s = 263.1 tok/s** client-wall. Long-context gate: a **160,043-token prompt** plus
27 generated tokens completed in 59.45 s. This is the new balanced lever: 48K more context
than round 2 while retaining the DFlash speed path.

## The four big discoveries (each one killed a wrong theory)

1. **DFlash's OOM was never "context too big."** The draft allocates a ~394 MB runtime buffer
   at FIRST REQUEST. gmu 0.90 leaves ~100 MB free → serves fine, dies on generation. Ceiling
   with DFlash ≈ gmu 0.88.
2. **k=15 is a trap on this stack.** Acceptance collapses after position ~5; k=7 doubled draft
   efficiency at identical accepted length (+22% decode).
3. **fp8 KV is ALWAYS ON — there is no fp8 lever.** The INT4 checkpoint ships a calibrated
   `kv_cache_scheme` (8-bit float, tensor) that vLLM auto-activates under `--kv-cache-dtype auto`,
   for target AND draft. Proven integer-exactly: the 98,827-token pool only fits fp8 page math
   (a bf16 pool would need 2.03 GiB; only 0.96 GiB was available). Passing `--kv-cache-dtype fp8`
   is a no-op; passing `bfloat16` would HALVE the pool. Two smart-sounding theories (draft-page
   padding; engine-wide silent fp8 drop) both died against the allocator arithmetic.
4. **KV admission is not the final long-context gate.** A 196K build booted with a 198,273
   pool and passed a short generation, then a fresh ~180K prompt OOMed in Marlin's MoE
   `aten::new_empty` workspace with only 2-14 MiB free. Reducing prefill chunks 4096→2048
   and gmu .88→.87 made 176K stable while increasing the measured pool to 212,112.

## Where the memory actually goes (per 24 GB card)

Weights are 17.53 GiB per worker including the draft. With DFlash, FULL_AND_PIECEWISE is
resolved to PIECEWISE. At seqs 8, the graph estimate was 0.74 GiB and available KV was
1.49 GiB. At seqs 4, the graph estimate fell to 0.43 GiB (0.36 actual). The champion's
2K prefill chunks plus gmu .87 leave **1.88 GiB available KV**. Do not disable the graph
estimate: both estimate-off attempts crashed.

## Why the pool ≠ simple math: hybrid accounting

vLLM's hybrid KV manager charges sliding layers only ~289 blocks/request (window-bounded)
vs 5,120 for full layers. The DFlash draft's 6 layers are deliberately full-context
(`sliding_window = None` override — load-bearing: DFlash inserts verifier K/V at absolute
slots; do NOT "fix" it). Never pass `--disable-hybrid-kv-cache-manager` (pool → ~41K).

## Config landmines (from the bring-up, all cost real hours)

1. Image entrypoint is `vllm serve` — model path only, no leading `serve`
2. Set scheduling explicitly. The champion uses `--max-num-seqs 4` and
   `--max-num-batched-tokens 2048`; larger values consume graph/prefill workspace.
3. `--moe-backend marlin` (WNA16; official recipe's triton note is about DeepGEMM+DFlash)
4. `--disable-custom-all-reduce` (PCIe cross-pair)
5. A stale `/home/tony/vllm-watchdog.sh` on the box points at a different model and has
   restarted the container once — disable it before trusting uptime
6. Hand-carried weights: sha256 every shard vs HF's git-LFS oids (size checks pass on
   corrupt shards; 5 shards were silently wrong once)

## Credits
Kai (bring-up: the 4 config fixes, shard-integrity forensics) · poolside (Laguna + DFlash,
calibrated fp8 KV scales in-checkpoint) · vLLM team ([official recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1),
PRs #43081/#45181/#48787 — the DFlash+fp8 lineage) · **Will / @hampsonw** (86 t/s base,
425,799-token pool datapoint, and the `max-num-seqs` lever that unlocked rounds 2-3).
