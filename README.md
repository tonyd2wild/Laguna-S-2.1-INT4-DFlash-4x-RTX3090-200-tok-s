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
| **SPEED (current)** | graphs, gmu .88, k7 | **80K** | 98,827 | **113-245** (HTML ~200 warm) | the daily driver |
| context (no spec) | drop DFlash, FULL graphs | 80K | 142,220 | 86-90 | matches community 86 t/s |
| **MAX (pending gates)** | + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | ~224K | **~234,900 (computed)** | expect 113-245 | the everything-config |

## The three big discoveries (each one killed a wrong theory)

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

## Where the memory actually goes (per 24 GB card, gmu 0.88)

weights ~18 GB · cudagraph-profiler reserve ~1.3 GB · DFlash draft + buffers · ~0.96 GB KV.
The **cudagraph profiler reserve is the recoverable lever**: it sizes for FULL graphs that
spec-decode never uses (PIECEWISE forced). `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
→ KV ~2.24 GB → ~235K-token pool (computed from specs; boot gates pending).

## Why the pool ≠ simple math: hybrid accounting

vLLM's hybrid KV manager charges sliding layers only ~289 blocks/request (window-bounded)
vs 5,120 for full layers. The DFlash draft's 6 layers are deliberately full-context
(`sliding_window = None` override — load-bearing: DFlash inserts verifier K/V at absolute
slots; do NOT "fix" it). Never pass `--disable-hybrid-kv-cache-manager` (pool → ~41K).

## Config landmines (from the bring-up, all cost real hours)

1. Image entrypoint is `vllm serve` — model path only, no leading `serve`
2. Spec decode needs explicit `--max-num-seqs 16 --max-num-batched-tokens 4096`
3. `--moe-backend marlin` (WNA16; official recipe's triton note is about DeepGEMM+DFlash)
4. `--disable-custom-all-reduce` (PCIe cross-pair)
5. A stale `/home/tony/vllm-watchdog.sh` on the box points at a different model and has
   restarted the container once — disable it before trusting uptime
6. Hand-carried weights: sha256 every shard vs HF's git-LFS oids (size checks pass on
   corrupt shards; 5 shards were silently wrong once)

## Credits
Kai (bring-up: the 4 config fixes, shard-integrity forensics) · poolside (Laguna + DFlash,
calibrated fp8 KV scales in-checkpoint) · vLLM team ([official recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1),
PRs #43081/#45181/#48787 — the DFlash+fp8 lineage) · community X post (86 t/s base +
425,799-token pool datapoint that triggered the allocator investigation).
