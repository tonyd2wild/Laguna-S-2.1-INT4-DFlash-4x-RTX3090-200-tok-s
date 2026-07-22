# Experiment history and findings

This file preserves the investigation behind the 200K serving recipe without crowding the
quick-start path. Every numeric result below was observed on the four-RTX-3090 host.

## Starting point

The initial goal was to combine Laguna's DFlash speed with substantially more than 80K context.
The starting DFlash configuration reached 113–245 tok/s but exposed only a 98,827-token printed
KV pool at an 80K context limit. A community report showed a 425,799-token pool and approximately
86 tok/s on nominally similar hardware, raising the question of where the extra cache came from.

## Finding 1: the “425K pool” was not missing VRAM

Will / @hampsonw shared his actual command. It omitted an explicit context cap, speculative
decoding, and memory utilization while setting `--max-num-seqs=8`. On the tested image that meant:

- native 262,144-token model limit;
- no DFlash target/draft overhead;
- image-default GPU memory utilization;
- half the sequence concurrency of the original local recipe.

Running the equivalent command locally produced a **390,566-token pool at 256K and 88.9 tok/s**,
reproducing the report within version/default variance.

Laguna contains 12 global-attention layers and 36 sliding-window layers with a 512-token window.
For vLLM 0.25.1's hybrid KV manager, the reported pool follows the configured per-request block
requirement rather than a context-independent bytes-per-token number:

```text
pool = floor(num_blocks × configured_context / per_request_blocks)

per_request_blocks = full_attention_blocks + window_bounded_sliding_blocks
```

Increasing `max-model-len` changes the ratio because the global layers grow with context while
the sliding layers remain window-bounded. Therefore two configurations with similar KV GiB can
print very different pool-token counts.

## Finding 2: FP8 KV was already active

The INT4 checkpoint contains a calibrated 8-bit floating-point KV-cache scheme. vLLM promotes
`--kv-cache-dtype auto` to FP8 from checkpoint metadata, and the DFlash draft inherits the cache
configuration. Page-allocation arithmetic confirmed the measured pools could not fit as BF16.

Consequences:

- adding `--kv-cache-dtype fp8` did not change the pool;
- forcing BF16 would approximately double KV bytes and reduce capacity;
- the community “at bf16” label was a description error, not a separate large-BF16 result.

## Finding 3: DFlash changes both speed and memory accounting

The DFlash draft contributes six additional full-context cache layers and allocates an additional
runtime buffer on first generation. It also forces piecewise CUDA-graph behavior on this stack.

At `gpu-memory-utilization=0.90`, the server could start but died on the first generation when the
draft requested approximately 394 MB of runtime memory. `0.88` became the proven upper serving
region; the final recipe uses `0.87` to retain explicit workspace margin.

DFlash depth k=15 was also inefficient: acceptance collapsed after roughly position five, wasting
about 60% of draft work. k=7 improved draft efficiency and decode throughput.

## Finding 4: sequence concurrency is a context lever

Will's `max-num-seqs` tip transferred to the DFlash build:

| Change | Graph estimate | KV result |
|---|---:|---:|
| seqs 8, 128K | 0.74 GiB | 160,335 tokens |
| seqs 4, 128K | 0.43 GiB (0.36 actual) | 193,917 tokens |

Reducing sequence concurrency recovered roughly 0.31 GiB of estimated graph reservation and made
larger contexts possible. This trades maximum concurrent scheduling capacity for single-request
context headroom.

## Finding 5: KV admission does not prove runtime safety

A 192K candidate using seqs 4, 4096-token prefill chunks, and gmu .88 booted with a 198,273-token
pool and passed a short generation. A fresh ~180K prompt then OOMed in
`moe_wna16_marlin_gemm` / `aten::new_empty`: 24 MiB was requested while only 2–14 MiB was
physically free on the workers. Approximately 444 MiB was reserved but unusable for that request.

The correction was to halve `max-num-batched-tokens` to 2048 and lower gmu to .87. That produced:

| Candidate | Pool | Long-context result | Speed result |
|---|---:|---|---:|
| 176K | 212,112 | 160,043 input + 27 output passed in 59.45 s | 263.1 tok/s |
| **200K** | **212,822** | **190,002 input + 32 output passed in 73.623 s** | **282.5 tok/s** |

The 200K row became the champion and remains the serving configuration.

## Four-bit KV investigation

The linked KV-quantization discussion motivated a 4-bit cache probe. On this Laguna/vLLM layout,
`--kv-cache-dtype int4_per_token_head` failed during engine initialization:

```text
shape '[64, 2, 16, 2, 68]' is invalid for input of size 540672
```

Will noted that key and value precision do not need to match. vLLM 0.25.1 exposes a mixed
`turboquant_k8v4` mode (8-bit K, 4-bit V), which is the next candidate. It has **not** been
promoted because it has not completed boot, short generation, and fresh long-prompt gates on
this model.

## Known-dead branches

| Branch | Observed result | Decision |
|---|---|---|
| Disable CUDA-graph memory estimate at gmu .88 | Larger printed pool, illegal access on generation | Do not retry |
| Disable CUDA-graph memory estimate at gmu .85 | OOM during graph capture | Do not retry |
| DFlash at gmu .90 | First-generation runtime-buffer OOM | Do not serve |
| CUDA graphs at gmu .85 | Insufficient KV to boot | Do not serve |
| DFlash k15 | Low late-position acceptance | Use k7 |
| `int4_per_token_head` | Invalid Laguna cache reshape | Incompatible on current stack |

## Current roadmap

1. Generation-gate mixed `turboquant_k8v4` at a conservative context before increasing length.
2. If it works, test 224K and then 256K with the same workspace discipline.
3. Re-measure quality as well as capacity; lower-bit V cache can affect output even when serving is stable.
4. Keep the 200K FP8 configuration as the rollback champion until a candidate passes every gate.
