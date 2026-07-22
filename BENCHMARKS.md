# Benchmarks

All numbers in this file were measured on the same four-RTX-3090 host with
`vllm/vllm-openai:v0.25.1`. Projected results are excluded.

## Current serving result

| Metric | Result | Method |
|---|---:|---|
| Max model length | **204,800** | `/v1/models` and engine configuration |
| FP8 KV pool | **212,822 tokens** | vLLM boot log |
| Maximum full-length concurrency | **1.04×** | vLLM boot log |
| Available KV memory | **1.88 GiB/GPU** | vLLM profiler log |
| Model + draft allocation | **17.53 GiB/GPU** | vLLM model-load log |
| Long-context validation | **190,002 prompt + 32 output** | Fresh completion request |
| Long-context wall time | **73.623 s** | Client wall clock |
| HTML chat generation | **282.5 tok/s** | 500 output tokens / 1.770 s client wall |

## Champion progression

| Round | DFlash config | Context | KV pool | Measured decode | Long gate |
|---|---|---:|---:|---:|---|
| Initial speed | k7, graphs, seqs 16, batch 4K, gmu .88 | 80K | 98,827 | 113–245 tok/s | Short generation |
| Round 2 | k7, graphs, seqs 8, batch 4K, gmu .88 | 128K | 160,335 | 149.9–259.7 tok/s | Generation passed |
| Round 3A | k7, graphs, seqs 4, batch 2K, gmu .87 | 176K | 212,112 | 263.1 tok/s | 160,043 + 27 passed |
| **Round 3B** | **k7, graphs, seqs 4, batch 2K, gmu .87** | **200K** | **212,822** | **282.5 tok/s** | **190,002 + 32 passed** |

Decode measurements use client-wall time and `usage.completion_tokens`. They are
workload-sensitive because DFlash acceptance varies with content. The 176K and 200K speed
rows are 500-token HTML generations; the earlier rows include code and HTML prompts.

## Context-first references

| Configuration | Context | KV pool | Decode | Purpose |
|---|---:|---:|---:|---|
| No spec, FULL graphs, gmu .90 | 128K | 224,858 | 87–89 tok/s | Stable non-DFlash alternative |
| Will-reference: no spec, seqs 8, image defaults | 256K | 390,566 | 88.9 tok/s | Reproduced the community's large-pool result |

The Will-reference result resolved the “425K KV pool” mystery. The community configuration
used the model's 256K default context, no DFlash, lower sequence concurrency, and checkpoint-
selected FP8 KV. On this host the equivalent command produced 390,566 pool tokens; version and
default differences explain the remaining gap to the reported 425,799.

## Validation gates

| Candidate | Boot | Short generation | Fresh long prompt | Verdict |
|---|---|---|---|---|
| 128K, seqs 4, batch 4K, gmu .88 | Pass | 300-token output pass | Not required | Workspace probe passed |
| 192K, seqs 4, batch 4K, gmu .88 | Pass, pool 198,273 | Pass | ~180K prompt OOM | Rejected |
| 176K, seqs 4, batch 2K, gmu .87 | Pass, pool 212,112 | Pass | 160,043 + 27 pass | Stable |
| **200K, seqs 4, batch 2K, gmu .87** | **Pass, pool 212,822** | **Pass** | **190,002 + 32 pass** | **Serving** |

## Failure catalog

| Attempt | Failure | Root cause / lesson |
|---|---|---|
| DFlash k15, eager, gmu .90 | Engine died on first generation | DFlash allocated an additional ~394 MB runtime buffer without sufficient margin |
| CUDA graphs, gmu .85 | Boot rejected the KV allocation | Graph reservation left only ~0.25 GiB KV versus ~0.36 GiB required |
| Disable CUDA-graph estimate, gmu .88 | Illegal memory access on first generation | The apparent reclaimed memory was not safe runtime capacity |
| Disable CUDA-graph estimate, gmu .85 | OOM during graph capture | The profiler reserve is load-bearing |
| 192K, batch 4K, gmu .88 | ~180K prefill killed engine | Marlin MoE requested 24 MiB with only 2–14 MiB physically free |
| `int4_per_token_head`, 256K | Engine initialization failed | Laguna cache layout produced an invalid reshape (`[64,2,16,2,68]`) |

## Reproducing measurements

```bash
# Short health + generation gate
./scripts/smoke-test.sh http://localhost:8080/v1

# Three 500-token client-wall HTML chat runs
python3 scripts/benchmark.py \
  --base-url http://localhost:8080/v1 \
  --max-tokens 500 --repeats 3

# Destructive-to-latency near-limit gate; run on an otherwise idle endpoint
python3 scripts/long-context-gate.py \
  --base-url http://localhost:8080/v1 \
  --prompt-tokens 190000 --max-output-tokens 32
```

Before quoting a single-stream result, confirm the endpoint is otherwise idle. Report the
prompt, output token count, temperature, concurrency, client-wall time, and whether the run
was cold or warm.

## KV-pool interpretation

Laguna has 12 full-attention and 36 sliding-window layers. vLLM's hybrid KV manager charges
sliding layers by the window-bounded block count and full layers by configured context. The
printed pool therefore changes with `max-model-len` even when KV bytes barely change.

**Never compare pool-token numbers across different context limits without also comparing
the exact engine configuration and KV memory.** See [EXPERIMENTS.md](EXPERIMENTS.md) for the
allocator investigation and formula.
