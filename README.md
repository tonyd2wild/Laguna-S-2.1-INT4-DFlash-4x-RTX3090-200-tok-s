# Laguna S 2.1 INT4 + DFlash at 200K on 4× RTX 3090

Reproducible vLLM recipe for serving Poolside's 117.6B-total / 8.5B-active
[`Laguna-S-2.1-INT4`](https://huggingface.co/poolside/Laguna-S-2.1-INT4) with the
precision-matched DFlash drafter on four 24 GB RTX 3090s.

The current balanced configuration reaches **200K context and up to 282.5 output tok/s**
on the same endpoint. It was validated with a fresh **190,002-token prompt plus generation**;
it is not a boot-only claim.

## Result

| Item | Validated value |
|---|---:|
| Hardware | 4× NVIDIA RTX 3090 24 GB |
| Runtime | vLLM `v0.25.1`, Docker |
| Tensor parallelism | TP4 |
| Max model length | **204,800 tokens (200K)** |
| FP8 KV pool | **212,822 tokens (1.04×)** |
| Long-context gate | **190,002 input + 32 output tokens passed** |
| Long-context wall time | **73.623 s** |
| HTML chat decode | **500 tokens in 1.770 s = 282.5 tok/s** |
| Served model name | `laguna` |

## Quick start

Prerequisites are Docker, NVIDIA Container Toolkit, four visible 24 GB GPUs, approximately
85 GB of model storage, and the two Poolside checkpoints:

```bash
hf download poolside/Laguna-S-2.1-INT4 \
  --local-dir /models/Laguna-S-2.1-INT4

hf download poolside/Laguna-S-2.1-DFlash-INT4 \
  --local-dir /models/Laguna-S-2.1-DFlash-INT4

docker pull vllm/vllm-openai:v0.25.1
```

Launch the measured configuration:

```bash
MODEL_CACHE=/models ./launch.sh --dry-run
MODEL_CACHE=/models ./launch.sh
./scripts/smoke-test.sh http://localhost:8080/v1
```

The server exposes an OpenAI-compatible API at `http://localhost:8080/v1` and serves the
model as `laguna`. See [DEPLOYMENT.md](DEPLOYMENT.md) for host setup, the complete command,
API examples, and validation steps.

## Why this configuration works

The final step was reducing serving concurrency and prefill workspace while retaining
DFlash and CUDA graphs:

| Lever | Final value | Effect |
|---|---:|---|
| `--max-num-seqs` | `4` | Reduced graph reservation from an estimated 0.74 GiB at seqs 8 to 0.43 GiB at seqs 4. |
| `--max-num-batched-tokens` | `2048` | Reduced peak prefill workspace versus the failed 4096-token chunk configuration. |
| `--gpu-memory-utilization` | `0.87` | Preserved runtime headroom for Marlin MoE and DFlash allocations. |
| DFlash depth | `7` | Avoided the acceptance collapse seen beyond approximately five draft positions at k=15. |
| KV dtype | checkpoint-selected FP8 | The INT4 checkpoint enables calibrated FP8 KV automatically; an explicit FP8 flag is a no-op. |

The previous 196K attempt booted and passed a short request, but a fresh ~180K prefill
OOMed in Marlin workspace. That result established the repository's central validation rule:
**a printed KV pool and a healthy `/v1/models` response do not prove long-context serving.**

## Measured progression

| Serving build | Context | KV pool | Decode result | Outcome |
|---|---:|---:|---:|---|
| DFlash, seqs 16 | 80K | 98,827 | 113–245 tok/s | Original speed build |
| DFlash, seqs 8 | 128K | 160,335 | 150–260 tok/s | Round-2 champion |
| DFlash, seqs 4, 2K chunks | 176K | 212,112 | 263.1 tok/s | Stable stepping stone |
| **DFlash, seqs 4, 2K chunks** | **200K** | **212,822** | **282.5 tok/s** | **Current champion** |
| No DFlash, Will-reference | 256K | 390,566 | 88.9 tok/s | Context-first reference |

These rows use different `max-model-len` values. Laguna's hybrid sliding/global attention
accounting makes the printed pool depend on configured context, so pool-token counts must not
be compared as if each token represented a fixed engine-wide byte cost. Full methodology and
failed configurations are in [BENCHMARKS.md](BENCHMARKS.md) and [EXPERIMENTS.md](EXPERIMENTS.md).

## Repository map

| File | Purpose |
|---|---|
| [`launch.sh`](launch.sh) | Exact champion launcher with dry-run, status, logs, and stop commands |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | From-zero installation and API usage |
| [`OPERATIONS.md`](OPERATIONS.md) | Live endpoint, health checks, safe operating procedure, and rollback |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Clean, measured result tables and methodology |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Full investigation timeline, discoveries, and dead ends |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Known failure signatures and recoveries |
| [`NOTICE.md`](NOTICE.md) | Model/runtime provenance and third-party notice |
| [`scripts/benchmark.py`](scripts/benchmark.py) | Reproducible client-wall chat benchmark |
| [`scripts/long-context-gate.py`](scripts/long-context-gate.py) | Fresh near-limit prefill + generation gate |
| [`scripts/smoke-test.sh`](scripts/smoke-test.sh) | API health and short generation check |

## Important constraints

- The champion is a **single-stream/low-concurrency balanced build**. `max-num-seqs=4` is a
  memory lever, not a claim of high-throughput concurrency at 200K.
- The server has no built-in API key in this recipe. Keep it behind Tailscale, a firewall,
  or an authenticated reverse proxy.
- Do not disable vLLM's CUDA-graph memory estimate. Both estimate-off tests crashed.
- Do not raise DFlash builds to `gpu-memory-utilization >= 0.90`; the first-generation
  DFlash runtime buffer has previously OOMed there.
- `int4_per_token_head` KV failed to initialize for Laguna's cache shape. Mixed
  `turboquant_k8v4` is a documented next experiment, not a validated serving mode.

## Upstream references and credit

- [Official vLLM Laguna S 2.1 recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1)
- [Poolside Laguna S 2.1 INT4 model card](https://huggingface.co/poolside/Laguna-S-2.1-INT4)
- [Poolside Laguna S 2.1 DFlash INT4](https://huggingface.co/poolside/Laguna-S-2.1-DFlash-INT4)

Credit to Poolside for Laguna and the calibrated FP8 KV/DFlash checkpoints, the vLLM team
for Laguna and DFlash support, Kai for the initial four-GPU bring-up and shard-integrity
work, and **Will / [@hampsonw](https://x.com/hampsonw)** for the `max-num-seqs` lead and
the mixed K8/V4 suggestion.
