# Troubleshooting

## `/v1/models` works, but the first generation kills the engine

This is the known DFlash runtime-buffer failure. A model-list response does not allocate every
generation-time workspace. Check for CUDA OOM messages immediately after the first request.

Recovery:

```bash
docker update --restart no vllm-laguna
docker logs --tail 200 vllm-laguna
docker rm -f vllm-laguna
./launch.sh
```

Do not raise the champion's `gpu-memory-utilization` above .87 without a full generation gate.
DFlash previously failed at .90.

## Long prompts fail although the printed pool exceeds the prompt

KV admission covers cache blocks, not every temporary Marlin MoE or prefill allocation. The
rejected 192K candidate booted with 198,273 pool tokens and passed a short request, then OOMed on
a fresh ~180K prefill.

Use the champion values together:

```text
max-model-len=204800
gpu-memory-utilization=0.87
max-num-seqs=4
max-num-batched-tokens=2048
```

Do not restore the failed 4096-token prefill chunk merely because its boot pool looks adequate.

## The container is crash-looping

The serving launcher uses `unless-stopped`. Disable automatic restarts before diagnosis:

```bash
docker update --restart no vllm-laguna
docker inspect -f '{{.RestartCount}}' vllm-laguna
docker logs --tail 300 vllm-laguna
```

After diagnosis, remove only the exact Laguna container and relaunch the champion.

## Logs repeat `shm_broadcast: No available shared memory`

The multiprocess engine is wedged, commonly after an illegal CUDA access. Waiting does not repair
the worker group. Disable restarts, capture the logs, remove the exact container, and relaunch.

## Explicit `--kv-cache-dtype fp8` does not increase capacity

Expected. The target checkpoint embeds a calibrated FP8 KV scheme and vLLM activates it under
`auto`. The explicit flag is effectively a no-op for this checkpoint.

## `int4_per_token_head` fails with an invalid shape

Expected on the tested stack. The Laguna hybrid target/draft cache layout failed initialization
with an invalid reshape ending in dimension 68. Remove the flag and return to checkpoint-selected
FP8. Mixed `turboquant_k8v4` remains experimental and must not be treated as a fallback champion.

## Graph capture OOMs at gmu .85

At .85, CUDA-graph reservation left too little KV memory for the DFlash configuration. The final
recipe uses .87 and smaller scheduling dimensions. Do not disable the graph-memory estimate; that
produced a larger apparent pool but crashed later.

## Tokenizer regex or reasoning-token warnings appear

The tested vLLM/Transformers stack logs a Mistral-regex warning and may warn that reasoning token
IDs could not be auto-initialized. These warnings were present on the validated serving build and
did not prevent generation. Treat any tokenizer or chat-template change as a quality-regression
event rather than applying an untested startup edit.

## Tool calls or reasoning are not parsed

Confirm all three settings are present:

```text
--enable-auto-tool-choice
--tool-call-parser poolside_v1
--reasoning-parser poolside_v1
```

The launcher enables thinking by default. Per-request clients can override
`chat_template_kwargs.enable_thinking`.

## Performance is much lower than the headline

DFlash speed is workload- and acceptance-dependent. Before diagnosing the engine:

1. Verify no other requests are running.
2. Use chat completions, the same prompt, output length, and temperature.
3. Warm the endpoint once, then report every run rather than only the best.
4. Inspect speculative-decoding metrics in the logs.
5. Confirm the current command still uses DFlash k7 and CUDA graphs.

The headline 282.5 tok/s is a measured 500-token HTML chat generation, not a guaranteed rate for
every prompt.
