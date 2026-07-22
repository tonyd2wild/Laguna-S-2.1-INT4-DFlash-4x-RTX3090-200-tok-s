# Operations runbook

## Current production-like endpoint

| Item | Value |
|---|---|
| API base | `http://<your-host>:8080/v1` |
| Network | Tailscale/private network |
| Served model | `laguna` |
| Container | `vllm-laguna` |
| Image | `vllm/vllm-openai:v0.25.1` |
| Max context | `204800` tokens |
| KV pool | `212822` tokens |
| Restart policy | `unless-stopped` |

The endpoint has no API-key enforcement. It should remain on the private network or behind an
authenticated proxy.

## Routine checks

```bash
./launch.sh --status

curl -s http://<your-host>:8080/v1/models | python3 -m json.tool

docker logs vllm-laguna 2>&1 | \
  grep -E 'GPU KV cache size|Maximum concurrency|ERROR|Traceback' | tail -20
```

A healthy model-list response is necessary but insufficient. Always run a short generation after
a restart or configuration change:

```bash
./scripts/smoke-test.sh http://<your-host>:8080/v1
```

## Start, logs, status, stop

```bash
./launch.sh
./launch.sh --logs
./launch.sh --status
./launch.sh --stop
```

The launcher removes only the exact container named by `CONTAINER_NAME` before starting a new
instance. On Tony's host, do not touch `glm-lan-proxy`; it serves another workload and uses no GPU.
The stopped Qwen containers should remain stopped unless Tony explicitly requests their restore.

## Safe change procedure

1. Record the current `docker inspect vllm-laguna` command and boot telemetry.
2. Set experimental containers to `--restart no` so a bad configuration cannot crash-loop.
3. Confirm the expected KV pool and graph allocation in logs.
4. Run a short generation.
5. For any context or memory change, run a fresh near-limit prompt plus output generation.
6. Measure performance only while the endpoint is otherwise idle.
7. Promote the candidate to `--restart unless-stopped` only after all gates pass.
8. Restore the champion if the candidate fails any gate.

## Crash-loop recovery

```bash
docker update --restart no vllm-laguna
docker logs --tail 200 vllm-laguna
docker rm -f vllm-laguna
./launch.sh
```

If logs repeatedly report `shm_broadcast: No available shared memory`, treat the engine as wedged,
remove the exact Laguna container, and relaunch the champion. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Host-specific cautions

- Audit the host for stale watchdog/restart scripts from previous deployments before trusting uptime — a leftover script that `pkill`s vllm or boots an old model config will sabotage this serve.
- The host's wired link is approximately 100 Mbps. Reuse the verified local checkpoints instead
  of downloading them again.
- The four cards form two NVLink pairs with PCIe between pairs. Keep
  `--disable-custom-all-reduce` unless a new topology-specific test proves otherwise.
- Do not alter the target or draft weight directories in place during serving.

## Current checkpoint paths on Tony's host

```text
<MODEL_CACHE>/Laguna-S-2.1-INT4
<MODEL_CACHE>/Laguna-S-2.1-DFlash-INT4
```

They are mounted into the container at `/root/.cache/huggingface`.
