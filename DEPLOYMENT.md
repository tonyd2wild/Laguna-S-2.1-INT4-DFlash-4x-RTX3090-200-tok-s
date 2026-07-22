# Deployment guide

This guide recreates the measured **200K context + DFlash** endpoint on one four-GPU host.
The exact tested target is 4× RTX 3090 24 GB using vLLM 0.25.1.

## 1. Requirements

- Linux host with four NVIDIA GPUs visible to one Docker daemon
- 24 GB VRAM per GPU for the measured RTX 3090 recipe
- NVIDIA driver and NVIDIA Container Toolkit
- Docker with permission to use `--gpus all`
- Approximately 85 GB free storage for target + DFlash checkpoints, plus image/cache margin
- Hugging Face access sufficient to download both Poolside checkpoints

Verify the GPU runtime before downloading weights:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 2. Download the checkpoints

Install the Hugging Face CLI and stage each model as a normal directory under one cache root:

```bash
python3 -m pip install -U huggingface_hub

mkdir -p /models
hf download poolside/Laguna-S-2.1-INT4 \
  --local-dir /models/Laguna-S-2.1-INT4

hf download poolside/Laguna-S-2.1-DFlash-INT4 \
  --local-dir /models/Laguna-S-2.1-DFlash-INT4
```

The target is roughly 72–77 GB depending on how size is reported. The DFlash checkpoint is
approximately 2.1 GB. Do not mix a DFlash draft from a different target or precision.

## 3. Pull the pinned runtime

```bash
docker pull vllm/vllm-openai:v0.25.1
```

The version is load-bearing. Laguna INT4 and DFlash support are recent, and memory behavior can
change across releases. Treat any image upgrade as a full revalidation event.

## 4. Launch

```bash
git clone https://github.com/tonyd2wild/Laguna-S-2.1-INT4-DFlash-4x-RTX3090-200-tok-s
cd Laguna-S-2.1-INT4-DFlash-4x-RTX3090-200-tok-s

MODEL_CACHE=/models ./launch.sh --dry-run
MODEL_CACHE=/models ./launch.sh
./launch.sh --logs
```

The launcher maps host port `8080` to vLLM port `8000`. Override it if needed:

```bash
MODEL_CACHE=/models HOST_PORT=9000 ./launch.sh
```

### Exact serving flags

| Setting | Value | Reason |
|---|---:|---|
| Image | `vllm/vllm-openai:v0.25.1` | Measured Laguna/DFlash runtime |
| `tensor-parallel-size` | `4` | One rank per RTX 3090 |
| `max-model-len` | `204800` | Validated 200K serving target |
| `gpu-memory-utilization` | `0.87` | Leaves runtime workspace margin |
| `max-num-seqs` | `4` | Reduces graph/scheduler workspace |
| `max-num-batched-tokens` | `2048` | Prevents the long-prefill workspace failure seen at 4096 |
| `num_speculative_tokens` | `7` | Best measured DFlash depth on this stack |
| `moe-backend` | `marlin` | Measured WNA16 MoE backend on Ampere |
| `disable-custom-all-reduce` | enabled | Stable across the two NVLink pairs + PCIe topology |
| Tool/reasoning parser | `poolside_v1` | Laguna's native formats |

Although the upstream recipe may recommend a different MoE backend on other hardware, this
repository documents the exact Ampere configuration that passed on four RTX 3090s.

## 5. Wait for readiness

On the measured host, model loading took about 124 seconds and startup completed after compile,
profiling, KV allocation, and graph capture. Follow the logs:

```bash
docker logs -f vllm-laguna
```

Expected lines include:

```text
Model loading took 17.53 GiB memory
Available KV cache memory: 1.88 GiB
GPU KV cache size: 212,822 tokens
Maximum concurrency for 204,800 tokens per request: 1.04x
Application startup complete.
```

## 6. Validate the endpoint

Do not stop at `/v1/models`. Run an actual generation:

```bash
./scripts/smoke-test.sh http://localhost:8080/v1
```

For a new host or any changed serving flag, run the fresh long-context gate while the endpoint
is otherwise idle:

```bash
python3 scripts/long-context-gate.py \
  --base-url http://localhost:8080/v1 \
  --prompt-tokens 190000 \
  --max-output-tokens 32
```

This request is intentionally expensive. It validates prefill and subsequent decoding near the
configured limit, where a prior candidate failed despite passing boot and short generation.

## 7. Use the OpenAI-compatible API

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laguna",
    "messages": [{"role": "user", "content": "Write a Python retry helper."}],
    "max_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

### Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="laguna",
    messages=[{"role": "user", "content": "Write a Python retry helper."}],
    max_tokens=300,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

Thinking is enabled by default in the measured launcher. Clients can override it per request
through `chat_template_kwargs` if their client supports `extra_body`.

## 8. Network and authentication

vLLM listens on `0.0.0.0` and this recipe does not configure an API key. Do not expose the port
directly to the public internet. Use Tailscale, a host firewall, or an authenticated TLS reverse
proxy. Restricting the bind or adding a proxy does not change the measured engine configuration.
