#!/usr/bin/env bash
set -euo pipefail

# Measured 200K Laguna + DFlash configuration for 4× RTX 3090 24 GB.
# Override paths/ports with environment variables; serving flags remain pinned.

CONTAINER_NAME="${CONTAINER_NAME:-vllm-laguna}"
HOST_PORT="${HOST_PORT:-8080}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
MODEL_CACHE="${MODEL_CACHE:-/home/tony/club-3090/models-cache}"
MODEL_DIR="${MODEL_DIR:-Laguna-S-2.1-INT4}"
DRAFT_DIR="${DRAFT_DIR:-Laguna-S-2.1-DFlash-INT4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

MODEL_PATH="/root/.cache/huggingface/${MODEL_DIR}"
DRAFT_PATH="/root/.cache/huggingface/${DRAFT_DIR}"

usage() {
  printf '%s\n' \
    "Usage: ./launch.sh [--dry-run|--status|--logs|--stop]" \
    "" \
    "Environment overrides:" \
    "  MODEL_CACHE      Host directory containing both checkpoints" \
    "  MODEL_DIR        Target checkpoint directory" \
    "  DRAFT_DIR        DFlash checkpoint directory" \
    "  HOST_PORT        Host API port (default: 8080)" \
    "  CONTAINER_NAME   Docker container name (default: vllm-laguna)" \
    "  IMAGE            vLLM image (default: vllm/vllm-openai:v0.25.1)" \
    "  GPU_IDS          CUDA device list (default: 0,1,2,3)"
}

require_safe_name() {
  if [[ -z "${CONTAINER_NAME}" || "${CONTAINER_NAME}" == "/" ]]; then
    printf 'Unsafe CONTAINER_NAME; refusing.\n' >&2
    exit 2
  fi
}

cmd=(
  docker run -d
  --name "${CONTAINER_NAME}"
  --gpus all
  -e "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0
  -e VLLM_USE_DEEP_GEMM=0
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --ipc=host
  --shm-size=16g
  --restart unless-stopped
  -p "${HOST_PORT}:8000"
  -v "${MODEL_CACHE}:/root/.cache/huggingface"
  "${IMAGE}"
  "${MODEL_PATH}"
  --trust-remote-code
  --tensor-parallel-size 4
  --pipeline-parallel-size 1
  --max-model-len 204800
  --gpu-memory-utilization 0.87
  --max-num-seqs 4
  --max-num-batched-tokens 2048
  --enable-auto-tool-choice
  --tool-call-parser poolside_v1
  --reasoning-parser poolside_v1
  --moe-backend marlin
  --disable-custom-all-reduce
  --speculative-config "{\"model\":\"${DRAFT_PATH}\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"
  --served-model-name laguna
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --host 0.0.0.0
  --port 8000
)

action="${1:-launch}"
case "${action}" in
  --dry-run)
    printf 'docker rm -f %q\n' "${CONTAINER_NAME}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    ;;
  --status)
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}'
    curl --fail --silent --show-error "http://localhost:${HOST_PORT}/v1/models"
    printf '\n'
    ;;
  --logs)
    docker logs -f "${CONTAINER_NAME}"
    ;;
  --stop)
    require_safe_name
    docker rm -f "${CONTAINER_NAME}"
    ;;
  launch)
    require_safe_name
    if [[ ! -d "${MODEL_CACHE}/${MODEL_DIR}" ]]; then
      printf 'Missing target checkpoint: %s\n' "${MODEL_CACHE}/${MODEL_DIR}" >&2
      exit 2
    fi
    if [[ ! -d "${MODEL_CACHE}/${DRAFT_DIR}" ]]; then
      printf 'Missing DFlash checkpoint: %s\n' "${MODEL_CACHE}/${DRAFT_DIR}" >&2
      exit 2
    fi
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    "${cmd[@]}"
    printf 'Launching %s; follow with: ./launch.sh --logs\n' "${CONTAINER_NAME}"
    printf 'API will be: http://localhost:%s/v1 (model: laguna)\n' "${HOST_PORT}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
