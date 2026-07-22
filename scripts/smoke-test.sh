#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8080/v1}"

printf 'Model list:\n'
curl --fail --silent --show-error "${BASE_URL}/models"
printf '\n\nShort generation:\n'
curl --fail --silent --show-error "${BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"laguna","messages":[{"role":"user","content":"Reply with exactly: Laguna serving is healthy."}],"max_tokens":32,"temperature":0}'
printf '\n'
