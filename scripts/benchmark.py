#!/usr/bin/env python3
"""Client-wall benchmark for the Laguna OpenAI-compatible chat endpoint."""

import argparse
import json
import statistics
import time
import urllib.request


DEFAULT_PROMPT = (
    "Create a polished single-file HTML landing page for an AI coding assistant. "
    "Include CSS, a hero, feature cards, pricing, and responsive design. Return only HTML."
)


def request_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="laguna")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    results = []
    for run in range(1, args.repeats + 1):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        started = time.perf_counter()
        response = request_json(
            f"{args.base_url.rstrip('/')}/chat/completions",
            payload,
            args.timeout,
        )
        elapsed = time.perf_counter() - started
        completion_tokens = response["usage"]["completion_tokens"]
        tokens_per_second = completion_tokens / elapsed
        result = {
            "run": run,
            "completion_tokens": completion_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_second, 1),
            "finish_reason": response["choices"][0]["finish_reason"],
        }
        results.append(tokens_per_second)
        print(json.dumps(result))

    print(
        json.dumps(
            {
                "runs": len(results),
                "mean_tokens_per_second": round(statistics.mean(results), 1),
                "median_tokens_per_second": round(statistics.median(results), 1),
                "min_tokens_per_second": round(min(results), 1),
                "max_tokens_per_second": round(max(results), 1),
            }
        )
    )


if __name__ == "__main__":
    main()
