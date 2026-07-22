#!/usr/bin/env python3
"""Fresh near-limit prefill plus generation gate for the Laguna endpoint."""

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only on an idle endpoint; this intentionally submits a very large prompt."
    )
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="laguna")
    parser.add_argument("--prompt-tokens", type=int, default=190000)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    # For the validated tokenizer, repeating "x " produces approximately one token per repeat.
    # The API's returned usage is authoritative and is printed below.
    prompt = "x " * args.prompt_tokens
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_output_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started

    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 3),
                "usage": result.get("usage"),
                "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
            }
        )
    )


if __name__ == "__main__":
    main()
