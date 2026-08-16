"""Generate method-faithful DACA prompts with the Gemini REST API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_deepseek_attacks import daca, load_daca_constants, load_env


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
TARGETS = ROOT / "data" / "manifests" / "grsp_attack_targets.csv"
OUTPUT = ROOT / "data" / "manifests" / "grsp_daca_pilot_01_prompts.csv"
TRACE = ROOT / "data" / "raw" / "grsp_daca_pilot" / "gemini_trace.json"
JDB_DACA = ROOT / "tmp" / "JailbreakDiffusionBench" / "jailbreak_diffusion" / "attack" / "DACA.py"
DEFAULT_MODEL = "gemini-3.5-flash"


class Gemini:
    def __init__(self, api_key: str, model: str, trace_path: Path) -> None:
        self.api_key = api_key
        self.model = model
        self.trace_path = trace_path
        self.trace: list[dict[str, Any]] = []
        if trace_path.exists():
            previous = json.loads(trace_path.read_text(encoding="utf-8"))
            if previous.get("model") != model:
                raise RuntimeError(
                    f"Existing trace model {previous.get('model')!r} does not match {model!r}"
                )
            self.trace = list(previous.get("requests", []))

    def query(self, messages: list[dict[str, str]], target_id: str, method: str, stage: str) -> str:
        cached = next(
            (
                row
                for row in self.trace
                if row.get("target_id") == target_id
                and row.get("method") == method
                and row.get("stage") == stage
                and str(row.get("content", "")).strip()
            ),
            None,
        )
        if cached is not None:
            print(f"  resume {stage}", flush=True)
            return str(cached["content"]).strip()

        contents = [
            {"role": "model" if message["role"] == "assistant" else "user", "parts": [{"text": message["content"]}]}
            for message in messages
        ]
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2400},
        }
        encoded_model = urllib.parse.quote(self.model, safe="-._")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:generateContent"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        rate_limit_retries = 0
        transient_retries = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if error.code in {500, 502, 503, 504} and transient_retries < 6:
                    transient_retries += 1
                    delay = min(5.0 * (2 ** (transient_retries - 1)), 60.0)
                    print(
                        f"  temporary HTTP {error.code} during {stage}; waiting {delay:.0f}s "
                        f"(retry {transient_retries}/6)",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                if error.code != 429 or rate_limit_retries >= 8:
                    raise RuntimeError(
                        f"Gemini HTTP {error.code} during {target_id}/{method}/{stage}: {body}"
                    ) from error
                match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body)
                raw_delay = float(match.group(1)) if match else 30.0
                if raw_delay > 90.0:
                    raise RuntimeError(
                        f"Gemini quota window exceeds 90 seconds during "
                        f"{target_id}/{method}/{stage}; retryDelay={raw_delay:.0f}s"
                    ) from error
                delay = raw_delay + 2.0
                delay = max(5.0, min(delay, 60.0))
                rate_limit_retries += 1
                print(
                    f"  rate limit during {stage}; waiting {delay:.0f}s "
                    f"(retry {rate_limit_retries}/8)",
                    flush=True,
                )
                time.sleep(delay)
        elapsed = time.perf_counter() - started
        candidates = result.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        content = "".join(str(part.get("text", "")) for part in parts).strip()
        trace_row = {
            "target_id": target_id,
            "method": method,
            "stage": stage,
            "request_index": len(self.trace) + 1,
            "model": self.model,
            "elapsed_seconds": elapsed,
            "rate_limit_retries": rate_limit_retries,
            "transient_retries": transient_retries,
            "usage": result.get("usageMetadata", {}),
            "prompt_feedback": result.get("promptFeedback", {}),
            "finish_reason": candidates[0].get("finishReason") if candidates else None,
            "messages": messages,
            "content": content,
        }
        self.trace.append(trace_row)
        write_trace(self.trace_path, self)
        if not content:
            raise RuntimeError(
                f"Gemini returned no text during {target_id}/{method}/{stage}; "
                f"finish_reason={trace_row['finish_reason']!r}, prompt_feedback={trace_row['prompt_feedback']!r}"
            )
        return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--targets-file", type=Path, default=TARGETS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--trace", type=Path, default=TRACE)
    return parser.parse_args()


def write_trace(path: Path, client: Gemini) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "provider": "Google Gemini",
                "model": client.model,
                "temperature": 0.0,
                "safety_settings": "provider defaults",
                "source_implementation": str(JDB_DACA),
                "requests": client.trace,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    load_env(ENV_FILE)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    with args.targets_file.open("r", encoding="utf-8", newline="") as handle:
        targets = list(csv.DictReader(handle))
    if args.target:
        selected = set(args.target)
        targets = [target for target in targets if target["target_id"] in selected]
    if not targets:
        raise RuntimeError("No targets selected")

    constants = load_daca_constants(JDB_DACA)
    client = Gemini(api_key, args.model, args.trace)
    output_rows: list[dict[str, str]] = []
    try:
        for target in targets:
            target_id = target["target_id"]
            suffix = target_id.rsplit("_", 1)[1]
            print(f"{target_id}: DACA")
            attack_prompt = daca(client, target_id, target["direct_prompt"], constants)
            output_rows.extend(
                [
                    {
                        "target_id": target_id,
                        "prompt_id": f"benign_daca_{suffix}",
                        "class_label": "benign",
                        "attack_type": "benign",
                        "generator_model": "human_control",
                        "positive_prompt": target["benign_prompt"],
                    },
                    {
                        "target_id": target_id,
                        "prompt_id": f"direct_daca_{suffix}",
                        "class_label": "unsafe",
                        "attack_type": "direct",
                        "generator_model": "human_control",
                        "positive_prompt": target["direct_prompt"],
                    },
                    {
                        "target_id": target_id,
                        "prompt_id": f"daca_attack_{suffix}",
                        "class_label": "unsafe",
                        "attack_type": "DACA",
                        "generator_model": args.model,
                        "positive_prompt": attack_prompt,
                    },
                ]
            )
    finally:
        write_trace(args.trace, client)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Prompts: {args.output}")
    print(f"Trace: {args.trace}")
    print(f"API calls: {len(client.trace)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
