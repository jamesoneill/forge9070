#!/usr/bin/env python3
"""Exercise Forge9070's real CPU inference and local HTTP server.

Run after building:
  python tests/smoke.py --binary target/debug/forge9070

Uses only Python's standard library and the committed synthetic GGUF fixture.
Never downloads weights, touches drivers, or binds a non-loopback address.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
KEY = "forge9070-local-test-only"


def check(value: bool, detail: str) -> None:
    if not value:
        raise AssertionError(detail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "target" / "debug"
                        / ("forge9070.exe" if os.name == "nt" else "forge9070"))
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "tests" / "fixtures" / "tiny-qwen3moe.gguf")
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    fixture = str(args.fixture.resolve())
    results: list[str] = []

    generated = subprocess.run(
        [binary, "generate", "--model", fixture, "--cpu", "--prompt", "hello",
         "--max-tokens", "4", "--ctx", "128", "--ram-margin", "0"],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    check(generated.returncode == 0,
          f"generate failed ({generated.returncode}): {generated.stderr}")
    check(generated.stdout.strip() == "AAAA",
          f"unexpected generated output: {generated.stdout!r}")
    results.append("CPU generation executes real model and emits AAAA")

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    env = os.environ.copy()
    env["STRATA_API_KEY"] = KEY
    env["FORGE9070_API_KEY"] = KEY

    def request(path: str, body: dict | None = None, auth: bool = True):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        headers = {"Authorization": f"Bearer {KEY}"} if auth else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request("POST" if body is not None else "GET", path,
                               json.dumps(body) if body is not None else None,
                               headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode()
        finally:
            connection.close()

    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [binary, "serve", "--model", fixture, "--cpu", "--port", str(port),
             "--ctx", "128", "--max-tokens", "16", "--ram-margin", "0"],
            cwd=ROOT, env=env, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                check(process.poll() is None, "server exited before becoming ready")
                try:
                    status, _, content = request("/health", auth=False)
                    if status == 200:
                        check(json.loads(content)["status"] == "ok", content)
                        break
                except (OSError, http.client.HTTPException):
                    pass
                check(time.monotonic() < deadline, "server startup timed out")
                time.sleep(0.05)
            results.append("HTTP health waits until real model is loaded")

            status, _, _ = request("/v1/models", auth=False)
            check(status == 401, f"missing API key returned {status}")
            results.append("Configured API key is enforced")

            status, _, content = request("/v1/models")
            check(status == 200, f"model list: {status} {content}")
            models = json.loads(content)["data"]
            check(bool(models), "server advertised no model")
            model_id = models[0]["id"]
            results.append("OpenAI model listing returns loaded model")

            body = {
                "model": model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            status, _, content = request("/v1/chat/completions", body)
            check(status == 200, f"chat: {status} {content}")
            data = json.loads(content)
            check(data["choices"][0]["message"]["content"] == "AAAA", content)
            check(data["usage"]["completion_tokens"] == 4, content)
            check(data["choices"][0]["finish_reason"] == "length", content)
            results.append("OpenAI chat returns exact text, token count and finish reason")

            body["stream"] = True
            status, headers, content = request("/v1/chat/completions", body)
            check(status == 200, f"stream: {status} {content}")
            check("text/event-stream" in next(
                (v for k, v in headers.items() if k.lower() == "content-type"), ""
            ), f"wrong SSE content type: {headers}")
            chunks = [line[6:] for line in content.splitlines() if line.startswith("data: ")]
            check(chunks and chunks[-1] == "[DONE]", content)
            events = [json.loads(chunk) for chunk in chunks[:-1]]
            text = "".join(choice.get("delta", {}).get("content") or ""
                           for event in events for choice in event.get("choices", []))
            check(text == "AAAA", content)
            check(any(choice.get("finish_reason") == "length"
                      for event in events for choice in event.get("choices", [])), content)
            results.append("OpenAI SSE stream returns exact text and DONE marker")

            # Explicit output budgets must not be silently shortened: a code
            # client needs an actionable error before generation begins.
            body["stream"] = False
            body["max_tokens"] = 32
            status, _, content = request("/v1/chat/completions", body)
            check(status == 400, f"capped request: {status} {content}")
            data = json.loads(content)
            check("error" in data and "16" in content and "32" in content, content)
            results.append("Server rejects a 32-token request above its 16-token cap before inference")

            status, _, content = request("/v1/chat/completions", {"messages": []})
            check(status == 400 and "error" in json.loads(content),
                  f"malformed request: {status} {content}")
            results.append("Malformed chat request returns HTTP 400")

            status, _, content = request('/v1/chat/completions', {
                'messages': [{'role': 'user', 'content': 'hello'}],
                'presence_penalty': 1.5,
            })
            check(status == 400 and 'presence_penalty' in content and 'not implemented' in content,
                  f'unsupported sampling request was not rejected: {status} {content}')
            results.append('Unsupported output-changing sampling options fail explicitly instead of being silently ignored')
        except BaseException:
            log.seek(0)
            print(log.read().decode(errors="replace"))
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    print(json.dumps({
        "status": "passed", "checks": results, "fixture": fixture,
        "backend_tested": "cpu", "rx9070xt_gpu_tested": False,
        "large_moe_model_tested": False,
    }, indent=2))


if __name__ == "__main__":
    main()
