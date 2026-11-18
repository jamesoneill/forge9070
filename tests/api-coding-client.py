#!/usr/bin/env python3
"""Exercise the local Chat Completions contract with deterministic CPU fixtures.

Only Python's standard library is used. A temporary copy of the synthetic GGUF
can emit one complete function-call token, so the real HTTP and SSE serializers
are exercised without claiming that an untrained fixture can select useful tools.
No downloads, driver changes, external listeners or actual tools are involved.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
KEY = "forge9070-coding-client-local-test"
TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read one source file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
TOOL_TEXT = (
    '<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n'
    '</parameter>\n</function>\n</tool_call>'
)


def check(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def fixture_copy(source: Path, destination: Path, tool_response: bool) -> None:
    """Rewrite only this known synthetic fixture's context and optional output row."""
    raw = source.read_bytes()
    check(raw[:8] == b"GGUF" + struct.pack("<I", 3), "expected a GGUF v3 fixture")
    tensors, metadata = struct.unpack_from("<QQ", raw, 8)
    pos = 24

    def string() -> str:
        nonlocal pos
        length, = struct.unpack_from("<Q", raw, pos)
        pos += 8
        value = raw[pos:pos + length].decode()
        pos += length
        return value

    def pack_string(value: str) -> bytes:
        text = value.encode()
        return struct.pack("<Q", len(text)) + text

    def value(kind: int):
        nonlocal pos
        if kind == 8:
            return string()
        if kind == 9:
            child, count = struct.unpack_from("<IQ", raw, pos)
            pos += 12
            return [value(child) for _ in range(count)]
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        fmt = "<" + formats[kind]
        result, = struct.unpack_from(fmt, raw, pos)
        pos += struct.calcsize(fmt)
        return result

    entries = []
    output_token = None
    for _ in range(metadata):
        start = pos
        key = string()
        kind, = struct.unpack_from("<I", raw, pos)
        pos += 4
        data = value(kind)
        if key.endswith(".context_length"):
            entries.append(pack_string(key) + struct.pack("<II", 4, 4096))
        elif key == "tokenizer.ggml.tokens" and tool_response:
            check(data[-1] == "</think>", "unexpected fixture vocabulary")
            output_token = len(data) - 1
            data[-1] = TOOL_TEXT
            entries.append(pack_string(key) + struct.pack("<IIQ", 9, 8, len(data))
                           + b"".join(pack_string(token) for token in data))
        else:
            entries.append(raw[start:pos])
    descriptors = pos
    output = None
    for _ in range(tensors):
        name = string()
        ndims, = struct.unpack_from("<I", raw, pos)
        pos += 4
        dims = struct.unpack_from("<" + "Q" * ndims, raw, pos)
        pos += 8 * ndims
        dtype, offset = struct.unpack_from("<IQ", raw, pos)
        pos += 12
        if name == "output.weight":
            output = (dims, dtype, offset)
    tensor_descriptions = raw[descriptors:pos]
    data_start = (pos + 31) // 32 * 32
    payload = bytearray(raw[data_start:])
    if tool_response:
        check(output is not None and output_token is not None, "missing fixture output")
        dims, dtype, offset = output
        check(dtype == 0 and len(dims) == 2, "fixture output must be F32")
        width, vocab = dims
        for row in range(vocab):
            struct.pack_into("<" + "f" * width, payload, offset + row * width * 4,
                             *([1.0 if row == output_token else 0.0] * width))
    header = raw[:24] + b"".join(entries) + tensor_descriptions
    destination.write_bytes(header + bytes((-len(header)) % 32) + payload)


@contextmanager
def server(binary: Path, model: Path):
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    env = os.environ.copy()
    env["STRATA_API_KEY"] = KEY
    env["FORGE9070_API_KEY"] = KEY

    def request(path: str, body: dict | None = None, auth: bool = True):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        headers = {"Authorization": f"Bearer {KEY}"} if auth else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request("POST" if body is not None else "GET", path,
                               json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode()
        finally:
            connection.close()

    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [str(binary), "serve", "--model", str(model), "--cpu", "--port", str(port),
             "--ctx", "4096", "--max-tokens", "16", "--ram-margin", "0"],
            cwd=ROOT, env=env, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                check(process.poll() is None, "server exited during startup")
                try:
                    status, _, data = request("/health", auth=False)
                    if status == 200 and json.loads(data)["status"] == "ok":
                        break
                except (OSError, http.client.HTTPException):
                    pass
                check(time.monotonic() < deadline, "server startup timed out")
                time.sleep(0.05)
            yield request
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


def events(text: str) -> list[dict]:
    chunks = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    check(chunks and chunks[-1] == "[DONE]", f"stream did not finish: {text}")
    return [json.loads(chunk) for chunk in chunks[:-1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/forge9070")
    args = parser.parse_args()
    binary = args.binary.resolve()
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="forge-api-client-") as temp:
        directory = Path(temp)
        plain, tools = directory / "plain.gguf", directory / "tools.gguf"
        fixture = ROOT / "tests/fixtures/tiny-qwen3moe.gguf"
        fixture_copy(fixture, plain, False)
        fixture_copy(fixture, tools, True)
        with server(binary, plain) as request:
            status, _, text = request("/v1/models", auth=False)
            check(status == 401, f"unauthenticated model discovery: {status} {text}")
            status, _, text = request("/v1/models")
            model_list = json.loads(text)
            check(status == 200 and model_list["object"] == "list", text)
            model = model_list["data"][0]["id"]
            checks.append("Authenticated GET /v1/models exposes the actual model ID")
            body = {
                "model": model,
                "messages": [{"role": "developer", "content": "Keep code indentation."},
                             {"role": "user", "content": "hello"}],
                "max_completion_tokens": 4, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            status, _, text = request("/v1/chat/completions", body)
            reply = json.loads(text)
            check(status == 200, text)
            check(reply["object"] == "chat.completion" and reply["model"] == model, text)
            check(reply["choices"][0]["message"]["content"] == "AAAA", text)
            check(reply["choices"][0]["finish_reason"] == "length", text)
            check(reply["usage"]["completion_tokens"] == 4, text)
            checks.append("Developer instructions and max_completion_tokens reach nonstreaming generation")
            identifiers = {reply["id"]}
            for usage in [False, True]:
                streaming = dict(body, stream=True, stream_options={"include_usage": usage})
                status, headers, text = request("/v1/chat/completions", streaming)
                check(status == 200, text)
                check(any(k.lower() == "content-type" and "text/event-stream" in v
                          for k, v in headers.items()), str(headers))
                chunks = events(text)
                ids = {chunk["id"] for chunk in chunks}
                check(len(ids) == 1 and not (ids & identifiers), text)
                identifiers.update(ids)
                check(all(chunk["model"] == model and chunk["object"] == "chat.completion.chunk"
                          for chunk in chunks), text)
                check(len({chunk["created"] for chunk in chunks}) == 1, text)
                check("".join(choice.get("delta", {}).get("content", "") or ""
                              for chunk in chunks for choice in chunk["choices"]) == "AAAA", text)
                finished = [choice["finish_reason"] for chunk in chunks
                            for choice in chunk["choices"] if choice.get("finish_reason")]
                check(finished == ["length"], text)
                if usage:
                    check(chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 4, text)
                    check(all("usage" in chunk and chunk["usage"] is None for chunk in chunks[:-1]), text)
                else:
                    check(all("usage" not in chunk for chunk in chunks), text)
                checks.append(f"SSE has stable IDs, truthful model, finish reason and include_usage={usage}")
            status, _, text = request("/v1/chat/completions", dict(body, stop="A"))
            stopped = json.loads(text)
            check(status == 200 and stopped["choices"][0]["finish_reason"] == "stop", text)
            check(stopped["choices"][0]["message"]["content"] == "", text)
            checks.append("Stop sequences end generation with stop instead of length")
            status, _, text = request("/v1/chat/completions", dict(body, model="wrong-model"))
            check(status == 409 and "loaded_model" in json.loads(text)["error"], text)
            checks.append("Explicit model mismatch does not silently use another checkpoint")
            image_messages = [{"role": "user", "content": [
                {"type": "text", "text": "Inspect this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ]}]
            for streaming in [False, True]:
                status, headers, text = request("/v1/chat/completions", dict(
                    body, messages=image_messages, stream=streaming))
                check(status == 400, f"preparation failure: {status} {text}")
                check(json.loads(text)["error"]["code"] == "prompt_preparation_failed", text)
                check(any(k.lower() == "content-type" and "application/json" in v
                          for k, v in headers.items()), str(headers))
                checks.append(f"Unavailable image preparation returns JSON400 before SSE, stream={streaming}")
            status, _, text = request("/v1/messages/count_tokens", {
                "model": model,
                "messages": [{"role": "user", "content": [{"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "AA==",
                }}]}],
            })
            check(status == 400 and json.loads(text)["error"]["code"] == "prompt_preparation_failed", text)
            checks.append("Messages token counting reports image preparation errors as JSON400")
            deadline = time.monotonic() + 2
            while True:
                status, _, text = request("/v1/status")
                current = json.loads(text)
                check(status == 200, text)
                counts = current["admission"]
                if (current["generating_requests"], counts["active"], counts["reserved"], counts["queued"]) == (0, 0, 0, 0):
                    break
                check(time.monotonic() < deadline, f"preparation errors leaked request capacity: {text}")
                time.sleep(0.01)
            checks.append("Preparation failures release execution capacity and leave no queued request")
            invalid = [
                ({"max_tokens": 3}, "disagree"),
                ({"tool_choice": "required"}, "not implemented"),
                ({"tool_choice": {"type": "function", "function": {"name": "read_file"}}}, "not implemented"),
                ({"parallel_tool_calls": False}, "not implemented"),
                ({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}, "not implemented"),
                ({"logprobs": True}, "not implemented"),
                ({"reasoning_effort": "high"}, "not implemented"),
                ({"stream": "true"}, "boolean"),
                ({"stream_options": {"include_usage": True}}, "requires stream=true"),
                ({"tools": [{"type": "function", "function": {"name": "read_file", "strict": True}}]}, "not implemented"),
                ({"messages": [{"role": "tool", "tool_call_id": "orphan", "content": "result"}]}, "pending"),
                ({"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": "file-1"}}]}]}, "unsupported"),
            ]
            for extra, expected in invalid:
                status, _, text = request("/v1/chat/completions", dict(body, **extra))
                check(status == 400 and expected in json.loads(text)["error"]["message"], f"{extra}: {status} {text}")
                checks.append(f"HTTP 400 explicitly rejects unsupported/invalid {next(iter(extra))}")

        with server(binary, tools) as request:
            status, _, text = request("/v1/models")
            check(status == 200, text)
            model = json.loads(text)["data"][0]["id"]
            body = {
                "model": model, "messages": [{"role": "user", "content": "read the file"}],
                "tools": [TOOL], "tool_choice": "auto",
                "max_completion_tokens": 1, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            status, _, text = request("/v1/chat/completions", body)
            reply = json.loads(text)
            check(status == 200 and reply["choices"][0]["finish_reason"] == "tool_calls", text)
            assistant = reply["choices"][0]["message"]
            call = assistant["tool_calls"][0]
            check(call["type"] == "function" and call["function"]["name"] == "read_file", text)
            check(json.loads(call["function"]["arguments"]) == {"path": "src/main.rs"}, text)
            check(bool(call["id"]), text)
            checks.append("A generated fixture function call returns a call ID, JSON arguments and tool_calls finish reason")
            status, _, text = request("/v1/chat/completions", dict(
                body, messages=body["messages"] + [assistant, {
                    "role": "tool", "tool_call_id": call["id"], "content": "fn main() {}",
                }]))
            check(status == 200 and json.loads(text)["choices"][0]["finish_reason"] == "tool_calls", text)
            checks.append("Assistant tool_calls and matching tool results survive a second API turn")
            status, _, text = request("/v1/chat/completions", dict(
                body, stream=True, stream_options={"include_usage": True}))
            check(status == 200, text)
            chunks = events(text)
            streamed_calls = [call for chunk in chunks for choice in chunk["choices"]
                              for call in choice.get("delta", {}).get("tool_calls", [])]
            check(len(streamed_calls) == 1 and streamed_calls[0]["index"] == 0, text)
            check(streamed_calls[0]["function"]["name"] == "read_file", text)
            check(json.loads(streamed_calls[0]["function"]["arguments"]) == {"path": "src/main.rs"}, text)
            check(any(choice.get("finish_reason") == "tool_calls"
                      for chunk in chunks for choice in chunk["choices"]), text)
            check(chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 1, text)
            checks.append("SSE tool-call deltas include function, arguments, index, ID, finish reason and final usage")

    print(json.dumps({
        "status": "passed", "checks": checks, "backend_tested": "cpu",
        "fixture": "temporary synthetic GGUFs; one emits a forced function-call token",
        "real_model_tool_selection_tested": False,
        "qwen38_gpu_correctness_tested": False,
        "actual_external_coding_client_tested": False,
    }, indent=2))


if __name__ == "__main__":
    main()
