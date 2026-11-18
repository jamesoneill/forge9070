#!/usr/bin/env python3
"""Exercise bounded generation admission through Forge's real local HTTP API.

Uses only the standard library. A temporary, untrained CPU MoE fixture repeats
the existing nonzero test layers so a real decode remains active while requests
queue. Transitions are observed through /v1/status, not inferred from sleeps.
No model downloads, product delay hooks, GPU claims, or external listeners.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
KEY = "forge9070-scheduling-local-test"
CONTEXT = 32768
LONG_OUTPUT = 16384
LAYERS = 4


def check(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def expand_fixture(source: Path, destination: Path) -> dict:
    """Repeat only this known F32 fixture's layer tensors; retain its A output."""
    raw = source.read_bytes()
    check(raw[:8] == b"GGUF" + struct.pack("<I", 3), "fixture must be GGUF v3")
    tensors, metadata = struct.unpack_from("<QQ", raw, 8)
    pos = 24

    def string() -> str:
        nonlocal pos
        length, = struct.unpack_from("<Q", raw, pos)
        pos += 8
        result = raw[pos:pos + length].decode()
        pos += length
        return result

    def packed_string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

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
    architecture = None
    for _ in range(metadata):
        start = pos
        key = string()
        kind, = struct.unpack_from("<I", raw, pos)
        pos += 4
        data = value(kind)
        if key == "general.architecture":
            architecture = data
        if key.endswith(".block_count"):
            check(kind == 4 and data == 1, "expected the one-layer synthetic fixture")
            entries.append(packed_string(key) + struct.pack("<II", 4, LAYERS))
        elif key.endswith(".context_length"):
            entries.append(packed_string(key) + struct.pack("<II", 4, CONTEXT))
        else:
            entries.append(raw[start:pos])
    check(architecture == "qwen3moe", "scheduling fixture must use the synthetic Qwen3MoE model")
    descriptions = []
    for _ in range(tensors):
        name = string()
        ndims, = struct.unpack_from("<I", raw, pos)
        pos += 4
        dims = struct.unpack_from("<" + "Q" * ndims, raw, pos)
        pos += 8 * ndims
        dtype, offset = struct.unpack_from("<IQ", raw, pos)
        pos += 12
        check(dtype == 0, "only the known synthetic F32 tensors may be expanded")
        descriptions.append((name, dims, offset))
    data_start = (pos + 31) // 32 * 32
    payload = bytearray()
    rewritten = []
    for name, dims, offset in descriptions:
        size = math.prod(dims) * 4
        data = raw[data_start + offset:data_start + offset + size]
        check(len(data) == size, f"truncated fixture tensor {name}")
        names = [name.replace("blk.0.", f"blk.{layer}.", 1) for layer in range(LAYERS)] \
            if name.startswith("blk.0.") else [name]
        for new_name in names:
            rewritten.append(packed_string(new_name) + struct.pack("<I", len(dims))
                             + struct.pack("<" + "Q" * len(dims), *dims)
                             + struct.pack("<IQ", 0, len(payload)))
            payload.extend(data)
            payload.extend(bytes((-len(payload)) % 32))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(rewritten), metadata) \
        + b"".join(entries) + b"".join(rewritten)
    expanded = header + bytes((-len(header)) % 32) + payload
    destination.write_bytes(expanded)
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "expanded_sha256": hashlib.sha256(expanded).hexdigest(),
        "expanded_bytes": len(expanded), "layers": LAYERS,
        "declared_context_tokens": CONTEXT,
        "description": "Untrained F32 CPU Qwen3MoE; repeated real nonzero fixture layers, deterministic A output",
    }


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: str
    seconds: float

    def json(self):
        return json.loads(self.body)


class Job:
    """Continuously read the response so slow-client backpressure is not a delay hook."""
    def __init__(self, port: int, body: dict):
        self.started = threading.Event()
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.response: Response | None = None
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.error: BaseException | None = None
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.socket.settimeout(20)
        raw = json.dumps(body).encode()
        headers = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                   f"Authorization: Bearer {KEY}\r\nContent-Type: application/json\r\n"
                   f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n").encode()
        self.socket.sendall(headers + raw)
        self.thread = threading.Thread(target=self._read, name="scheduling-http-reader", daemon=True)
        self.thread.start()

    def _read(self):
        response = None
        start = time.monotonic()
        try:
            response = http.client.HTTPResponse(self.socket)
            response.begin()
            self.status = response.status
            self.headers = {k.lower(): v for k, v in response.getheaders()}
            self.started.set()
            body = response.read(16 * 1024 * 1024 + 1)
            check(len(body) <= 16 * 1024 * 1024, "test response exceeded its bounded capture")
            self.response = Response(self.status, self.headers, body.decode(), time.monotonic() - start)
        except BaseException as error:
            self.error = error
        finally:
            if response is not None:
                response.close()
            self.socket.close()
            self.done.set()

    def wait_started(self, timeout: float = 10):
        check(self.started.wait(timeout), f"request did not receive headers: {self.error}")

    def wait(self, timeout: float = 10) -> Response:
        check(self.done.wait(timeout), "request did not complete before test deadline")
        if self.error:
            raise AssertionError(f"HTTP job failed: {self.error}")
        check(self.response is not None, "request completed without a response")
        return self.response

    def cancel(self):
        self.cancelled.set()
        if not self.done.is_set():
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.socket.close()
            self.thread.join(timeout=1)
        check(not self.thread.is_alive(), "HTTP reader did not stop during cleanup")


class Server:
    def __init__(self, binary: Path, fixture: Path, queue_capacity: int,
                 queue_timeout: int, start_loaded: bool):
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            self.port = reserve.getsockname()[1]
        self.log = tempfile.NamedTemporaryFile(mode="w+b", dir=fixture.parent,
                                               prefix="server-", suffix=".log", delete=False)
        self.jobs: list[Job] = []
        self.closed = False
        self.closed_logs = ""
        env = os.environ.copy()
        env["STRATA_API_KEY"] = env["FORGE9070_API_KEY"] = KEY
        command = [str(binary), "serve", "--cpu", "--port", str(self.port),
                   "--models-dir", str(fixture.parent), "--ctx", str(CONTEXT),
                   "--max-tokens", str(LONG_OUTPUT), "--threads", "1", "--ram-margin", "0",
                   "--queue-capacity", str(queue_capacity), "--queue-timeout", str(queue_timeout)]
        # Omit --parallel: this tests the real public default's CPU resolution.
        if start_loaded:
            command.extend(["--model", str(fixture)])
        self.command = command
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.log, stderr=self.log)
        self.model = None

    def request(self, path: str, body: dict | None = None, auth: bool = True,
                timeout: float = 3) -> Response:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Authorization": f"Bearer {KEY}"} if auth else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        start = time.monotonic()
        try:
            connection.request("POST" if body is not None else "GET", path,
                               json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            return Response(response.status, {k.lower(): v for k, v in response.getheaders()},
                            response.read().decode(), time.monotonic() - start)
        finally:
            connection.close()

    def status(self) -> dict:
        response = self.request("/v1/status")
        check(response.status == 200, f"status endpoint: {response.status} {response.body}")
        return response.json()

    def wait_state(self, predicate, description: str, timeout: float = 10) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            check(self.process.poll() is None, f"server exited while {description}: {self.logs()}")
            try:
                last = self.status()
                if predicate(last):
                    return last
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.01)
        raise AssertionError(f"deadline waiting for {description}; last state={last}; log={self.logs()[-4000:]}")

    def body(self, label: str, long: bool = False) -> dict:
        body = {"messages": [{"role": "user", "content": label}],
                "max_tokens": LONG_OUTPUT if long else 4, "stream": long,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        if self.model:
            body["model"] = self.model
        return body

    def job(self, label: str, long: bool = False) -> Job:
        job = Job(self.port, self.body(label, long))
        self.jobs.append(job)
        return job

    def long_active(self, label: str) -> Job:
        job = self.job(label, True)
        job.wait_started()
        check(job.status == 200, f"long request failed: {job.wait().body if job.done.is_set() else job.status}")
        self.wait_state(lambda s: s["admission"]["active"] == 1 and s["generating_requests"] == 1,
                        "one real CPU generation running")
        check(not job.done.is_set(), "synthetic decode completed before concurrency could be observed")
        return job

    def idle(self):
        return self.wait_state(lambda s: s["admission"]["active"] == 0
                               and s["admission"]["reserved"] == 0
                               and s["admission"]["queued"] == 0
                               and s["generating_requests"] == 0, "all generation work released")

    def logs(self) -> str:
        self.log.flush()
        # A separately opened reader cannot move the subprocess's write offset.
        with open(self.log.name, "rb") as reader:
            return reader.read(4 * 1024 * 1024).decode(errors="replace")

    def starts(self) -> int:
        return sum(line.startswith("request: POST /v1/chat/completions ") for line in self.logs().splitlines())

    def close(self):
        if self.closed:
            return self.closed_logs
        errors = []
        for job in self.jobs:
            try:
                job.cancel()
            except BaseException as error:
                errors.append(str(error))
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        logs = self.logs()
        self.log.close()
        Path(self.log.name).unlink(missing_ok=True)
        self.closed, self.closed_logs = True, logs
        check(not errors, f"reader cleanup failed: {errors}")
        return logs


def assert_reply(response: Response):
    check(response.status == 200, f"recovery response: {response.status} {response.body}")
    data = response.json()
    check(data["choices"][0]["message"]["content"] == "AAAA", response.body)
    check(data["usage"]["completion_tokens"] == 4, response.body)
    check("x-forge-queue-ms" in response.headers, "missing admission wait response header")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/forge9070")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/tiny-qwen3moe.gguf")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/scheduling-api-0.7.0.json")
    args = parser.parse_args()
    binary = args.binary.resolve()
    report = {
        "schema": "forge9070.scheduling-api.v1", "version": "0.7.0", "status": "running",
        "binary": str(binary), "backend_tested": "cpu", "checks": [],
        "rx9070xt_gpu_tested": False, "four_gpu_sequences_tested": False,
        "qwen38_inference_tested": False, "performance_benchmark": False,
        "synchronization": "Observed live active/queued/status counters and actual HTTP responses; no product test-delay hooks",
        "scope": "CPU HTTP admission, queue bounds, FIFO, timeout and cancellation; not model quality or GPU throughput",
    }
    start = time.monotonic()
    servers = []

    def passed(name: str, **details):
        report["checks"].append({"name": name, "status": "passed", **details})

    try:
        check(binary.is_file(), f"build the binary first: {binary}")
        with tempfile.TemporaryDirectory(prefix="forge-scheduling-") as temporary:
            fixture = Path(temporary) / "scheduling-synthetic.gguf"
            report["fixture"] = expand_fixture(args.fixture.resolve(), fixture)
            server = Server(binary, fixture, queue_capacity=2, queue_timeout=30, start_loaded=False)
            servers.append(server)
            try:
                state = server.wait_state(lambda s: s["status"] == "no_model", "server without a loaded model")
                check(state["admission"]["capacity"] == 0, str(state))
                check(server.request("/v1/status", auth=False).status == 401, "status accepted a missing API key")
                passed("Authenticated status reports no-model readiness and zero execution capacity")
                response = server.request("/v1/chat/completions", server.body("before model load"))
                check(response.status == 503 and response.json()["error"]["code"] == "model_not_ready", response.body)
                passed("Generation without a model returns structured HTTP 503")
                catalog = server.request("/api/models").json()["models"]
                check(len(catalog) == 1, f"unexpected temporary model catalog: {catalog}")
                loaded = server.request("/api/load", {"model": catalog[0]["id"]}, timeout=30)
                check(loaded.status == 200, loaded.body)
                state = server.wait_state(lambda s: s["status"] == "ready", "CPU model load")
                server.model = state["model"]
                check(state["backend"] == "cpu" and state["admission"]["capacity"] == 1, str(state))
                check(state["concurrency"]["mode"] == "auto" and state["concurrency"]["slots"] == 1, str(state))
                check(state["admission"]["queue_capacity"] == 2 and state["admission"]["queue_timeout_seconds"] == 30, str(state))
                passed("Default auto concurrency resolves to one active CPU request", status_snapshot=state)

                first = server.long_active("a")
                second = server.job("b", True)
                server.wait_state(lambda s: s["admission"]["queued"] == 1, "second request queued")
                third = server.job("c")
                server.wait_state(lambda s: s["admission"]["queued"] == 2, "third request queued")
                response = server.request("/v1/chat/completions", server.body("overflow request"))
                check(response.status == 429 and response.json()["error"]["code"] == "queue_full", response.body)
                check(int(response.headers.get("retry-after", "0")) > 0, str(response.headers))
                state = server.status()
                check(state["admission"]["rejected_full_total"] == 1 and state["admission"]["queued"] == 2, str(state))
                passed("Full generation queue returns HTTP 429 with queue_full and Retry-After", admission=state["admission"])

                response = server.request("/v1/chat/completions", {"messages": []})
                check(response.status == 400, response.body)
                check(server.status()["admission"]["rejected_full_total"] == 1, "malformed request entered the full generation queue")
                passed("Malformed chat is rejected before generation admission while the queue is full")
                response = server.request("/v1/status")
                check(response.status == 200 and response.seconds < 2, f"busy status response: {response.seconds}s")
                check(response.json()["generating_requests"] == 1 and not first.done.is_set(), response.body)
                passed("Status remains responsive during real CPU decoding and a full queue", response_seconds=response.seconds)

                first.cancel()
                second.wait_started()
                check(second.status == 200 and not second.done.is_set(), "second long request did not take the next slot")
                state = server.wait_state(lambda s: s["admission"]["active"] == 1 and s["admission"]["queued"] == 1,
                                          "older second request active and third still queued")
                check(not third.started.is_set(), "third request bypassed the older second request")
                second.cancel()
                assert_reply(third.wait())
                server.idle()
                passed("FIFO is observable over HTTP: second queued generation starts before the third", observed_middle_state=state["admission"])

                starts = server.starts()
                cancelled_before = server.status()["admission"]["cancelled_total"]
                active = server.long_active("d")
                cancelled = server.job("this cancelled queued request must never execute")
                server.wait_state(lambda s: s["admission"]["queued"] == 1, "request queued before disconnect")
                cancelled.cancel()
                state = server.wait_state(lambda s: s["admission"]["queued"] == 0
                                          and s["admission"]["cancelled_total"] == cancelled_before + 1,
                                          "disconnected waiter removed and counted")
                check(not cancelled.started.is_set() and not active.done.is_set(), "cancelled waiter acquired a response or active decode ended early")
                active.cancel()
                server.idle()
                assert_reply(server.request("/v1/chat/completions", server.body("recovery after queued disconnect"), timeout=10))
                server.idle()
                check(server.starts() - starts == 2, "a queued disconnected request reached generation; unexpected engine start log")
                passed("Queued disconnect cancels once, never reaches generation, and capacity recovers",
                       cancelled_total=state["admission"]["cancelled_total"], engine_start_log_delta=2)
            finally:
                report.setdefault("server_logs", []).append(server.close()[-12000:])
                servers.remove(server)

            server = Server(binary, fixture, queue_capacity=1, queue_timeout=1, start_loaded=True)
            servers.append(server)
            try:
                state = server.wait_state(lambda s: s["status"] == "ready", "timeout-test CPU model load", timeout=30)
                server.model = state["model"]
                starts = server.starts()
                active = server.long_active("t")
                timeout_job = server.job("expire before admission")
                server.wait_state(lambda s: s["admission"]["queued"] == 1, "timeout request queued")
                response = timeout_job.wait(timeout=5)
                check(response.status == 503 and response.json()["error"]["code"] == "queue_timeout", response.body)
                check(int(response.headers.get("retry-after", "0")) > 0, str(response.headers))
                state = server.status()
                check(state["admission"]["timed_out_total"] == 1 and state["admission"]["queued"] == 0, str(state))
                check(state["admission"]["active"] == 1 and not active.done.is_set(), "active decode did not span the queued deadline")
                check(server.starts() - starts == 1, "expired request reached inference")
                passed("Waiting deadline returns HTTP 503 queue_timeout without executing the expired request",
                       response_seconds=response.seconds, admission=state["admission"], engine_start_log_delta=1)
                active.cancel()
                server.idle()
                assert_reply(server.request("/v1/chat/completions", server.body("recovery after deadline"), timeout=10))
                state = server.idle()
                check(state["admission"]["timed_out_total"] == 1 and state["admission"]["cancelled_total"] == 0, str(state))
                passed("Timeout and active cancellation release capacity; subsequent real inference returns AAAA")
            finally:
                report.setdefault("server_logs", []).append(server.close()[-12000:])
                servers.remove(server)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for server in servers:
            try:
                report.setdefault("server_logs", []).append(server.close()[-12000:])
            except BaseException as error:
                report.setdefault("cleanup_errors", []).append(str(error))
        report["elapsed_seconds"] = time.monotonic() - start
        report["passed_checks"] = sum(c["status"] == "passed" for c in report["checks"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "passed_checks": report["passed_checks"],
                          "evidence": str(args.output.resolve()), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
