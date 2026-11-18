#!/usr/bin/env python3
"""Measure the active server's aggregate throughput and per-request latency.

Configure the server's --parallel separately. This client never loads/unloads a
model or changes server settings. It sends the same prompt in each request and
records actual usage counts, including any prompt-cache reuse. Aggregate output
tokens/second includes queueing and prefill; it is not individual-stream speed.
Uses Python's standard library and STRATA_API_KEY for optional authentication.
Use an administrative key so execution settings can be checked before and after.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import ssl
import statistics
import sys
import time
from urllib.parse import urlsplit


class ApiError(RuntimeError):
    def __init__(self, path: str, status: int, message: str):
        self.status = status
        super().__init__(f'{path}: HTTP {status}: {message}')


def bounded(name: str, low: int, high: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f'{name} must be an integer') from error
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f'{name} must be between {low} and {high}')
        return number
    return parse


class Client:
    def __init__(self, server: str, timeout: int):
        try:
            url = urlsplit(server)
            port = url.port
        except ValueError as error:
            raise ValueError('--server must be an HTTP(S) root URL') from error
        if (url.scheme not in {'http', 'https'} or not url.hostname
                or url.username is not None or url.password is not None
                or url.path not in {'', '/'} or url.query or url.fragment):
            raise ValueError('--server must be an HTTP(S) root URL without credentials or /v1')
        if port is not None and not 1 <= port <= 65535:
            raise ValueError('--server port must be between 1 and 65535')
        self.url = url
        self.port = port
        self.timeout = timeout
        self.key = os.environ.get('STRATA_API_KEY')
        self.tls = ssl.create_default_context() if url.scheme == 'https' else None

    def request(self, path: str, body: dict | None = None) -> dict:
        # A new direct connection per request measures end-to-end API latency.
        # In particular, local API traffic does not enter environment proxies.
        if self.tls is not None:
            connection = http.client.HTTPSConnection(self.url.hostname, self.port,
                                                    timeout=self.timeout, context=self.tls)
        else:
            connection = http.client.HTTPConnection(self.url.hostname, self.port,
                                                   timeout=self.timeout)
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        try:
            connection.request('POST' if body is not None else 'GET', path,
                               json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise RuntimeError(f'{path}: response exceeds the 32 MiB client limit')
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as error:
                raise RuntimeError(f'{path}: HTTP {response.status} response was not JSON') from error
            if not isinstance(result, dict):
                raise RuntimeError(f'{path}: expected a JSON object')
            if not 200 <= response.status < 300:
                detail = result.get('error', {})
                message = detail.get('message', 'request failed') if isinstance(detail, dict) else 'request failed'
                # Redirects are errors; authorization is never sent to a new origin.
                raise ApiError(path, response.status, str(message)[:512])
            return result
        finally:
            connection.close()

    def active_model(self) -> str:
        models = self.request('/v1/models').get('data')
        if (not isinstance(models, list) or len(models) != 1
                or not isinstance(models[0], dict) or not isinstance(models[0].get('id'), str)):
            raise RuntimeError('The server must have exactly one active model; load one before benchmarking')
        return models[0]['id']

    def snapshot(self) -> dict:
        health = self.request('/health')
        slots = health.get('slots')
        if type(slots) is not int or slots < 1:
            raise RuntimeError('The server did not report a valid compute-slot count')
        snapshot = {'slots': slots, 'backend': None, 'device': None, 'context_tokens': None,
                    'mtp_drafts': None, 'expert_tail_min_rank': None}
        try:
            stats = self.request('/api/stats')
        except ApiError as error:
            if error.status not in {401, 403}:
                raise
        else:
            snapshot.update(backend=stats.get('backend'), device=stats.get('device'),
                            context_tokens=stats.get('ctx'), mtp_drafts=stats.get('mtp_drafts'))
            tail = stats.get('expert_tail')
            if isinstance(tail, dict):
                # Omission counters normally grow while benchmarking; only the
                # configured threshold belongs in configuration comparison.
                snapshot['expert_tail_min_rank'] = tail.get('min_rank')
        # Model ID alone cannot detect a reload with different KV precision,
        # context allocation or worker settings. Read the resolved runtime
        # options, not the frontend's advertised starting profile.
        options = self.request('/api/models').get('options')
        fields = ('ctx', 'kv', 'cache_ram', 'vram_margin', 'mtp', 'chunk', 'gpu',
                  'parallel', 'threads', 'llama_numerics', 'model_identity',
                  'decode_cache_policy')
        if not isinstance(options, dict) or any(options.get(key) is None for key in fields):
            raise RuntimeError('Complete execution settings are required; use the Forge9070 0.4 server and an administrative API key')
        snapshot['run_options'] = {key: options[key] for key in fields}
        return snapshot

    def complete(self, index: int, model: str, prompt: str, max_tokens: int) -> dict:
        start = time.perf_counter()
        result = self.request('/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': 0, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False},
        })
        elapsed = time.perf_counter() - start
        if result.get('model') != model:
            raise RuntimeError('The server answered with a different model; keep the active model unchanged during the benchmark')
        usage = result.get('usage', {})
        if not isinstance(usage, dict):
            raise RuntimeError('The server did not return a token-usage object')
        tokens = usage.get('completion_tokens')
        prompt_tokens = usage.get('prompt_tokens')
        if type(tokens) is not int or tokens < 0 or type(prompt_tokens) is not int or prompt_tokens < 0:
            raise RuntimeError('The server did not return valid actual token usage')
        prompt_details = usage.get('prompt_tokens_details', {})
        if not isinstance(prompt_details, dict):
            raise RuntimeError('The server returned invalid prompt-token details')
        cached = prompt_details.get('cached_tokens', 0)
        if type(cached) is not int or not 0 <= cached <= prompt_tokens:
            raise RuntimeError('The server returned invalid cached-token usage')
        timings = result.get('timings', {})
        if not isinstance(timings, dict):
            timings = {}
        decode_rate = timings.get('predicted_per_second')
        if (not isinstance(decode_rate, (int, float)) or isinstance(decode_rate, bool)
                or not math.isfinite(decode_rate) or decode_rate < 0):
            decode_rate = None
        choices = result.get('choices', [])
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError('The server did not return a completion choice')
        return {'request': index, 'completion_tokens': tokens, 'prompt_tokens': prompt_tokens,
                'cached_prompt_tokens': cached, 'latency_seconds': elapsed,
                'end_to_end_output_tokens_per_second': tokens / elapsed,
                'server_reported_decode_tokens_per_second': decode_rate,
                'finish_reason': choices[0].get('finish_reason')}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = math.floor(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8080')
    parser.add_argument('--clients', type=bounded('--clients', 1, 64), default=1)
    parser.add_argument('--requests', type=bounded('--requests', 1, 1024), default=8)
    parser.add_argument('--max-tokens', type=bounded('--max-tokens', 1, 16384), default=128)
    parser.add_argument('--warmup', type=bounded('--warmup', 0, 8), default=1,
                        help='Sequential warmup requests excluded from measured totals (default: 1)')
    parser.add_argument('--timeout', type=bounded('--timeout', 1, 7200), default=300,
                        help='Timeout for each HTTP request in seconds (default: 300)')
    parser.add_argument('--prompt', default='Explain how a CPU and a GPU can cooperate to run a mixture-of-experts language model.')
    parser.add_argument('--json-out', type=Path, help='Write the JSON report to this file as well as stdout')
    args = parser.parse_args()
    if args.requests < args.clients:
        parser.error('--requests must be at least --clients')
    prompt_bytes = args.prompt.encode('utf-8')
    if not prompt_bytes or len(prompt_bytes) > 1024 * 1024:
        parser.error('--prompt must contain between 1 byte and 1 MiB of UTF-8 text')
    client = Client(args.server, args.timeout)
    model = client.active_model()
    before = client.snapshot()
    for index in range(args.warmup):
        client.complete(index, model, args.prompt, args.max_tokens)
    samples = []
    failures = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = {pool.submit(client.complete, index, model, args.prompt, args.max_tokens): index
                   for index in range(args.requests)}
        for future in concurrent.futures.as_completed(futures):
            if future.cancelled():
                continue
            try:
                samples.append(future.result())
            except Exception as error:
                failures.append(f'request {futures[future]}: {error}')
                for pending in futures:
                    pending.cancel()
    wall = time.perf_counter() - started
    if failures:
        raise RuntimeError('Benchmark failed; no successful throughput claim is produced. ' + '; '.join(failures[:3]))
    after = client.snapshot()
    if client.active_model() != model or after != before:
        raise RuntimeError('The active model or server compute configuration changed during the benchmark; repeat with stable settings')
    assert len(samples) == args.requests
    samples.sort(key=lambda sample: sample['request'])
    tokens = sum(sample['completion_tokens'] for sample in samples)
    latencies = [sample['latency_seconds'] for sample in samples]
    report = {
        'schema': 'forge9070.concurrency-bench.v2', 'status': 'passed', 'model': model,
        'server': args.server.rstrip('/'), 'server_compute': before,
        'clients': args.clients, 'measured_requests': len(samples),
        'warmup_requests_excluded': args.warmup, 'requested_max_tokens_per_request': args.max_tokens,
        'actual_completion_tokens': tokens, 'measured_wall_seconds': wall,
        'aggregate_output_tokens_per_second': tokens / wall,
        'aggregate_rate_definition': 'actual completion tokens divided by full timed-run wall time, including queueing and prefill; not individual-stream decode speed',
        'request_latency_seconds': {'mean': statistics.mean(latencies), 'p50': percentile(latencies, .5),
                                    'p95': percentile(latencies, .95), 'max': max(latencies)},
        'prompt_reused_across_requests': True,
        'prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
        'cached_prompt_tokens': sum(sample['cached_prompt_tokens'] for sample in samples),
        'requests': samples,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.json_out is not None and args.json_out != Path('-'):
        args.json_out.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(1)
