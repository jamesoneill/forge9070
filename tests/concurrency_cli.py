#!/usr/bin/env python3
"""Exercise overlapping HTTP clients, bounded admission and truthful CPU slots.

This executes the real CPU model. It establishes concurrent network handling and
sequential CPU inference; GPU multi-sequence compute requires a target GPU test.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
KEY = 'forge9070-concurrency-local-test'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'target/debug/forge9070')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/tiny-qwen3moe.gguf')
    args = parser.parse_args()
    binary, fixture = str(args.binary.resolve()), str(args.fixture.resolve())
    checks: list[str] = []
    for flags, message in [(['--cpu', '--parallel', '2'], 'requires GPU'),
                           (['--cpu', '--mtp', '1'], 'requires GPU'),
                           (['--parallel', '2', '--mtp', '1'], 'cannot run together'),
                           (['--expert-tail-min-rank', '8'], 'expert omission is disabled')]:
        rejected = subprocess.run([binary, 'serve', '--model', fixture, *flags],
                                  capture_output=True, text=True, timeout=10, cwd=ROOT)
        assert rejected.returncode != 0 and message in rejected.stderr, rejected.stderr
    checks.append('Unsupported CPU concurrency/MTP and conflicting GPU options fail explicitly')

    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]

    def request(path: str):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        try:
            connection.request('GET', path, headers={'Authorization': f'Bearer {KEY}'})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), json.loads(response.read())
        finally:
            connection.close()

    def held_request(body: dict) -> tuple[socket.socket, bytes]:
        """Send complete headers, then hold the body to make overlap observable."""
        data = json.dumps(body).encode()
        client = socket.create_connection(('127.0.0.1', port), timeout=10)
        client.settimeout(10)
        headers = (f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                   f'Authorization: Bearer {KEY}\r\nContent-Type: application/json\r\n'
                   f'Content-Length: {len(data)}\r\nConnection: close\r\n\r\n')
        client.sendall(headers.encode())
        return client, data

    def wait_until(predicate, detail: str):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, detail
            time.sleep(0.01)

    with tempfile.TemporaryFile(mode='w+b') as log:
        env = dict(os.environ, STRATA_API_KEY=KEY)
        process = subprocess.Popen([binary, 'serve', '--model', fixture, '--cpu',
                                    '--parallel', '1', '--ctx', '512', '--max-tokens', '64',
                                    '--max-waiting', '8', '--queue-capacity', '4', '--threads', '2', '--ram-margin', '0',
                                    '--port', str(port)], cwd=ROOT, env=env, stdout=log, stderr=log)
        held: list[tuple[socket.socket, bytes]] = []
        try:
            def ready():
                assert process.poll() is None, 'server exited before readiness'
                try:
                    return request('/health')[0] == 200
                except (OSError, http.client.HTTPException):
                    return False
            wait_until(ready, 'server startup timed out')
            _, _, stats = request('/api/stats')
            assert stats['backend'] == 'cpu' and stats['slots'] == 1, stats
            model_id = request('/v1/models')[2]['data'][0]['id']
            checks.append('CPU server reports one compute slot rather than parallel GPU execution')

            def body(index: int, stream: bool = False):
                return {'model': model_id,
                        'messages': [{'role': 'user', 'content': f'concurrent client {index}'}],
                        'max_tokens': 64, 'temperature': 0, 'stream': stream,
                        'chat_template_kwargs': {'enable_thinking': False}}

            held = [held_request(body(i, i % 2 == 1)) for i in range(4)]
            wait_until(lambda: request('/health')[2]['waiting'] == 4,
                       'four open clients were not observed')
            checks.append('Four overlapping clients are admitted while health remains responsive')

            barrier = threading.Barrier(4)
            def complete(item):
                index, (client, data) = item
                try:
                    barrier.wait(timeout=10)
                    client.sendall(data)
                    response = http.client.HTTPResponse(client)
                    response.begin()
                    content = response.read().decode()
                    assert response.status == 200, (response.status, content)
                    if index % 2:
                        chunks = [line[6:] for line in content.splitlines() if line.startswith('data: ')]
                        assert chunks[-1] == '[DONE]', content
                        events = [json.loads(chunk) for chunk in chunks[:-1]]
                        text = ''.join(choice.get('delta', {}).get('content') or ''
                                       for event in events for choice in event.get('choices', []))
                        assert any(choice.get('finish_reason') == 'length'
                                   for event in events for choice in event.get('choices', [])), content
                    else:
                        result = json.loads(content)
                        text = result['choices'][0]['message']['content']
                        assert result['usage']['completion_tokens'] == 64, result
                        assert result['choices'][0]['finish_reason'] == 'length', result
                    assert text == 'A' * 64, (index, text)
                finally:
                    client.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(complete, enumerate(held)))
            held.clear()
            wait_until(lambda: request('/api/stats')[2]['requests'] == 4,
                       'completed requests were not counted')
            _, _, stats = request('/api/stats')
            assert stats['completion_tokens'] == 256 and stats['slots'] == 1, stats
            checks.append('Four concurrent clients receive isolated complete replies, including two SSE streams')
            checks.append('All four requests and 256 completion tokens are counted exactly once')

            # The cap includes sockets still sending their bodies. Holding all
            # eight makes overload reproducible without large/slow inference.
            held = [held_request(body(i)) for i in range(8)]
            status, headers, overloaded = request('/health')
            assert status == 503 and overloaded['error']['type'] == 'overloaded_error', overloaded
            assert headers.get('Retry-After') == '2', headers
            checks.append('Bounded admission returns HTTP 503 with Retry-After when all eight connections are held')
            for client, _ in held:
                client.close()
            held.clear()
            wait_until(lambda: request('/health')[0] == 200, 'server did not recover after disconnects')
            checks.append('Disconnected incomplete clients release capacity and service recovers')

            with tempfile.TemporaryDirectory(prefix='forge9070 concurrency report ') as directory:
                report_path = Path(directory) / 'report.json'
                benchmark = subprocess.run(['python3', 'scripts/bench-concurrency.py',
                                            '--server', f'http://127.0.0.1:{port}',
                                            '--clients', '2', '--requests', '4', '--max-tokens', '4',
                                            '--warmup', '1', '--timeout', '10',
                                            '--prompt', 'hello', '--json-out', str(report_path)],
                                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                assert benchmark.returncode == 0, benchmark.stderr
                report = json.loads(benchmark.stdout)
                assert report == json.loads(report_path.read_text())
                assert report['status'] == 'passed' and report['server_compute']['slots'] == 1, report
                assert report['server_compute']['backend'] == 'cpu', report
                assert report['server_compute']['mtp_drafts'] == 0, report
                assert report['server_compute']['expert_tail_min_rank'] == 0, report
                assert report['server_compute']['run_options']['kv'] == 'f16', report
                assert report['server_compute']['run_options']['threads'] == 2, report
                assert report['measured_requests'] == 4 and report['warmup_requests_excluded'] == 1, report
                assert report['actual_completion_tokens'] == 16, report
                assert math.isclose(report['aggregate_output_tokens_per_second'],
                                    16 / report['measured_wall_seconds'], rel_tol=1e-9), report
                assert len(report['requests']) == 4 and all(sample['latency_seconds'] > 0
                                                           for sample in report['requests']), report
            checks.append('Concurrency benchmark measures actual usage, excludes warmup, labels aggregate rate and reports one CPU slot')

            # Exercise the real management API: a model ID may stay unchanged
            # while a reload changes execution precision. The benchmark must
            # see that difference before claiming a stable configuration.
            module_spec = importlib.util.spec_from_file_location('forge_bench', ROOT / 'scripts/bench-concurrency.py')
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            client = module.Client(f'http://127.0.0.1:{port}', 10)
            client.key = KEY
            before_options = client.snapshot()
            client.request('/api/load', {'model': model_id, 'options': {'kv': 'q8'}})
            after_options = client.snapshot()
            assert client.active_model() == model_id
            assert before_options != after_options
            assert after_options['run_options']['kv'] == 'q8'
            client.request('/api/load', {'model': model_id, 'options': {'kv': 'f16'}})
            checks.append('Benchmark configuration snapshot detects a KV change even when the active model ID stays the same')
        except BaseException:
            log.seek(0)
            print(log.read().decode(errors='replace'))
            raise
        finally:
            for client, _ in held:
                client.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    print(json.dumps({'status': 'passed', 'checks': checks, 'overlapping_clients': 4,
                      'compute_backend_tested': 'cpu', 'compute_slots_tested': 1,
                      'gpu_parallel_compute_tested': False, 'wsl_kernel_tested': False}, indent=2))


if __name__ == '__main__':
    main()
