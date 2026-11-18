#!/usr/bin/env python3
"""Verify real CPU benchmark accounting without making a hardware speed claim."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'target/debug/forge9070')
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix='forge9070 bench test ') as directory:
        output = Path(directory) / 'measured results.json'
        result = subprocess.run([
            str(args.binary.resolve()), 'bench', '--cpu',
            '--model', str(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf'),
            '--ctx', '128', '--pp', '16,32', '--tg', '32', '--reps', '2',
            '--warmup', '1', '--threads', '2', '--ram-margin', '0',
            '--json-out', str(output),
        ], capture_output=True, text=True, timeout=60, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        record = json.loads(output.read_text())
        assert record['backend'] == 'cpu' and record['kv'] == 'f16', record
        assert record['threads'] == 2 and record['mtp'] == 0, record
        assert record['approximate_expert_tail_enabled'] is False, record
        checks.append('Real CPU benchmark executes the MoE fixture and reports the requested fidelity configuration')

        samples = record['samples']
        assert record['warmup_repetitions_per_prompt_excluded'] == 1
        assert len(samples) == 4 and sum(s['completion_tokens'] for s in samples) == 128
        for length in (16, 32):
            rows = [s for s in samples if s['prompt_tokens'] == length]
            assert sorted(s['repetition'] for s in rows) == [0, 1]
            assert all(s['completion_tokens'] == 32 for s in rows)
        checks.append('Only measured repetitions enter JSON samples; warmup is excluded and 128 actual output tokens are counted')

        for sample in samples:
            assert all(math.isfinite(sample[key]) and sample[key] > 0
                       for key in ('prefill_seconds', 'decode_seconds', 'decode_tokens_per_second'))
            assert math.isclose(sample['decode_tokens_per_second'],
                                sample['completion_tokens'] / sample['decode_seconds'], rel_tol=1e-12)
        for group in record['results']:
            rows = [s for s in samples if s['prompt_tokens'] == group['pp']]
            rate = statistics.mean(s['decode_tokens_per_second'] for s in rows)
            assert math.isclose(group['decode_tok_s']['mean'], round(rate, 1), abs_tol=1e-9)
        checks.append('Per-run rates use paired actual token/time values and match the aggregate report')

        assert [p['tokens'] for p in record['prompts']] == [16, 32]
        hashes = [p['token_ids_sha256'] for p in record['prompts']]
        assert len(set(hashes)) == 2 and all(len(h) == 64 for h in hashes)
        assert isinstance(record['model']['cache_identity'], str) and record['model']['cache_identity']
        checks.append('Prompt token identities and the loaded checkpoint identity are recorded')

    print(json.dumps({'status': 'passed', 'checks': checks, 'compute_backend_tested': 'cpu',
                      'synthetic_model': True, 'gpu_calibration_tested': False,
                      'pretrained_model_accuracy_tested': False}, indent=2))


if __name__ == '__main__':
    main()
