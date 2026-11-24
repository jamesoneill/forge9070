#!/usr/bin/env python3
"""Pure validation tests for the native concurrency sweep; no real GPU required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

module_path=Path(__file__).with_name("benchmark-concurrency.py")
spec=importlib.util.spec_from_file_location("nextora_benchmark_sweep",module_path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def example(slots=1):
    return {
        "engine":"nextora-native",
        "identical_greedy_checksums":True,
        "concurrent_requests":slots,
        "prompt_tokens_per_request":128,
        "decode_steps_per_request":32,
        "repetitions":5,
        "runtime":{"execution":{"backend":"cpu"}},
        "median_aggregate_prefill_tokens_per_second":110.0*slots,
        "median_aggregate_decode_steps_per_second":11.0*slots,
        "p95_request_decode_completion_seconds":3.0+slots,
    }

class SweepTests(unittest.TestCase):
    def test_successful_native_measurements(self):
        records=[module.validate_run(example(n),n) for n in (1,2,4)]
        comparison=module.compare(records)
        self.assertEqual(comparison["decode_rate_ratios_vs_one"]["4"],4.0)
        self.assertFalse(comparison["proven_optimal"])

    def test_rejects_invalid_identity_and_checksums(self):
        report=example()
        report["engine"]="other"
        with self.assertRaises(ValueError):
            module.validate_run(report,1)
        report=example()
        report["identical_greedy_checksums"]=False
        with self.assertRaises(ValueError):
            module.validate_run(report,1)

    def test_rejects_non_finite_rates_and_backend_changes(self):
        import math
        report=example()
        report["median_aggregate_prefill_tokens_per_second"]=math.nan
        with self.assertRaises(ValueError):
            module.validate_run(report,1)
        records=[module.validate_run(example(n),n) for n in (1,2,4)]
        records[1]["backend"]="vulkan"
        with self.assertRaises(ValueError):
            module.compare(records)

    def test_writes_json_without_prompt_text(self):
        with tempfile.TemporaryDirectory() as td:
            report=Path(td)/"report.json"
            module.save_report(report,{"schema":"nextora.benchmark_sweep.v1","prompt_text_exported":False})
            import json
            self.assertEqual(json.loads(report.read_text())["schema"],"nextora.benchmark_sweep.v1")

if __name__=="__main__":
    unittest.main()
