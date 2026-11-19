#!/usr/bin/env python3
"""Exercise the independent Forge executable with untrained CPU fixtures.

This validates CLI selection, strict capability failures, actual generation,
and benchmark accounting. It is not a GPU or pretrained-model quality test.
Full numerical cross-implementation parity lives in tests/new_engine_parity.rs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/forge-engine")
    parser.add_argument("--frontend", type=Path, help="Also validate forge9070 engine dispatch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert binary.is_file(), f"Build the independent executable first: {binary}"
    fixture = ROOT / "tests/fixtures/tiny-qwen3moe.gguf"
    checks: list[str] = []
    skipped: list[str] = []
    executions: list[dict] = []

    def run(arguments: list[str], *, success: bool = True, stdin: str | None = None,
            executable: Path = binary, prefix: tuple[str, ...] = (),
            environment: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        command = [str(executable), *prefix, *arguments]
        started = time.monotonic()
        # Text-mode stdin translates LF to CRLF on Windows. Send exact UTF-8
        # bytes so token counts describe the requested prompt on every platform.
        result = subprocess.run(command, input=None if stdin is None else stdin.encode("utf-8"),
                                capture_output=True, timeout=90, env=environment)
        result.stdout = result.stdout.decode("utf-8")
        result.stderr = result.stderr.decode("utf-8")
        executions.append({"command": command, "exit_code": result.returncode,
                           "elapsed_seconds": time.monotonic() - started,
                           "stdout": result.stdout, "stderr": result.stderr})
        assert (result.returncode == 0) == success, (
            f"Unexpected result for {command}: {result.returncode}\n{result.stderr}\n{result.stdout}")
        return result

    def document(arguments: list[str], **kwargs) -> dict:
        return json.loads(run(arguments, **kwargs).stdout)

    capabilities = document(["capabilities"])
    assert capabilities["engine"] == "forge-native"
    assert capabilities["independent_runtime"] is True
    assert capabilities["rusting_strata_dependency"] is False
    assert capabilities["fallback"] == "auto_selection_only_no_runtime_or_external_fallback"
    assert capabilities["backends"] == ["cpu", "vulkan", "hybrid"]
    assert capabilities["backend_policies"] == ["auto", "cpu", "vulkan", "hybrid"]
    assert capabilities["gpu_vendor_policy"] == "capabilities_not_vendor_or_model"
    assert capabilities["auto_backend"]["model_or_execution_errors_trigger_fallback"] is False
    assert capabilities["vulkan"] and capabilities["qwen3_8"] and capabilities["gguf_shards"]
    assert capabilities["http_server"] and capabilities["tool_calling_api"]
    assert not capabilities["gpu_performance_verified"]
    assert capabilities["moe_loading"]["selected_experts_omitted"] == 0
    checks.append("Capabilities identify implemented independent backends separately from unverified target-GPU performance")

    inspected = document(["inspect", "--model", str(fixture), "--ctx", "64"])
    assert inspected["engine"] == capabilities["engine"]
    assert inspected["compute_submitted"] is False
    assert inspected["weight_file_bytes"] == fixture.stat().st_size
    assert inspected["fallback"] == "none"
    checks.append("Independent loader validates the actual committed GGUF and reports inspection without compute")

    generation = ["generate", "--cpu", "--model", str(fixture), "--ctx", "64",
                  "--prompt", "hello", "--max-tokens", "4", "--json"]
    generated = document(generation)
    assert generated["text"] == "AAAA" and generated["token_ids"] == [65] * 4
    assert generated["runtime"]["execution"]["backend"] == "cpu"
    assert generated["runtime"]["execution"]["fallback"] == "none"
    assert generated["runtime"]["backend_selection"]["requested"] == "cpu"
    assert generated["runtime"]["backend_selection"]["selected"] == "cpu"
    assert generated["prompt_tokens"] == 5 and generated["generated_tokens"] == 4
    assert generated["decode_steps"] == 3 and generated["finish_reason"] == "length"
    assert generated["kv_precision"] == "F32" and generated["gpu_performance_verified"] is False
    for field in ["load_seconds", "tokenize_seconds", "prefill_seconds", "decode_seconds"]:
        assert math.isfinite(generated[field]) and generated[field] >= 0
    assert math.isclose(generated["decode_steps_per_second"],
                        generated["decode_steps"] / generated["decode_seconds"], rel_tol=1e-10)
    checks.append("Actual MoE CPU generation produces complete output and distinguishes output tokens from decode steps")

    stdin_generation = generation.copy()
    stdin_generation[stdin_generation.index("--prompt") + 1] = "-"
    for text in ("£ café\n日本語 😀", "£ café\r\n日本語 😀"):
        unicode_result = document(stdin_generation, stdin=text)
        assert unicode_result["prompt_tokens"] == len(text.encode("utf-8"))
        assert unicode_result["text"] == "AAAA"
    checks.append("Standard-input Unicode prompts preserve UTF-8 bytes and LF/CRLF line endings")

    denied = run([argument for argument in generation if argument != "--cpu"], success=False)
    assert "--cpu" in denied.stderr and "fallback" in denied.stderr.lower()
    checks.append("Backend selection is explicit and never silently falls back to Strata")

    exhausted = generation.copy()
    exhausted[exhausted.index("--ctx") + 1] = "8"
    rejected = run(exhausted, success=False)
    assert "context" in rejected.stderr.lower()
    checks.append("Prompt plus requested output must fit the configured context before generation")

    empty = generation.copy()
    empty[empty.index("--prompt") + 1] = ""
    assert "prompt" in run(empty, success=False).stderr.lower()
    zero = generation.copy()
    zero[zero.index("--max-tokens") + 1] = "0"
    assert "positive" in run(zero, success=False).stderr.lower()
    checks.append("Empty prompts and zero output reservations are rejected")

    with tempfile.TemporaryDirectory(prefix="forge-native-cli-") as directory:
        temporary = Path(directory)
        # These loader overrides affect only child processes and deliberately
        # expose no Vulkan ICD. Actual CPU inference must work on that host.
        no_vulkan = dict(os.environ,
                         VK_DRIVER_FILES=str(temporary / "missing-vulkan-driver.json"),
                         VK_ICD_FILENAMES=str(temporary / "missing-vulkan-driver.json"))
        auto_generation = ["generate", "--backend", "auto", *generation[2:]]
        automatic = document(auto_generation, environment=no_vulkan)
        selection = automatic["runtime"]["backend_selection"]
        assert selection["requested"] == "auto" and selection["selected"] == "cpu"
        assert selection["automatic"] is True and selection["reason"]
        assert automatic["runtime"]["execution"]["backend"] == "cpu"
        assert automatic["runtime"]["execution"]["fallback"] == "auto_selection_cpu"
        assert automatic["text"] == generated["text"]
        assert automatic["token_ids"] == generated["token_ids"]
        # Automatic CPU selection must reserve the same host state as explicit CPU.
        assert automatic["runtime"]["host_memory_plan"]["state_reserve_bytes"] == generated["runtime"]["host_memory_plan"]["state_reserve_bytes"]
        checks.append("Auto without Vulkan executes CPU inference, reports its reason and uses the CPU host-state reservation")

        hardware = document(["doctor"], environment=no_vulkan)
        assert hardware["cpu_available"] is True
        assert hardware["adapters"] == []
        assert hardware["compute_submitted"] is False
        checks.append("Hardware doctor remains readable on a host without a Vulkan driver")

        explicit_gpu = auto_generation.copy()
        explicit_gpu[explicit_gpu.index("--backend") + 1] = "vulkan"
        failed_gpu = run(explicit_gpu, environment=no_vulkan, success=False)
        assert "auto selected CPU" not in failed_gpu.stderr
        for extra in (["--device", "999"], ["--vram-mib", "1"], ["--allow-software-vulkan"]):
            run(auto_generation + extra, environment=no_vulkan, success=False)
        checks.append("Explicit Vulkan and auto with explicit GPU settings fail instead of silently ignoring the requested configuration")

        source = fixture.read_bytes()
        architecture = b"qwen3moe"
        encoded = len(architecture).to_bytes(8, "little") + architecture
        assert source.count(encoded) == 1
        unsupported = temporary / "unsupported-qwen38.gguf"
        unsupported.write_bytes(source.replace(encoded, len(architecture).to_bytes(8, "little") + b"qwen4exp", 1))
        result = run(["inspect", "--model", str(unsupported), "--ctx", "64"], success=False)
        assert "qwen4exp" in result.stderr
        checks.append("Relabeling a Qwen3 fixture as qwen4exp fails mandatory architecture-metadata validation")

        truncated = temporary / "truncated.gguf"
        truncated.write_bytes(source[:-4])
        run(["inspect", "--model", str(truncated), "--ctx", "64"], success=False)
        bad_auto = auto_generation.copy()
        bad_auto[bad_auto.index("--model") + 1] = str(truncated)
        run(bad_auto, environment=no_vulkan, success=False)
        checks.append("Truncated GGUF tensor payload is rejected by the independent loader")

        report = temporary / "benchmark.json"
        bench = ["--threads", "2", "bench", "--cpu", "--model", str(fixture),
                 "--ctx", "64", "--prompt", "hello", "--decode-steps", "8",
                 "--reps", "2", "--warmup", "1", "--requests", "4", "--json-out", str(report)]
        measured = document(bench)
        assert json.loads(report.read_text()) == measured
        assert measured["runtime"]["execution"]["backend"] == "cpu" and measured["concurrent_requests"] == 4
        assert measured["worker_threads"] == 2
        assert measured["warmup_iterations_excluded"] == 1
        assert measured["repetitions"] == len(measured["samples"]) == 2
        assert measured["prompt_tokens_per_request"] == 5
        assert measured["decode_steps_per_request"] == 8
        assert measured["identical_greedy_checksums"] is True
        checksum = 0xCBF29CE484222325
        for _ in range(8):
            checksum = ((checksum ^ 65) * 0x100000001B3) & ((1 << 64) - 1)
        rates = []
        for sample in measured["samples"]:
            assert sample["greedy_checksum"] == [checksum] * 4
            assert len(sample["request_prefill_seconds"]) == 4
            assert len(sample["request_decode_completion_seconds"]) == 4
            assert sample["prefill_seconds"] > 0 and sample["decode_seconds"] > 0
            assert math.isclose(sample["aggregate_prefill_tokens_per_second"],
                                5 * 4 / sample["prefill_seconds"], rel_tol=1e-10)
            assert math.isclose(sample["aggregate_decode_steps_per_second"],
                                8 * 4 / sample["decode_seconds"], rel_tol=1e-10)
            assert all(0 <= duration <= sample["decode_seconds"]
                       for duration in sample["request_decode_completion_seconds"])
            rates.append(sample["aggregate_decode_steps_per_second"])
        assert measured["median_aggregate_decode_steps_per_second"] == statistics.median(rates)
        checks.append("Four independent CPU sequences execute within two worker threads with reproducible token checksums")
        checks.append("Benchmark excludes warmup/load/tokenization and derives rates from paired measured durations and executed steps")
        checks.append("Saved benchmark JSON exactly matches the reported samples")

        invalid = bench.copy()
        invalid[invalid.index("--requests") + 1] = "5"
        assert "--requests" in run(invalid, success=False).stderr
        invalid = bench.copy()
        invalid[invalid.index("--warmup") + 1] = "0"
        assert "--warmup" in run(invalid, success=False).stderr
        checks.append("Invalid concurrency and warmup settings fail explicitly")

        # Exercise the real CLI with disposable checkpoint bytes. A report
        # must never truncate its own live model mapping, even through a link.
        protected_model = temporary / "protected-model.gguf"
        protected_model.write_bytes(source)
        aliases_input = bench.copy()
        aliases_input[aliases_input.index("--model") + 1] = str(protected_model)
        aliases_input[aliases_input.index("--json-out") + 1] = str(protected_model)
        refused = run(aliases_input, success=False)
        assert "--json-out" in refused.stderr and "overwrite" in refused.stderr
        assert protected_model.read_bytes() == source
        checks.append("Benchmark report output equal to the input checkpoint is rejected with all model bytes preserved")

        linked_report = temporary / "linked-report.json"
        try:
            os.link(protected_model, linked_report)
        except OSError as error:
            skipped.append(f"Hard-link output alias check unavailable on this filesystem: {error}")
        else:
            aliases_input[aliases_input.index("--json-out") + 1] = str(linked_report)
            refused = run(aliases_input, success=False)
            assert "--json-out" in refused.stderr and "overwrite" in refused.stderr
            assert protected_model.read_bytes() == source and linked_report.read_bytes() == source
            checks.append("Benchmark report output hard-linked to the input checkpoint is rejected without modifying either path")

    if args.frontend:
        frontend = args.frontend.resolve()
        assert frontend.is_file()
        forwarded = document(["capabilities"], executable=frontend, prefix=("engine",))
        assert forwarded == capabilities
        via_frontend = document(generation, executable=frontend, prefix=("engine",))
        assert via_frontend["engine"] == "forge-native" and via_frontend["token_ids"] == [65] * 4
        checks.append("Forge9070 engine subcommand selects the independent engine and executes generation")

    result = {"status": "passed", "checks_passed": len(checks), "checks": checks,
              "checks_skipped": len(skipped), "skipped": skipped,
              "scope": "Actual standalone CPU CLI and synthetic-model execution; not HTTP, GPU or pretrained-model quality",
              "gpu_tested": False, "qwen38_tested": False,
              "checksums_do_not_replace_numerical_parity": True,
              "executions": executions}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "executions"}, indent=2))


if __name__ == "__main__":
    main()
