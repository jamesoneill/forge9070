//! Local-only hardware advice backed by existing Nextora discovery and benchmark evidence.
//! No telemetry, model download, pricing feed, automatic tuning, or performance prediction.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{fs, path::Path, time::{SystemTime, UNIX_EPOCH}};

const MAX_REPORT_BYTES: u64 = 8 * 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;

/// Read only the existing native `nextora bench --json-out` schema.
/// Reject unsupported or unsuccessful measurements instead of treating them as proof.
pub fn read_benchmark(path: &Path) -> Result<Value> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("read benchmark metadata: {}", path.display()))?;
    ensure!(metadata.is_file(), "benchmark report is not a regular file");
    ensure!(metadata.len() <= MAX_REPORT_BYTES, "benchmark report exceeds 8 MiB limit");
    let data = fs::read(path)
        .with_context(|| format!("read benchmark: {}", path.display()))?;
    ensure!(data.len() as u64 <= MAX_REPORT_BYTES, "benchmark report exceeds 8 MiB limit");
    let report: Value = serde_json::from_slice(&data).context("parse native benchmark JSON")?;
    validate_benchmark(&report)?;
    Ok(report)
}

fn validate_benchmark(report: &Value) -> Result<()> {
    ensure!(
        report["engine"].as_str() == Some(crate::ENGINE_ID),
        "Insight requires a native Nextora bench report"
    );
    ensure!(
        report["identical_greedy_checksums"].as_bool() == Some(true),
        "benchmark must have passing repeated greedy-token checksums"
    );
    ensure!(
        report["repetitions"].as_u64().is_some_and(|n| (1..=100).contains(&n))
            && report["warmup_iterations_excluded"].as_u64().is_some_and(|n| (1..=20).contains(&n)),
        "benchmark repetitions/warmup missing or invalid"
    );
    ensure!(
        report["concurrent_requests"].as_u64().is_some_and(|n| (1..=4).contains(&n))
            && report["prompt_tokens_per_request"].as_u64().is_some_and(|n| n > 0)
            && report["decode_steps_per_request"].as_u64().is_some_and(|n| n > 0),
        "benchmark workload missing or invalid"
    );
    for key in [
        "median_aggregate_prefill_tokens_per_second",
        "median_aggregate_decode_steps_per_second",
        "p95_request_decode_completion_seconds",
    ] {
        ensure!(
            report[key].as_f64().is_some_and(|n| n.is_finite() && n > 0.0),
            "benchmark missing positive, finite {key}"
        );
    }
    ensure!(
        report.pointer("/runtime/execution/backend").and_then(Value::as_str).is_some(),
        "benchmark actual execution backend missing"
    );
    ensure!(
        report["samples"].as_array().is_some_and(|s| s.len() == report["repetitions"].as_u64().unwrap_or(0) as usize),
        "benchmark sample count does not match repetitions"
    );
    Ok(())
}

fn is_wsl() -> bool {
    #[cfg(target_os = "linux")]
    for file in ["/proc/sys/kernel/osrelease", "/proc/version"] {
        if let Ok(value) = fs::read_to_string(file) {
            let value = value.to_ascii_lowercase();
            if value.contains("microsoft") || value.contains("wsl") {
                return true;
            }
        }
    }
    false
}

fn host_snapshot() -> Value {
    let (available_ram_bytes, memory_error) = match crate::memory::available_memory() {
        Ok(bytes) => (Some(bytes), None),
        Err(error) => (None, Some(error.to_string())),
    };
    json!({
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "wsl": is_wsl(),
        "logical_cpu_threads": std::thread::available_parallelism().ok().map(|n| n.get()),
        "available_host_ram_bytes": available_ram_bytes,
        "host_ram_discovery_error": memory_error,
        "ram_scope": "currently available to this process, includes cgroup limit on Linux when detectable; not installed RAM",
    })
}

/// Snapshot hardware only. Benchmark measurement remains the separate, existing `bench` command.
pub fn collect(reports: &[Value]) -> Value {
    build(crate::runtime::hardware_report(), host_snapshot(), reports)
}

fn summary(report: &Value) -> Value {
    let adapter = report.pointer("/runtime/execution/adapter");
    json!({
        "backend": report.pointer("/runtime/execution/backend"),
        "adapter_name": adapter.and_then(|a| a.get("name")),
        "software_vulkan": adapter.and_then(|a| a.get("software")),
        "engine_version": report.get("engine_version"),
        "concurrent_requests": report.get("concurrent_requests"),
        "repetitions": report.get("repetitions"),
        "warmup_iterations_excluded": report.get("warmup_iterations_excluded"),
        "prompt_tokens_per_request": report.get("prompt_tokens_per_request"),
        "decode_steps_per_request": report.get("decode_steps_per_request"),
        "model_file_bytes": report.get("model_file_bytes"),
        "median_prefill_tokens_per_second": report.get("median_aggregate_prefill_tokens_per_second"),
        "median_decode_steps_per_second": report.get("median_aggregate_decode_steps_per_second"),
        "p95_request_decode_completion_seconds": report.get("p95_request_decode_completion_seconds"),
        "repeated_greedy_checksums_passed": true,
        "tiny_checkpoint_possible": report["model_file_bytes"].as_u64().is_some_and(|n| n < 16 * 1024 * 1024),
        "correctness_scope": "identical greedy tokens across repeats; does not establish pretrained quality or CPU/GPU numerical parity",
    })
}

fn advice(id: &str, priority: &str, evidence: &str, action: &str, purchasing: &str) -> Value {
    json!({
        "id": id,
        "priority": priority,
        "evidence": evidence,
        "action": action,
        "purchase": purchasing,
        "estimated_speedup": null,
        "basis": "hardware discovery or validated benchmark fields; no causal profiling or purchase-price data",
    })
}

fn recommendations(hardware: &Value, host: &Value, runs: &[Value]) -> Vec<Value> {
    let adapters = hardware["adapters"].as_array();
    let compatible: Vec<&Value> = adapters.into_iter().flatten()
        .filter(|a| a["compatible"].as_bool() == Some(true) && a["software"].as_bool() == Some(false))
        .collect();
    let wsl = host["wsl"].as_bool() == Some(true);
    let mut out = Vec::new();

    if compatible.is_empty() {
        let action = if wsl {
            "WSL has no eligible hardware Vulkan adapter. Try the native Windows Nextora executable with the Windows GPU driver; do not treat llvmpipe or /dev/dxg alone as GPU acceleration."
        } else {
            "No eligible hardware Vulkan adapter was detected. Inspect 'nextora doctor', driver and Vulkan loader installation before buying hardware; a discovery failure is not proof the GPU is inadequate."
        };
        out.push(advice(
            "restore-hardware-path", "high", "No eligible non-software Vulkan adapter in this process",
            action, "no_purchase",
        ));
    } else if !runs.iter().any(|r| r["backend"].as_str() == Some("vulkan")
        && r["software_vulkan"].as_bool() == Some(false)) {
        out.push(advice(
            "benchmark-vulkan-first", "high", "Eligible physical Vulkan adapter found; no real-device benchmark imported",
            "Run 'nextora bench' with --backend vulkan on a supported real GGUF and compare against a CPU run using identical workload, context, quantisation and driver settings.",
            "no_purchase",
        ));
    }

    if runs.is_empty() {
        out.push(advice(
            "record-baseline", "high", "No successful native model benchmark provided",
            "Run 'nextora bench --backend auto --reps 5 --warmup 1 --json-out bench.json' with a fixed model, prompt and context; then pass --bench-report bench.json to Insight.",
            "no_purchase",
        ));
    } else {
        if !runs.iter().any(|r| r["concurrent_requests"].as_u64().unwrap_or(0) > 1) {
            out.push(advice(
                "measure-concurrency", "medium", "Only single-request measurements supplied",
                "Compare --requests 1 and --requests 2 or 4 with the same checkpoint and prompt. Measure aggregate throughput, p95 completion and state-memory use before raising concurrency.",
                "no_purchase",
            ));
        }
        if runs.iter().any(|r| r["tiny_checkpoint_possible"].as_bool() == Some(true)) {
            out.push(advice(
                "real-model-baseline", "high", "A benchmark uses a very small checkpoint",
                "Repeat with a real, validated pretrained checkpoint. Tiny fixtures test benchmark plumbing but are not representative of coding-model speed or quality.",
                "no_purchase",
            ));
        }
        // A GGUF file size is *not* model residency: MoE expert storage, sparse
        // embedding tables and memory mapping require separate accounting.
        let largest_heap = compatible.iter()
            .filter_map(|a| a["largest_usable_device_local_heap_bytes"].as_u64())
            .max();
        if let Some(heap) = largest_heap {
            if runs.iter().any(|r| r["backend"].as_str() == Some("vulkan")
                && r["model_file_bytes"].as_u64().is_some_and(|bytes| bytes > heap)) {
                out.push(advice(
                    "evaluate-vram-capacity", "low",
                    "A tested model's mapped file size exceeds the largest reported GPU heap; this does not measure resident VRAM or prove a bottleneck",
                    "Inspect actual GPU allocation, cache misses, CPU expert execution and transfer time. Consider a larger-VRAM GPU only if measured offloading limits your target workload.",
                    "conditional_only",
                ));
            }
        }
    }

    if host["available_host_ram_bytes"].as_u64().is_some_and(|n| n < 4 * GIB) {
        out.push(advice(
            "check-host-memory-pressure", "medium",
            "Less than 4 GiB host RAM was available to this process during discovery",
            "Check paging and working-set pressure during a real model run. Extra RAM may help an offloaded workload only if host memory is the measured bottleneck.",
            "conditional_only",
        ));
    }

    if !runs.is_empty() {
        out.push(advice(
            "verify-quality-before-tuning", "medium", "Benchmarks passed repeated greedy-token checks but not reference-model quality tests",
            "Compare exact prompt tokens and reference outputs on CPU and hardware Vulkan before accepting changes to cache placement, quantisation or kernels.",
            "no_purchase",
        ));
    }
    out
}

fn build(hardware: Value, host: Value, reports: &[Value]) -> Value {
    let runs: Vec<_> = reports.iter().map(summary).collect();
    let recommendations = recommendations(&hardware, &host, &runs);
    json!({
        "schema": "nextora.insight.v1",
        "generated_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs()),
        "project": "Nextora Insight",
        "engine_version": crate::ENGINE_VERSION,
        "host": host,
        "hardware": hardware,
        "benchmarks": runs,
        "recommendations": recommendations,
        "privacy": {
            "collection": "local only",
            "network_upload": false,
            "benchmark_model_paths_exported": false,
            "usernames_and_serial_numbers_collected": false,
        },
        "limits": [
            "Discovery is not a successful shader, model load or compute test",
            "Available RAM is a snapshot, not evidence of sustained memory pressure",
            "Reported Vulkan heap size and allocation ceiling are not live free VRAM",
            "Model file/mapped size is not residency; MoE and lazy tensor paths change peak working set",
            "No thermal, power, disk I/O, PCIe or cache-miss profiling in Insight v1",
            "No prices or comparative GPU performance database; no percentage upgrade claims",
            "Benchmarks lack checkpoint content digests and driver identity, so Insight v1 does not compare reports as controlled A/B tests",
            "No model tuning, hardware purchase or settings modification is performed",
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_hardware(eligible: bool) -> Value {
        json!({"adapters": [{
            "name": "Test device", "compatible": eligible, "software": false,
            "largest_usable_device_local_heap_bytes": 16 * GIB
        }], "vulkan_available": true})
    }
    fn mock_host(wsl: bool, ram: u64) -> Value {
        json!({"wsl": wsl, "available_host_ram_bytes": ram})
    }
    fn mock_benchmark() -> Value {
        json!({
            "engine": crate::ENGINE_ID,
            "engine_version": crate::ENGINE_VERSION,
            "identical_greedy_checksums": true,
            "repetitions": 2, "warmup_iterations_excluded": 1,
            "concurrent_requests": 1, "prompt_tokens_per_request": 128,
            "decode_steps_per_request": 16, "model_file_bytes": 20 * GIB,
            "median_aggregate_prefill_tokens_per_second": 115.0,
            "median_aggregate_decode_steps_per_second": 12.0,
            "p95_request_decode_completion_seconds": 5.0,
            "samples": [{}, {}],
            "runtime": {"execution": {"backend": "vulkan", "adapter": {"name": "Test device", "software": false}}}
        })
    }
    fn has(report: &Value, id: &str) -> bool {
        report["recommendations"].as_array().unwrap().iter().any(|r| r["id"] == id)
    }
    #[test]
    fn diagnostic_path_is_free_and_does_not_claim_hardware_speed() {
        let report = build(mock_hardware(false), mock_host(true, 8 * GIB), &[]);
        assert!(has(&report, "restore-hardware-path"));
        assert!(has(&report, "record-baseline"));
        assert!(report["recommendations"].as_array().unwrap().iter().all(|r| r["estimated_speedup"].is_null()));
    }
    #[test]
    fn eligible_gpu_without_measurements_never_claims_it_was_benchmarked() {
        let report = build(mock_hardware(true), mock_host(false, 8 * GIB), &[]);
        assert!(has(&report, "benchmark-vulkan-first"));
        assert!(!has(&report, "evaluate-vram-capacity"));
    }
    #[test]
    fn capacity_advice_is_conditional_and_model_paths_are_not_exported() {
        let benchmark = mock_benchmark();
        validate_benchmark(&benchmark).unwrap();
        let report = build(mock_hardware(true), mock_host(false, 8 * GIB), &[benchmark]);
        assert!(has(&report, "evaluate-vram-capacity"));
        assert!(has(&report, "measure-concurrency"));
        assert_eq!(report["recommendations"].as_array().unwrap().iter()
            .find(|x| x["id"] == "evaluate-vram-capacity").unwrap()["purchase"], "conditional_only");
        assert!(report["benchmarks"][0].get("model").is_none());
    }
    #[test]
    fn invalid_or_nonmatching_benchmarks_are_rejected() {
        let mut benchmark = mock_benchmark();
        benchmark["identical_greedy_checksums"] = json!(false);
        assert!(validate_benchmark(&benchmark).is_err());
        benchmark = mock_benchmark();
        benchmark["engine"] = json!("untrusted-other-engine");
        assert!(validate_benchmark(&benchmark).is_err());
        benchmark = mock_benchmark();
        benchmark["samples"] = json!([]);
        assert!(validate_benchmark(&benchmark).is_err());
        benchmark = mock_benchmark();
        benchmark["median_aggregate_prefill_tokens_per_second"] = json!(-2.0);
        assert!(validate_benchmark(&benchmark).is_err());
    }
}
