//! Read-only live Insight metrics for the native service.
use super::Server;
use serde_json::{Value, json};
use std::{collections::VecDeque, sync::Mutex, time::{Duration, Instant}};

const MAX_SAMPLES: usize = 64;

#[derive(Clone, Debug)]
struct Sample {
    prefill_uncached_tokens: usize,
    cached_tokens: usize,
    generated_tokens: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    elapsed_seconds: f64,
}

struct Totals {
    started: u64,
    completed: u64,
    interrupted_or_failed: u64,
    recent: VecDeque<Sample>,
}
impl Default for Totals {
    fn default() -> Self {
        Self { started: 0, completed: 0, interrupted_or_failed: 0, recent: VecDeque::new() }
    }
}

pub(super) struct LiveTelemetry {
    since: Instant,
    records: Mutex<Totals>,
}
impl LiveTelemetry {
    pub fn new() -> Self {
        Self { since: Instant::now(), records: Mutex::new(Totals::default()) }
    }

    pub fn begin(&self, prompt_tokens: usize, cached_tokens: usize) -> RequestTrace<'_> {
        self.records.lock().unwrap().started += 1;
        RequestTrace {
            owner: self,
            began: Instant::now(),
            prompt_tokens,
            cached_tokens: cached_tokens.min(prompt_tokens),
            prefill_seconds: 0.0,
            successful: false,
        }
    }

    pub fn snapshot(&self) -> Value {
        let state = self.records.lock().unwrap();
        let recent = state.recent.iter().map(|s| {
            json!({
                "uncached_prompt_tokens": s.prefill_uncached_tokens,
                "cached_prompt_tokens": s.cached_tokens,
                "generated_tokens": s.generated_tokens,
                "prefill_seconds": s.prefill_seconds,
                "decode_seconds": s.decode_seconds,
                "elapsed_seconds": s.elapsed_seconds,
                "prefill_tokens_per_second": rate(s.prefill_uncached_tokens, s.prefill_seconds),
                "generation_tokens_per_second": rate(s.generated_tokens, s.decode_seconds),
            })
        }).collect::<Vec<_>>();
        let recent_prefill_tokens: usize = state.recent.iter().map(|s| s.prefill_uncached_tokens).sum();
        let recent_output_tokens: usize = state.recent.iter().map(|s| s.generated_tokens).sum();
        let recent_cached_tokens: usize = state.recent.iter().map(|s| s.cached_tokens).sum();
        let recent_prefill_seconds: f64 = state.recent.iter().map(|s| s.prefill_seconds).sum();
        let recent_decode_seconds: f64 = state.recent.iter().map(|s| s.decode_seconds).sum();
        json!({
            "source": "successful_native_service_generations",
            "since_seconds": self.since.elapsed().as_secs_f64(),
            "started_requests": state.started,
            "completed_requests": state.completed,
            "interrupted_or_failed_requests": state.interrupted_or_failed,
            "recent_window": state.recent.len(),
            "recent_generated_tokens": recent_output_tokens,
            "recent_cached_prompt_tokens": recent_cached_tokens,
            "recent_uncached_prompt_tokens": recent_prefill_tokens,
            "recent_aggregate_prefill_tokens_per_second": rate(recent_prefill_tokens, recent_prefill_seconds),
            "recent_aggregate_generation_tokens_per_second": rate(recent_output_tokens, recent_decode_seconds),
            "recent_samples": recent,
            "measurement_notes": [
                "Recent sample rates are aggregate phase throughput from up to 64 completed service requests, not GPU utilization or guaranteed token rates",
                "Decode timing includes token sampling, incremental decoding and streaming; it is not a GPU-only kernel benchmark",
                "Requests that fail or disconnect before successful output parsing are counted separately, without false speed samples",
                "A new service process starts with empty measurements; data is not uploaded or persisted"
            ]
        })
    }
}

pub(super) struct RequestTrace<'a> {
    owner: &'a LiveTelemetry,
    began: Instant,
    prompt_tokens: usize,
    cached_tokens: usize,
    prefill_seconds: f64,
    successful: bool,
}
impl RequestTrace<'_> {
    pub fn prefill_finished(&mut self, duration: Duration) {
        self.prefill_seconds = duration.as_secs_f64();
    }

    pub fn complete(&mut self, generated_tokens: usize, decode_seconds: f64) {
        if self.successful { return; }
        let sample = Sample {
            prefill_uncached_tokens: self.prompt_tokens.saturating_sub(self.cached_tokens),
            cached_tokens: self.cached_tokens,
            generated_tokens,
            prefill_seconds: self.prefill_seconds,
            decode_seconds,
            elapsed_seconds: self.began.elapsed().as_secs_f64(),
        };
        let mut s = self.owner.records.lock().unwrap();
        s.completed += 1;
        if s.recent.len() == MAX_SAMPLES { s.recent.pop_front(); }
        s.recent.push_back(sample);
        self.successful = true;
    }
}
impl Drop for RequestTrace<'_> {
    fn drop(&mut self) {
        if !self.successful {
            self.owner.records.lock().unwrap().interrupted_or_failed += 1;
        }
    }
}

fn rate(count: usize, seconds: f64) -> Option<f64> {
    (seconds > 0.0 && seconds.is_finite()).then(|| count as f64 / seconds)
}

/// Called only after API authorization. Does not reveal model paths or prompt contents.
pub(super) fn live(server: &Server) -> Value {
    let telemetry = server.insight.snapshot();
    let admission = server.admission.snapshot();
    let host = crate::insight::host_snapshot();
    let model = server.loaded.read().unwrap();
    let runtime = model.as_ref().map(|m| m.model.report());
    let gpu = runtime.as_ref().map(|r| &r["execution"]["memory"]);
    let cache = runtime.as_ref().map(|r| &r["expert_cache"]);
    let memory = json!({
        "mapped_model_file_bytes": runtime.as_ref().and_then(|r| r["mapped_weight_bytes"].as_u64()),
        "gpu_allocation_budget_bytes": gpu.and_then(|m|m["allocation_budget_bytes"].as_u64()),
        "gpu_live_owned_allocation_bytes": gpu.and_then(|m|m["live_allocation_bytes"].as_u64()),
        "gpu_high_water_owned_allocation_bytes": gpu.and_then(|m|m["high_water_allocation_bytes"].as_u64()),
        "gpu_cpu_expert_fallbacks": gpu.and_then(|m|m["cpu_expert_fallbacks"].as_u64()),
        "gpu_uploaded_bytes_cumulative": gpu.and_then(|m|m["uploaded_bytes"].as_u64()),
        "gpu_readback_bytes_cumulative": gpu.and_then(|m|m["readback_bytes"].as_u64()),
        "host_expert_cache_resident_bytes": cache.and_then(|m|m["resident_bytes"].as_u64()),
        "host_expert_cache_capacity_bytes": cache.and_then(|m|m["capacity_bytes"].as_u64()),
        "host_expert_cache_hits_cumulative": cache.and_then(|m|m["hits"].as_u64()),
        "host_expert_cache_misses_cumulative": cache.and_then(|m|m["misses"].as_u64()),
        "host_expert_cache_evictions_cumulative": cache.and_then(|m|m["evictions"].as_u64()),
        "host_expert_cache_bypasses_cumulative": cache.and_then(|m|m["mapped_bypasses"].as_u64()),
        "host_expert_cache_prefill_hits_cumulative": cache.and_then(|m|m["prefill_hits"].as_u64()),
        "host_expert_cache_prefill_misses_cumulative": cache.and_then(|m|m["prefill_misses"].as_u64()),
        "host_expert_payload_read_bytes_cumulative": cache.and_then(|m|m["payload_bytes_read"].as_u64()),
        "host_expert_direct_loads_cumulative": cache.and_then(|m|m["direct_expert_loads"].as_u64()),
        "host_expert_buffered_loads_cumulative": cache.and_then(|m|m["buffered_expert_loads"].as_u64()),
        "host_expert_read_errors_cumulative": cache.and_then(|m|m["read_errors"].as_u64()),
        "expert_cache_counter_scope": "cumulative counters since this model was loaded; no per-request timing or GPU transfer attribution",
        "gpu_allocation_scope": "Nextora-owned allocations and budget, not total device VRAM, free VRAM, or utilization",
        "model_file_scope": "mapped file bytes; not resident VRAM or RAM",
    });
    let advice = crate::insight::live_recommendations(&host, &admission, &memory, &telemetry,
        runtime.as_ref().and_then(|r|r["execution"]["backend"].as_str()));
    json!({
        "schema":"nextora.insight.live.v1",
        "host":host,
        "model_loaded":model.is_some(),
        "backend":runtime.as_ref().and_then(|r|r["execution"]["backend"].as_str()),
        "admission":admission,
        "memory":memory,
        "measurements":telemetry,
        "recommendations":advice,
        "privacy":"local authenticated endpoint; no telemetry upload, model path, prompt, user identity or hardware serial numbers",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_generation_populates_bounded_observation_window() {
        let stats = LiveTelemetry::new();
        {
            let mut x=stats.begin(120, 30);
            x.prefill_finished(Duration::from_millis(100));
            x.complete(20, 2.0);
        }
        let snapshot = stats.snapshot();
        assert_eq!(snapshot["completed_requests"], 1);
        assert_eq!(snapshot["interrupted_or_failed_requests"], 0);
        assert_eq!(snapshot["recent_samples"][0]["uncached_prompt_tokens"], 90);
        assert_eq!(snapshot["recent_samples"][0]["generation_tokens_per_second"], 10.0);
        assert!(snapshot.get("model_path").is_none());
    }

    #[test]
    fn interruption_is_not_a_successful_benchmark() {
        let stats = LiveTelemetry::new();
        { let _trace=stats.begin(2,0); }
        let snapshot = stats.snapshot();
        assert_eq!(snapshot["completed_requests"], 0);
        assert_eq!(snapshot["interrupted_or_failed_requests"], 1);
        assert_eq!(snapshot["recent_window"], 0);
    }
}
