//! Independently implemented Forge model runtime.
//!
//! This crate has no RustingStrata runtime dependency. Explicit CPU, Vulkan,
//! and diagnostic hybrid backends share a checked GGUF loader. Numerical gates
//! and measured target-hardware performance are separate acceptance criteria.

pub mod cli;
pub mod gguf;
pub mod math;
pub mod memory;
pub mod model;
pub mod performance;
pub mod protocol;
pub mod quant;
pub mod qwen38;
pub mod runtime;
pub mod server;
pub mod tokenizer;
pub mod vulkan;

pub const ENGINE_ID: &str = "forge-native";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn capabilities() -> serde_json::Value {
    serde_json::json!({
        "engine": ENGINE_ID,
        "version": ENGINE_VERSION,
        "independent_runtime": true,
        "rusting_strata_dependency": false,
        "stage": "native_gpu_and_coding_api_target_hardware_acceptance_pending",
        "architectures": ["qwen3", "qwen3moe", "qwen4exp"],
        "weight_formats": ["F32", "F16", "BF16", "Q4_0", "Q8_0", "Q2_0", "Q4_K", "Q5_K", "Q6_K", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS"],
        "backends": ["cpu", "vulkan", "hybrid"],
        "requires_explicit_backend": true,
        "vulkan": true,
        "qwen3_8": true,
        "gguf_shards": true,
        "tool_calling_api": true,
        "http_server": true,
        "configuration_ui": true,
        "startup_without_model": true,
        "settings_persistence": "explicit_json_file",
        "model_discovery": "bounded_read_only_complete_gguf_moe_scan",
        "http_architecture_policy": "validated_moe_only",
        "concurrency": {"active_maximum": 4, "default_active": 1, "bounded_fifo_queue": true, "continuous_batching": false},
        "moe_loading": {"owned_bounded_host_cache": true, "direct_io_with_reported_buffered_fallback": true, "bounded_gpu_expert_cache_for_qwen4exp": true, "lazy_ple_rows": true, "prefill_protects_decode_cache": true, "selected_experts_omitted": 0, "transfer_compute_overlap": false},
        "gpu_performance_verified": false,
        "state": "immutable_mapped_model_and_independent_sequences",
        "fallback": "no_implicit_backend_or_external_runtime_fallback",
        "qwen_vulkan_budget_policy": "retain_every_selected_expert; explicit_counted_CPU_expert_execution_if_device_working_set_cannot_fit",
        "hardware_performance_claim": "none"
    })
}
