//! Independently implemented Forge model runtime.
//!
//! This crate has no RustingStrata runtime dependency. CPU, Vulkan,
//! and diagnostic hybrid backends share a checked GGUF loader; the explicit
//! auto policy chooses compatible Vulkan hardware or reports CPU selection. Numerical gates
//! and measured target-hardware performance are separate acceptance criteria.

pub mod cli;
// Preserve the existing bounded, tracked-only repository context packer.
#[path = "../../../src/code_context.rs"]
pub mod context;
pub mod gguf;
pub mod insight;
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

pub const ENGINE_ID: &str = "nextora-native";
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
        "backend_policies": ["auto", "cpu", "vulkan", "hybrid"],
        "requires_explicit_backend": true,
        "unloaded_service_default_backend": "auto",
        "gpu_vendor_policy": "capabilities_not_vendor_or_model",
        "supported_host_os": ["windows", "linux"],
        "native_accelerator_apis": ["vulkan"],
        "auto_backend": {"preference": "compatible_vulkan_then_cpu", "resolution": "before_host_memory_planning", "explicit_vulkan_options_disable_cpu_fallback": true, "model_or_execution_errors_trigger_fallback": false},
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
        "fallback": "auto_selection_only_no_runtime_or_external_fallback",
        "qwen_vulkan_budget_policy": "retain_every_selected_expert; explicit_counted_CPU_expert_execution_if_device_working_set_cannot_fit",
        "hardware_performance_claim": "none"
    })
}
