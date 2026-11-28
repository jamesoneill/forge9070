//! Independently implemented Forge model runtime.
//!
//! This crate has no RustingStrata runtime dependency. CPU, Vulkan,
//! and diagnostic hybrid backends share a checked GGUF loader; the explicit
//! auto policy chooses compatible Vulkan hardware or reports CPU selection. Numerical gates
//! and measured target-hardware performance are separate acceptance criteria.

pub(crate) mod autotune;
pub mod cli;
// Preserve the existing bounded, tracked-only repository context packer.
#[path = "../../../src/code_context.rs"]
pub mod context;
pub mod gguf;
pub mod insight;
pub mod inkling_plan;
pub mod math;
pub mod model_fit;
pub mod offline_quant;
pub mod memory;
pub mod model;
pub mod performance;
pub mod protocol;
pub mod quant;
pub mod quality;
pub mod qwen38;
pub mod qwen35;
pub(crate) mod prism;
pub mod rope;
pub mod runtime;
pub mod server;
pub mod tokenizer;
pub mod vulkan;

pub const ENGINE_ID: &str = "nextora-native";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn capabilities() -> serde_json::Value {
    let direct_experts = serde_json::json!({
            "automatic":false,"automatic_uses":"packed","explicit_opt_in":true,
            "nonexpert_pipeline":"ordinary_single_buffer_with_verified_f32_specializations",
            "fast_f32_attention":{"query_heads":32,"kv_heads":4,"head_width":128,
                "original_history":"all_initialized_tokens","selection":"verified_driver_and_active_shape",
                "active_kernels":"see_runtime_adapter_and_dispatch_counters"},
            "fast_f32_router":{"input_width":2048,"output_rows":128,"zero_offsets":true,
                "hardware":"Windows RX 9070 XT, Vulkan driver 8389024, API 1.4.359"},
            "fast_q4_q6_expert_pipeline":"batched_array_64_lanes_packed_word_reads",
            "fast_decode_requires":["direct_experts","fast_math"],
            "specialized_shader_creation_fallback":"existing_direct_and_ordinary_pipelines",
            "fast_q6_expert_pipeline":"batched_array_explicit_scale_order_and_fma",
            "production_unsafe_array_override":false,
            "numerical_validation":{"model":"Qwen3-30B-A3B-abliterated.Q4_K_M","gpu":"AMD Radeon RX 9070 XT",
                "fast_math":true,"context_tokens":40960,"prompt_tokens":21,"forced_decode_steps":128,
                "compared_logits":19447808,"bit_exact_to_packed":true,"scope":"specified model, hardware and trajectory only"},
            "other_models_and_hardware":"experimental; numerical equivalence not verified"
        });
    serde_json::json!({
        "engine": ENGINE_ID,
        "version": ENGINE_VERSION,
        "independent_runtime": true,
        "rusting_strata_dependency": false,
        "stage": "native_gpu_and_coding_api_target_hardware_acceptance_pending",
        "experimental_prism_build": cfg!(feature = "experimental-prism"),
        "experimental_prism_vulkan_build": cfg!(feature = "prism-vulkan"),
        "architectures": ["qwen3", "qwen3moe", "qwen35", "qwen4exp"],
        "architecture_backends": {"qwen3":["cpu","vulkan","hybrid"],"qwen3moe":["cpu","vulkan","hybrid"],"qwen35":["cpu","hybrid"],"qwen4exp":["cpu","vulkan","hybrid"]},
        "weight_formats": ["F32", "F16", "BF16", "Q4_0", "Q8_0", "Q2_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS"],
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
        "model_discovery": "bounded_read_only_complete_gguf_scan",
        "http_architecture_policy": "validated_native_architectures",
        "model_setup": {"separate_native_worker":true,"phases":["verify","fit_memory","benchmark","save_profile","explicit_use","warmup"],"default_goal":"maximum_memory_fit","full_window_speed_verified":false},
        "experimental_direct_experts": direct_experts,
        "context_cache": {"gpu_state_placements":["gpu","ram"],"gpu_state_precision":["f32","q4_64"],"ram_placement_architectures":["qwen3","qwen3moe"],"compact_architectures":["qwen3","qwen3moe"],"retains_all_initialized_tokens":true},
        "concurrency": {"active_maximum": 4, "default_active": 1, "bounded_fifo_queue": true, "continuous_batching": false},
        "moe_loading": {"owned_bounded_host_cache": true, "direct_io_with_reported_buffered_fallback": true, "bounded_gpu_expert_cache_for_qwen4exp": true, "lazy_ple_rows": true, "prefill_protects_decode_cache": true, "selected_experts_omitted": 0, "transfer_compute_overlap": false},
        "gpu_performance_verified": false,
        "state": "immutable_mapped_model_and_independent_sequences",
        "fallback": "auto_selection_only_no_runtime_or_external_fallback",
        "qwen_vulkan_budget_policy": "retain_every_selected_expert; explicit_counted_CPU_expert_execution_if_device_working_set_cannot_fit",
        "hardware_performance_claim": "none"
    })
}
