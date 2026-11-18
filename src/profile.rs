use anyhow::{Result, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use std::{ffi::OsString, path::PathBuf};
use rusting_strata::serving_policy::{CapacityPlan, ModelFeatures, Parallelism, resolve_capacity};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kv { Q8, F16 }

impl Kv {
    pub fn as_str(self) -> &'static str { match self { Self::Q8 => "q8", Self::F16 => "f16" } }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset { #[default] General, Code, CodeLong }

impl Preset {
    pub fn name(self) -> &'static str { match self { Self::General => "general", Self::Code => "code", Self::CodeLong => "code-long" } }
    pub fn context(self) -> usize { match self { Self::General => 4096, Self::Code => 16384, Self::CodeLong => 32768 } }
    pub fn output(self) -> usize { match self { Self::General => 1024, Self::Code => 4096, Self::CodeLong => 8192 } }
    pub fn cache_ram(self) -> usize { if self == Self::General { 512 } else { 1024 } }
    pub fn ram_margin(self) -> usize { if self == Self::General { 4096 } else { 6144 } }
}

#[derive(Args, Debug)]
pub struct EngineOptions {
    /// Chosen starting settings; explicit flags win. Code presets target Qwen3.8 development workloads.
    #[arg(long, value_enum, default_value_t = Preset::General)] pub preset: Preset,
    /// GGUF model path. Serve may start empty; generation/chat/bench require it.
    #[arg(long)] pub model: Option<PathBuf>,
    /// Explicit CPU reference path. GPU errors never silently switch to CPU.
    #[arg(long, conflicts_with = "device")] pub cpu: bool,
    /// Vulkan backend index from doctor; defaults to the unique RX 9070 XT.
    #[arg(long)] pub device: Option<usize>,
    /// Context tokens. Larger contexts leave less VRAM for expert weights.
    #[arg(long, default_value_t = 4096,
        default_value_if("preset", "code", "16384"), default_value_if("preset", "code-long", "32768"))] pub ctx: usize,
    /// Prompt tokens per chunk [default: min(256, context)]. 0 uses token-at-a-time prefill.
    #[arg(long)] pub chunk: Option<usize>,
    /// MiB to leave outside the engine's reported VRAM allocation budget.
    #[arg(long, default_value_t = 2048)] pub vram_margin: usize,
    /// MiB to leave available for the OS and other applications when budgeting experts.
    #[arg(long, default_value_t = 4096,
        default_value_if("preset", "code", "6144"), default_value_if("preset", "code-long", "6144"))] pub ram_margin: usize,
    /// Optional explicit cap for the engine's estimated RAM in MiB (0 = automatic).
    #[arg(long, default_value_t = 0)] pub max_ram: usize,
    /// Owned expert RAM ceiling: auto uses available memory; 0 disables; a number is MiB.
    #[arg(long, default_value = "auto")] pub expert_ram: String,
    /// Populate the bounded expert working set at load time; false fills it on demand.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)] pub expert_warmup: bool,
    /// Host RAM MiB for prompt-state caching; separate from the expert-weight cache.
    #[arg(long, default_value_t = 512,
        default_value_if("preset", "code", "1024"), default_value_if("preset", "code-long", "1024"))] pub cache_ram: usize,
    /// Optional persistent GPU conversation-state cache. In-memory prefix reuse already works without this.
    #[arg(long)] pub prompt_cache: Option<PathBuf>,
    /// Preserve earlier reasoning in the template, allowing unchanged prefixes to be reused.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)] pub preserve_thinking: bool,
    /// Let the model reason before answering; API chat_template_kwargs can override this per request.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)] pub think: bool,
    /// Attention KV cache representation. F16 is the fidelity baseline; Q8 is an explicit memory/precision tradeoff.
    #[arg(long, value_enum, default_value = "f16")] pub kv: Kv,
    /// CPU worker count; default uses the engine's detected physical cores on GPU.
    #[arg(long)] pub threads: Option<usize>,
    /// Server capacity: auto selects compatible slots; explicit 1..4 is strict. Non-server commands use one slot.
    #[arg(long, default_value_t = Parallelism::Auto)] pub parallel: Parallelism,
    /// Ceiling for auto server capacity (1..4). Per-slot context/state is reserved at model load.
    #[arg(long, default_value_t = 4)] pub max_parallel: usize,
    /// MTP draft tokens per step (0..=2); requires a compatible head, GPU, and one sequence.
    #[arg(long, default_value_t = 0)] pub mtp: usize,
    /// Matching standalone MTP head in GGUF format; requires --mtp 1 or --mtp 2.
    #[arg(long)] pub mtp_file: Option<PathBuf>,
    /// Legacy option: only 0 is accepted; the fidelity profile retains all selected experts.
    #[arg(long, default_value_t = 0, hide = true)] pub expert_tail_min_rank: usize,
}

impl EngineOptions {
    pub fn chunk(&self) -> usize { self.chunk.unwrap_or(256.min(self.ctx)) }

    /// Conservative capacity before a model is known, also valid for later model
    /// switches. Architecture-specific restrictions can only lower auto capacity.
    pub fn maximum_capacity(&self) -> Result<CapacityPlan> {
        resolve_capacity(self.parallel, self.max_parallel, !self.cpu, self.mtp, ModelFeatures::default())
    }

    pub fn validate(&self) -> Result<()> {
        rusting_strata::memory::parse_expert_ram(&self.expert_ram)?;
        ensure!((128..=131_072).contains(&self.ctx), "--ctx must be between 128 and 131072 tokens");
        ensure!(self.chunk() <= self.ctx, "--chunk cannot exceed --ctx");
        ensure!(self.threads != Some(0), "--threads must be greater than zero");
        ensure!(!(self.cpu && self.device.is_some()), "--cpu and --device are mutually exclusive");
        ensure!(self.mtp <= 2, "--mtp must be 0, 1 or 2 draft tokens per step");
        ensure!(self.expert_tail_min_rank == 0,
            "expert omission is disabled by the fidelity profile; --expert-tail-min-rank must be 0");
        self.maximum_capacity()?;
        if let Some(path) = &self.mtp_file {
            ensure!(self.mtp > 0, "--mtp-file requires --mtp 1 or --mtp 2");
            ensure!(path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
                "--mtp-file must be an already-converted GGUF companion head");
            ensure!(path.is_file(), "--mtp-file is not a file: {}", path.display());
        }
        for (name, n) in [("--vram-margin", self.vram_margin), ("--ram-margin", self.ram_margin),
                          ("--max-ram", self.max_ram), ("--cache-ram", self.cache_ram)] {
            ensure!(n <= 1_048_576, "{name} exceeds the 1 TiB input limit (units are MiB)");
        }
        if !self.cpu { ensure!(self.vram_margin < 16_384,
            "--vram-margin must leave some of the RX 9070 XT's nominal 16 GiB for the model"); }
        Ok(())
    }

    pub fn validate_command(&self, command: &str) -> Result<()> {
        self.validate()?;
        ensure!(command == "serve" || matches!(self.parallel, Parallelism::Auto | Parallelism::Fixed(1)),
            "--parallel greater than 1 is only available for serve");
        Ok(())
    }

    pub fn arguments(&self, command: &str, device: Option<usize>) -> Result<Vec<OsString>> {
        self.validate_command(command)?;
        ensure!(self.cpu == device.is_none(), "internal backend selection mismatch");
        let mut args = vec!["forge9070".into(), command.into()];
        if let Some(model) = &self.model { super::arg(&mut args, "--model", model.as_os_str()); }
        else { ensure!(command == "serve", "--model is required for {command}"); }
        for (flag, value) in [("--ctx", self.ctx), ("--chunk", self.chunk()),
                              ("--vram-margin", self.vram_margin), ("--ram-margin", self.ram_margin),
                              ("--max-ram", self.max_ram), ("--cache-ram", self.cache_ram), ("--mtp", self.mtp)] {
            super::arg(&mut args, flag, value.to_string());
        }
        if command == "serve" {
            super::arg(&mut args, "--parallel", self.parallel.to_string());
            super::arg(&mut args, "--max-parallel", self.max_parallel.to_string());
        }
        super::arg(&mut args, "--expert-ram", &self.expert_ram);
        super::arg(&mut args, "--expert-warmup", self.expert_warmup.to_string());
        super::arg(&mut args, "--preserve-thinking", self.preserve_thinking.to_string());
        super::arg(&mut args, "--think", self.think.to_string());
        if let Some(path) = &self.prompt_cache { super::arg(&mut args, "--prompt-cache", path.as_os_str()); }
        if let Some(path) = &self.mtp_file { super::arg(&mut args, "--mtp-model", path.as_os_str()); }
        super::arg(&mut args, "--kv", self.kv.as_str());
        if let Some(threads) = self.threads { super::arg(&mut args, "--threads", threads.to_string()); }
        if let Some(device) = device {
            args.push("--gpu".into());
            super::arg(&mut args, "--backend", "vulkan");
            super::arg(&mut args, "--device", device.to_string());
        }
        Ok(args)
    }
}

pub fn default_profile() -> serde_json::Value {
    serde_json::json!({
        "name": "RX 9070 XT generation speed with all selected experts retained",
        "status": "chosen defaults; not benchmark-calibrated on your hardware",
        "gpu": "AMD Radeon RX 9070 XT",
        "architecture": "RDNA 4 / gfx1201",
        "nominal_vram_mib": 16384,
        "backend": "vulkan",
        "preferred_environment": "native Windows (or native Linux with working AMD Vulkan)",
        "rocm_required": false,
        "context_tokens": 4096,
        "prefill_chunk_tokens": 256,
        "vram_margin_mib": 2048,
        "ram_margin_mib": 4096,
        "expert_ram": "auto",
        "expert_warmup": true,
        "expert_ram_policy": "bounded owned expert LRU sized from available RAM; lease-safe eviction; prefill misses do not replace the decode working set",
        "expert_ram_runtime_reserve_mib": 1024,
        "prompt_cache_ram_mib": 512,
        "kv_format": "f16",
        "kv_tradeoff": "F16 avoids additional KV quantization; explicit Q8 reduces state memory but may change outputs",
        "mtp_drafts": 0,
        "mtp_supported_drafts": [0, 1, 2],
        "mtp_requirement": "matching embedded or standalone GGUF head; GPU execution; parallel=1",
        "parallel_mode": "auto",
        "parallel_requests": null,
        "parallel_requests_ceiling": 4,
        "parallel_requests_supported": [1, 2, 3, 4],
        "parallel_auto_policy": "CPU, MTP and HC/PLE/QSA models use one slot; compatible GPU models use the configured ceiling; this is compatibility detection, not VRAM fit or performance tuning",
        "parallel_strategy": "shared model weights; separate sequence state; occupied slots follow demand up to capacity chosen at model load",
        "parallel_allocation": "context/state buffers are reserved for the chosen capacity at model load; idle capacity does not release VRAM",
        "parallel_tradeoff": "more active contexts reduce VRAM available for experts; aggregate throughput and per-request latency must be measured",
        "queue_capacity": rusting_strata::serving_policy::DEFAULT_QUEUE_CAPACITY,
        "queue_timeout_seconds": rusting_strata::serving_policy::DEFAULT_QUEUE_TIMEOUT_SECS,
        "maximum_connections": rusting_strata::serving_policy::DEFAULT_MAX_WAITING,
        "expert_tail_min_rank": 0,
        "expert_tail_status": "disabled by this profile; every selected expert contributes",
        "tuning_objective": "median single-request decode speed among candidates matching tested greedy baseline outputs",
        "accuracy_scope": "preserve the selected model's computation; quantization and factual task accuracy still require evaluation",
        "cpu_fallback": "explicit --cpu only",
        "expert_cache": "hot GPU experts above an owned host RAM tier; cold weights read from storage; sparse PLE rows remain lazy",
        "limits": "RAM settings are startup ceilings, not operating-system reservations; owned storage is pageable; throughput needs a hardware run"
    })
}

pub fn selected_profile(preset: Preset) -> serde_json::Value {
    let mut p = default_profile();
    p["preset"] = preset.name().into();
    p["context_tokens"] = preset.context().into();
    p["maximum_output_tokens"] = preset.output().into();
    p["prompt_cache_ram_mib"] = preset.cache_ram().into();
    p["ram_margin_mib"] = preset.ram_margin().into();
    p["preserve_thinking"] = true.into();
    p["thinking"] = true.into();
    p["persistent_prompt_cache"] = "opt-in --prompt-cache; model-specific metadata identity, not a full weight checksum".into();
    p["request_budget"] = "Rendered prompt plus requested output must fit context; explicit oversize requests are rejected without truncating source".into();
    p["context_limit_tokens"] = 131072.into();
    p["available_presets"] = serde_json::json!(["general", "code", "code-long"]);
    if preset != Preset::General {
        p["name"] = "Qwen3.8 large-code starting profile for RX 9070 XT".into();
        p["sampling"] = serde_json::json!({"temperature":1.0,"top_p":0.95,"top_k":20,"source":"Qwen3.8 thinking-mode model-card guidance; explicit request values override"});
        p["parallel_requests_supported"] = serde_json::json!([1]);
        p["parallel_requests"] = 1.into();
        p["mtp_supported_drafts"] = serde_json::json!([0]);
        p["parallel_strategy"] = "Qwen3.8 qwen4exp: one generation sequence; additional clients queue".into();
        p["mtp_requirement"] = "Qwen3.8 qwen4exp MTP is not supported in this build".into();
        p["repository_context"] = "Use the context command for bounded query-focused excerpts; this is lexical retrieval, not a full semantic code index".into();
        p["development_memory"] = "Start the IDE and normal development services before loading; the 6 GiB reserve allows some later growth but is not a reservation".into();
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> EngineOptions {
        EngineOptions { preset: Preset::General, model: Some("a model.gguf".into()), cpu: false, device: None, ctx: 4096,
            chunk: None, vram_margin: 2048, ram_margin: 4096, max_ram: 0, expert_ram: "auto".into(), expert_warmup: true, cache_ram: 512,
            prompt_cache: None, preserve_thinking: true, think: true,
            kv: Kv::F16, threads: None, parallel: Parallelism::Auto, max_parallel: 4, mtp: 0, mtp_file: None, expert_tail_min_rank: 0 }
    }

    #[test]
    fn gpu_plan_pins_vulkan_and_keeps_paths_whole() {
        let args = options().arguments("generate", Some(2)).unwrap();
        assert!(args.windows(2).any(|p| p == ["--backend", "vulkan"]));
        assert!(args.windows(2).any(|p| p == ["--device", "2"]));
        assert!(args.windows(2).any(|p| p == ["--model", "a model.gguf"]));
        assert!(!args.iter().any(|s| s == "--cpu-fallback"));
        assert!(args.windows(2).any(|p| p == ["--expert-ram", "auto"]));
        assert!(args.windows(2).any(|p| p == ["--expert-warmup", "true"]));
    }

    #[test]
    fn expert_memory_controls_are_explicit_and_checked_before_loading() {
        let mut o = options();
        o.expert_ram = "40960".into();
        o.expert_warmup = false;
        let args = o.arguments("generate", Some(0)).unwrap();
        assert!(args.windows(2).any(|p| p == ["--expert-ram", "40960"]));
        assert!(args.windows(2).any(|p| p == ["--expert-warmup", "false"]));
        for invalid in ["-1", "64GB", "1048577"] {
            o.expert_ram = invalid.into();
            assert!(o.validate().is_err());
        }
    }

    #[test]
    fn cpu_plan_never_initializes_a_gpu_backend() {
        let mut options = options();
        options.cpu = true;
        let args = options.arguments("generate", None).unwrap();
        assert!(!args.iter().any(|s| s == "--gpu" || s == "--backend" || s == "--device"));
        assert!(options.arguments("generate", Some(0)).is_err());
    }

    #[test]
    fn small_context_clamps_default_chunk_and_rejects_oversize_override() {
        let mut options = options();
        options.ctx = 128;
        assert_eq!(options.chunk(), 128);
        assert!(options.validate().is_ok());
        options.chunk = Some(256);
        assert!(options.validate().is_err());
    }

    #[test]
    fn only_server_can_start_without_a_model() {
        let mut options = options();
        options.model = None;
        options.cpu = true;
        assert!(options.arguments("serve", None).is_ok());
        assert!(options.arguments("generate", None).is_err());
    }

    #[test]
    fn concurrent_server_sequences_are_explicit_and_forwarded_once() {
        let mut options = options();
        options.parallel = Parallelism::Fixed(4);
        let args = options.arguments("serve", Some(0)).unwrap();
        assert!(args.windows(2).any(|p| p == ["--parallel", "4"]));
        assert_eq!(args.iter().filter(|arg| *arg == "--parallel").count(), 1);
        assert!(options.arguments("generate", Some(0)).is_err());
        assert!(options.arguments("bench", Some(0)).is_err());
    }

    #[test]
    fn auto_capacity_is_server_only_and_the_ceiling_does_not_override_fixed_slots() {
        let mut options = options();
        options.max_parallel = 2;
        let args = options.arguments("serve", Some(0)).unwrap();
        assert!(args.windows(2).any(|p| p == ["--parallel", "auto"]));
        assert!(args.windows(2).any(|p| p == ["--max-parallel", "2"]));
        assert_eq!(options.maximum_capacity().unwrap().slots, 2);
        for command in ["generate", "chat", "bench"] {
            let args = options.arguments(command, Some(0)).unwrap();
            assert!(!args.iter().any(|s| s == "--parallel" || s == "--max-parallel"));
        }
        options.parallel = Parallelism::Fixed(4);
        assert_eq!(options.maximum_capacity().unwrap().slots, 4);
        options.parallel = Parallelism::Auto;
        options.cpu = true;
        assert_eq!(options.maximum_capacity().unwrap().slots, 1);
        assert!(options.arguments("serve", None).is_ok());
        options.max_parallel = 0;
        assert!(options.validate().is_err());
    }

    #[test]
    fn code_profiles_do_not_claim_parallel_qwen_or_elastic_gpu_memory() {
        for preset in [Preset::Code, Preset::CodeLong] {
            let p = selected_profile(preset);
            assert_eq!(p["parallel_mode"], "auto");
            assert_eq!(p["parallel_requests"], 1);
            assert_eq!(p["parallel_requests_supported"], serde_json::json!([1]));
            assert!(p["parallel_allocation"].as_str().unwrap().contains("does not release VRAM"));
        }
        assert!(default_profile()["parallel_requests"].is_null());
    }

    #[test]
    fn unsupported_concurrency_is_rejected_without_cpu_downgrade() {
        let mut options = options();
        for parallel in [0, 5, 255] {
            options.parallel = Parallelism::Fixed(parallel);
            assert!(options.validate().is_err());
        }
        options.parallel = Parallelism::Fixed(2);
        options.cpu = true;
        let error = options.validate().unwrap_err().to_string();
        assert!(error.contains("generates their replies sequentially"));
    }

    #[test]
    fn mtp_and_concurrency_conflicts_fail_before_loading() {
        let mut options = options();
        options.mtp = 2;
        options.parallel = Parallelism::Fixed(2);
        assert!(options.validate().unwrap_err().to_string().contains("cannot run together"));
        options.parallel = Parallelism::Fixed(1);
        options.cpu = true;
        assert!(options.validate().unwrap_err().to_string().contains("requires GPU"));
        options.cpu = false;
        options.mtp = 3;
        assert!(options.validate().is_err());
    }

    #[test]
    fn explicit_mtp_and_companion_path_are_preserved() {
        let mut options = options();
        // Profile validation checks the path/flags; the model preflight validates
        // the head tensors before actual execution.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.gguf");
        options.mtp_file = Some(path.clone());
        assert!(options.validate().unwrap_err().to_string().contains("requires --mtp"));
        options.mtp = 2;
        let args = options.arguments("generate", Some(1)).unwrap();
        assert!(args.windows(2).any(|p| p == ["--mtp", "2"]));
        assert!(args.windows(2).any(|p| p[0] == "--mtp-model" && p[1] == path.as_os_str()));
        assert!(!args.iter().any(|arg| arg == "--parallel"));
        options.mtp_file = Some(PathBuf::from("missing.safetensors"));
        assert!(options.validate().unwrap_err().to_string().contains("already-converted GGUF"));
    }

    #[test]
    fn fidelity_profile_rejects_expert_omission_in_every_execution_mode() {
        let mut options = options();
        assert_eq!(options.expert_tail_min_rank, 0);
        assert!(options.validate().is_ok());
        let args = options.arguments("serve", Some(0)).unwrap();
        assert!(!args.iter().any(|arg| arg == "--expert-tail-min-rank"));
        for cpu in [false, true] {
            options.cpu = cpu;
            for rank in [1, 8, 256, 257] {
                options.expert_tail_min_rank = rank;
                assert!(options.validate().unwrap_err().to_string().contains("expert omission is disabled"));
            }
        }
    }
}

