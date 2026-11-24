//! Typed selection of independently implemented execution paths.
use crate::{
    gguf::Gguf,
    model::{Config, Model, Sequence, Step},
    tokenizer::Tokenizer,
    vulkan::{KvPlacement, KvPrecision, DeviceOptions, GpuMatvecExecutor, GpuModel, GpuSequence},
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use std::{path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Selection policy, resolved to an execution backend before memory planning.
    Auto,
    Cpu,
    Vulkan,
    Hybrid,
}

#[derive(Args, Clone, Debug)]
pub struct BackendArgs {
    /// Explicit CPU execution (equivalent to --backend cpu).
    #[arg(long, conflicts_with = "backend")]
    pub cpu: bool,
    /// auto: compatible Vulkan or reported CPU selection; cpu: host; vulkan: GPU; hybrid: diagnostic.
    #[arg(long, value_enum)]
    pub backend: Option<Backend>,
    /// Vulkan physical-device index reported by engine doctor.
    #[arg(long)]
    pub device: Option<usize>,
    /// Permit a software Vulkan adapter for numerical testing only.
    #[arg(long)]
    pub allow_software_vulkan: bool,
    /// Quantize Vulkan activations to Q8 for packed integer dot products.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub fast_math: bool,
    /// Direct cached expert buffers: omitted selects policy, false keeps packed copies.
    #[arg(long, action = clap::ArgAction::Set)]
    pub direct_experts: Option<bool>,
    /// Context-cache storage; q4_64 retains all tokens with quantized K/V values.
    #[arg(long, value_enum, default_value = "f32")]
    pub kv_precision: KvPrecision,
    /// Keep attention history in GPU memory or system RAM read by Vulkan.
    #[arg(long, value_enum, default_value = "gpu")]
    pub kv_placement: KvPlacement,
    /// Optional context RAM ceiling in MiB; automatic reserves full history plus working buffers.
    #[arg(long)]
    pub kv_ram_mib: Option<usize>,
    /// Explicit cap for native Vulkan allocations; zero chooses a conservative device budget.
    #[arg(long, default_value_t = 0)]
    pub vram_mib: usize,
    /// Owned CPU expert-cache ceiling in MiB; omitted means automatic after OS/state reserves, zero disables.
    #[arg(long)]
    pub expert_ram_mib: Option<usize>,
    #[arg(long, default_value_t = 4096)]
    pub ram_margin_mib: usize,
    /// Fill the bounded owned expert working set before generation; never reads the entire PLE table.
    #[arg(long,default_value_t=true,action=clap::ArgAction::Set)]
    pub expert_warmup: bool,
    #[arg(skip = 1usize)]
    pub state_slots: usize,
}

impl BackendArgs {
    pub fn selected(&self) -> Result<Backend> {
        let backend = if self.cpu {
            Backend::Cpu
        } else {
            self.backend.ok_or_else(||anyhow::anyhow!("Select --backend auto, cpu, vulkan or hybrid (or --cpu). CPU fallback during hardware selection requires auto; explicit backends never silently change."))?
        };
        ensure!(
            backend != Backend::Cpu
                || (!self.allow_software_vulkan && self.device.is_none() && self.vram_mib == 0),
            "Vulkan options cannot be used with the CPU backend"
        );
        ensure!(
            !self.fast_math || matches!(backend, Backend::Auto | Backend::Vulkan),
            "fast math requires the Vulkan backend"
        );
        ensure!(self.direct_experts != Some(true) || matches!(backend, Backend::Auto | Backend::Vulkan), "direct expert access requires the Vulkan backend");
        ensure!(!self.kv_precision.is_compact() || matches!(backend, Backend::Auto | Backend::Vulkan), "compact context memory requires the Vulkan backend");
        ensure!(self.kv_placement.is_gpu() || matches!(backend, Backend::Auto | Backend::Vulkan), "RAM context placement requires the Vulkan backend");
        ensure!(self.kv_placement.is_gpu() || self.state_slots == 1, "RAM context placement currently supports one request state slot");
        ensure!(self.kv_ram_mib.is_none() || self.kv_placement == KvPlacement::Ram, "a context RAM cap requires RAM context placement");
        ensure!(!self.kv_precision.is_compact() || self.state_slots == 1, "compact context memory currently supports one request state slot");
        Ok(backend)
    }
    pub fn device_options(&self) -> Result<DeviceOptions> {
        Ok(DeviceOptions {
            fast_math: self.fast_math,
            direct_experts: self.direct_experts,
            kv_precision: self.kv_precision,
            kv_placement: self.kv_placement,
            host_memory_budget_bytes: self.kv_ram_mib.map(|mib| mib.checked_mul(1024*1024).context("context RAM budget overflow")).transpose()?.unwrap_or(0),
            device_index: self.device,
            allow_software: self.allow_software_vulkan,
            memory_budget_bytes: self
                .vram_mib
                .checked_mul(1024 * 1024)
                .ok_or_else(|| anyhow::anyhow!("VRAM budget overflow"))?,
        })
    }

    fn resolve(&self) -> Result<ResolvedBackend> {
        self.resolve_with_probe(crate::vulkan::select_device)
    }

    /// Resolve the policy before any backend-dependent host memory reservation.
    /// Only adapter discovery can choose CPU. Device creation, shader compilation,
    /// model validation, allocation and execution errors are never retried on CPU.
    fn resolve_with_probe(
        &self,
        probe: impl FnOnce(&DeviceOptions) -> Result<Option<usize>>,
    ) -> Result<ResolvedBackend> {
        let requested = self.selected()?;
        let mut device = self.device_options()?;
        let (selected, reason) = if requested != Backend::Auto {
            (requested, "Explicit backend selection".to_owned())
        } else {
            // An adapter index, allocation cap or software-test permission is
            // deliberate GPU configuration. Never silently discard that intent.
            let cpu_allowed =
                self.device.is_none() && self.vram_mib == 0 && !self.allow_software_vulkan && !self.fast_math && self.direct_experts != Some(true) && !self.kv_precision.is_compact() && self.kv_placement.is_gpu() && self.kv_ram_mib.is_none();
            match probe(&device) {
                Ok(Some(index)) => {
                    device.device_index = Some(index);
                    (
                        Backend::Vulkan,
                        format!("Compatible Vulkan adapter {index} selected by capabilities"),
                    )
                }
                Ok(None) if cpu_allowed => (
                    Backend::Cpu,
                    "No compatible hardware Vulkan adapter is available; auto selected CPU".into(),
                ),
                Ok(None) => bail!(
                    "auto found no compatible Vulkan adapter; explicit --device, --vram-mib, --allow-software-vulkan --fast-math or --kv-precision settings prevent CPU fallback"
                ),
                Err(error) if cpu_allowed => (
                    Backend::Cpu,
                    format!("Vulkan discovery unavailable; auto selected CPU: {error:#}"),
                ),
                Err(error) => {
                    return Err(error).context(
                        "auto cannot discard explicit Vulkan settings; CPU fallback is disabled",
                    );
                }
            }
        };
        Ok(ResolvedBackend {
            selection: BackendSelection {
                requested,
                selected,
                automatic: requested == Backend::Auto,
                reason,
            },
            device,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct BackendSelection {
    requested: Backend,
    selected: Backend,
    automatic: bool,
    reason: String,
}

struct ResolvedBackend {
    selection: BackendSelection,
    device: DeviceOptions,
}

/// Hardware discovery remains useful on CPU-only hosts without a Vulkan loader.
/// Eligibility describes API limits, not successful model execution or speed.
pub fn hardware_report() -> serde_json::Value {
    use serde_json::json;
    let mut report = match crate::vulkan::discover() {
        Ok(mut report) => {
            report["vulkan_available"] = json!(true);
            report["vulkan_discovery_error"] = serde_json::Value::Null;
            report
        }
        Err(error) => json!({
            "engine": "forge-native-vulkan",
            "adapters": [],
            "vulkan_available": false,
            "vulkan_discovery_error": format!("{error:#}"),
            "software_allowed_by_default": false,
            "compute_submitted": false,
            "hardware_performance_verified": false
        }),
    };
    report["platform"] = json!(std::env::consts::OS);
    report["cpu_quant_kernels"] = json!({
        "Q4_K":crate::quant::dot_kernel(crate::quant::QuantType::Q4_K,2048),
        "Q6_K":crate::quant::dot_kernel(crate::quant::QuantType::Q6_K,2048),
        "selection":"runtime CPU/OS feature checks; scalar and AVX2 fallbacks retained"
    });
    report["cpu_available"] = json!(cfg!(any(target_os = "linux", target_os = "windows")));
    report["backend_policies"] = json!(["auto", "cpu", "vulkan", "hybrid"]);
    report
}

enum Execution {
    Cpu(Model),
    Vulkan(GpuModel),
    Hybrid {
        model: Model,
        executor: Arc<GpuMatvecExecutor>,
    },
}

pub struct RuntimeModel {
    execution: Execution,
    pub source: Arc<Gguf>,
    pub tokenizer: Tokenizer,
    backend: Backend,
    selection: BackendSelection,
    host_plan: crate::memory::HostMemoryPlan,
    warmup: serde_json::Value,
    load_timings_ms: serde_json::Value,
}

pub enum RuntimeSequence {
    Cpu(Sequence),
    Vulkan(GpuSequence),
}

impl RuntimeSequence {
    pub fn position(&self) -> usize {
        match self {
            Self::Cpu(s) => s.position(),
            Self::Vulkan(s) => s.position(),
        }
    }
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Self::Cpu(s) => {
                s.reset();
                Ok(())
            }
            Self::Vulkan(s) => s.reset(),
        }
    }
}

pub(crate) enum LoadStep {
    Stage(&'static str, &'static str),
    Upload {
        message: &'static str,
        unit: &'static str,
        completed: usize,
        total: usize,
        bytes: u64,
    },
    Warmup {
        considered: usize,
        total: usize,
        ready: usize,
        bytes: usize,
    },
}

impl RuntimeModel {
    /// Geometry/RAM planning and physical budget discovery only; no Vulkan
    /// logical device, GPU allocation or inference is performed.
    pub fn fit_plan(path: &Path, context: usize, options: &BackendArgs) -> Result<serde_json::Value> {
        let source = Arc::new(Gguf::open(path)?);
        let checked = Model::load(Arc::clone(&source), context)?;
        ensure!(options.direct_experts != Some(true) || matches!(checked.config().architecture.as_str(), "qwen3" | "qwen3moe"), "direct expert access currently supports ordinary Qwen3 and Qwen3 MoE");
        let resolved = options.resolve()?;
        let backend = resolved.selection.selected;
        ensure!(options.kv_placement.is_gpu() || matches!(checked.config().architecture.as_str(),"qwen3"|"qwen3moe"),
            "RAM context placement currently supports ordinary Qwen3 and Qwen3 MoE");
        let host_plan = crate::memory::plan_with_kv(&source, context, options.state_slots,
            backend != Backend::Vulkan, options.expert_ram_mib, options.ram_margin_mib,
            options.kv_precision, options.kv_placement, options.kv_ram_mib)?;
        let (adapter, execution_plan) = if backend == Backend::Vulkan {
            (crate::vulkan::inspect_device(&resolved.device)?,
                GpuModel::fit_memory_report(checked.config(), &source, &resolved.device)?)
        } else if backend == Backend::Hybrid {
            (crate::vulkan::inspect_device(&resolved.device)?, serde_json::json!({
                "execution":"cpu_state_with_bounded_gpu_projection_cache",
                "full_context_gpu_state":false,"host_context_reserve_bytes":host_plan.state_reserve_bytes,
                "all_dense_weights_pinned":checked.config().architecture != "qwen35",
            }))
        } else { (serde_json::Value::Null, serde_json::Value::Null) };
        Ok(serde_json::json!({"engine":crate::ENGINE_ID,"backend_selection":resolved.selection,
            "config":checked.config(),"host_memory_plan":host_plan,"adapter":adapter,
            "execution_plan":execution_plan,"gpu_compute_submitted":false}))
    }
    /// Allocates/touches a disposable full-context state reserve. This is an
    /// allocation proof, distinct from processing a full prompt or model quality.
    pub fn probe_capacity(&self) -> Result<serde_json::Value> {
        match &self.execution {
            Execution::Vulkan(model) => model.probe_capacity(),
            Execution::Cpu(_) | Execution::Hybrid { .. } => {
                let bytes = usize::try_from(self.host_plan.state_reserve_bytes).context("host capacity probe exceeds address range")?;
                let mut allocation = Vec::<u8>::new();
                allocation.try_reserve_exact(bytes).context("allocate full host context reserve")?;
                allocation.resize(bytes, 0);
                // Volatile nonzero writes commit every page; reading calloc's
                // untouched zero pages would not establish physical RAM capacity.
                let mut checksum = 0u64;
                for offset in (0..bytes).step_by(4096) {
                    unsafe {
                        let byte = allocation.as_mut_ptr().add(offset);
                        std::ptr::write_volatile(byte, 1);
                        checksum += u64::from(std::ptr::read_volatile(byte));
                    }
                }
                let report = serde_json::json!({"capacity_proof":"host_geometry_reserve_allocated_and_touched",
                    "context_tokens":self.context(),"host_reserve_bytes":bytes,"touch_checksum":checksum,
                    "history_tokens_processed":0,"long_context_quality_verified":false});
                drop(allocation);
                Ok(report)
            }
        }
    }
    pub fn load(path: &Path, context: usize, options: &BackendArgs) -> Result<Self> {
        Self::load_with_progress(path, context, options, |_| {})
    }

    pub(crate) fn load_with_progress(
        path: &Path,
        context: usize,
        options: &BackendArgs,
        progress: impl Fn(LoadStep),
    ) -> Result<Self> {
        // Reject invalid GGUF and unsupported tokenizer metadata before probing
        // drivers or allocating any Vulkan resources. This also prevents an
        // unnecessary hardware probe on CPU-only hosts with an invalid model.
        let load_start = std::time::Instant::now();
        progress(LoadStep::Stage(
            "mapping",
            "Reading GGUF headers and mapping model shards",
        ));
        let mut g = Gguf::open(path)?;
        let mapping_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        ensure!(options.direct_experts != Some(true) || g.meta.get("general.architecture").and_then(|v| v.as_str()).is_some_and(|arch| matches!(arch, "qwen3" | "qwen3moe")), "direct expert access currently supports ordinary Qwen3 and Qwen3 MoE");
        ensure!(
            !options.fast_math || g.meta.get("general.architecture").and_then(|v| v.as_str()).is_some_and(|arch| matches!(arch, "qwen3" | "qwen3moe")),
            "fast math currently supports ordinary Qwen3 and Qwen3 MoE"
        );
        ensure!(options.kv_placement.is_gpu() || g.meta.get("general.architecture").and_then(|v|v.as_str()).is_some_and(|arch| matches!(arch,"qwen3"|"qwen3moe")), "RAM context placement currently supports ordinary Qwen3 and Qwen3 MoE");
        ensure!(!options.kv_precision.is_compact() || g.meta.get("general.architecture").and_then(|v| v.as_str()).is_some_and(|arch| matches!(arch, "qwen3" | "qwen3moe")), "compact context memory currently supports ordinary Qwen3 and Qwen3 MoE");
        let tokenizer_start = std::time::Instant::now();
        progress(LoadStep::Stage("tokenizer", "Building the tokenizer"));
        let tokenizer = Tokenizer::from_gguf(&g)?;
        let tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;
        let backend_start = std::time::Instant::now();
        progress(LoadStep::Stage(
            "detecting",
            "Selecting the execution backend",
        ));
        let mut resolved = options.resolve()?;
        let backend_selection_ms = backend_start.elapsed().as_secs_f64() * 1000.0;
        let backend = resolved.selection.selected;
        if resolved.selection.automatic {
            eprintln!("Nextora backend: {backend:?}. {}", resolved.selection.reason);
        }
        let routed_device_cache = backend != Backend::Vulkan
            || g.meta
                .get("general.architecture")
                .and_then(|v| v.as_str())
                .is_some_and(|arch| matches!(arch, "qwen3moe" | "qwen4exp"));
        let memory_plan_start = std::time::Instant::now();
        progress(LoadStep::Stage(
            "planning",
            "Planning RAM, expert storage and sequence state",
        ));
        let host_plan = crate::memory::plan_with_kv(
            &g,
            context,
            options.state_slots,
            backend != Backend::Vulkan,
            if routed_device_cache {
                options.expert_ram_mib
            } else {
                Some(0)
            },
            options.ram_margin_mib,
            options.kv_precision,
            options.kv_placement,
            options.kv_ram_mib,
        )?;
        if backend == Backend::Vulkan {
            resolved.device.host_memory_budget_bytes = usize::try_from(host_plan.vulkan_host_budget_bytes).context("Vulkan RAM budget exceeds address range")?;
        }
        g.set_expert_cache_capacity(if routed_device_cache {
            host_plan.expert_cache_capacity_bytes
        } else {
            0
        })?;
        let memory_plan_ms = memory_plan_start.elapsed().as_secs_f64() * 1000.0;
        let source = Arc::new(g);
        let runtime_start = std::time::Instant::now();
        progress(LoadStep::Stage(
            "initializing",
            match backend {
                Backend::Vulkan => "Initializing Vulkan and uploading GPU weights",
                Backend::Hybrid => "Initializing the CPU model and GPU projection cache",
                _ => "Initializing the CPU model",
            },
        ));
        let uploaded = |message, unit, completed, total, bytes| {
            progress(LoadStep::Upload {
                message,
                unit,
                completed,
                total,
                bytes,
            });
        };
        let execution = match backend {
            Backend::Cpu => Execution::Cpu(Model::load(source.clone(), context)?),
            Backend::Vulkan => Execution::Vulkan(GpuModel::load_with_progress(
                source.clone(),
                context,
                resolved.device,
                &uploaded,
            )?),
            Backend::Hybrid => {
                let executor = Arc::new(GpuMatvecExecutor::new(resolved.device)?);
                let model = Model::load_with_executor(source.clone(), context, executor.clone())?;
                // Dense Qwen3.5 checkpoints can exceed VRAM; keep their matrix
                // cache bounded instead of pinning every tensor (including MTP).
                if source.meta.get("general.architecture").and_then(|v| v.as_str()) != Some("qwen35") {
                    executor.pin_dense_with_progress(&source, &uploaded)?;
                }
                Execution::Hybrid { model, executor }
            }
            Backend::Auto => {
                bail!("automatic backend policy was not resolved before model loading")
            }
        };
        let config_vocab_size = match &execution {
            Execution::Cpu(model) | Execution::Hybrid { model, .. } => model.config().vocab_size,
            Execution::Vulkan(model) => model.config().vocab_size,
        };
        ensure!(
            tokenizer.vocab_size() == config_vocab_size,
            "tokenizer vocabulary and model disagree"
        );
        let runtime_init_ms = runtime_start.elapsed().as_secs_f64() * 1000.0;
        // Warmup is potentially expensive: only perform it for an accepted
        // tokenizer/model pair. Expert warmup never changes model precision.
        let warmup_start = std::time::Instant::now();
        let warmup = if options.expert_warmup && routed_device_cache {
            progress(LoadStep::Stage("warming", "Preparing the expert cache"));
            serde_json::to_value(source.warm_expert_cache_with_progress(|total, report| {
                progress(LoadStep::Warmup {
                    considered: report.experts_considered,
                    total,
                    ready: report.experts_ready,
                    bytes: report.payload_bytes_loaded,
                });
            })?)?
        } else {
            serde_json::json!({"enabled":false})
        };
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
        let load_timings_ms = serde_json::json!({
            "gguf_mapping": mapping_ms,
            "tokenizer_construction": tokenizer_ms,
            "backend_selection": backend_selection_ms,
            "memory_planning": memory_plan_ms,
            "runtime_initialization": runtime_init_ms,
            "expert_warmup": warmup_ms,
            "total": load_start.elapsed().as_secs_f64() * 1000.0,
            "unit": "milliseconds",
        });
        Ok(Self {
            execution,
            source,
            tokenizer,
            backend,
            selection: resolved.selection,
            host_plan,
            warmup,
            load_timings_ms,
        })
    }
    pub fn config(&self) -> &Config {
        match &self.execution {
            Execution::Cpu(m) | Execution::Hybrid { model: m, .. } => m.config(),
            Execution::Vulkan(m) => m.config(),
        }
    }
    pub fn context(&self) -> usize {
        self.config().context_length
    }
    pub fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    pub fn backend(&self) -> Backend {
        self.backend
    }
    pub fn new_sequence(&self) -> Result<RuntimeSequence> {
        Ok(match &self.execution {
            Execution::Cpu(m) | Execution::Hybrid { model: m, .. } => {
                RuntimeSequence::Cpu(m.new_sequence())
            }
            Execution::Vulkan(m) => RuntimeSequence::Vulkan(m.new_sequence()?),
        })
    }
    pub fn step_into(
        &self,
        state: &mut RuntimeSequence,
        token: u32,
        output: &mut Step,
    ) -> Result<()> {
        match (&self.execution, state) {
            (Execution::Cpu(m) | Execution::Hybrid { model: m, .. }, RuntimeSequence::Cpu(s)) => {
                m.step_into(s, token, output)
            }
            (Execution::Vulkan(m), RuntimeSequence::Vulkan(s)) => m.step_into(s, token, output),
            _ => bail!("sequence belongs to another execution backend"),
        }
    }
    pub fn prefill(&self, state: &mut RuntimeSequence, tokens: &[u32]) -> Result<Step> {
        match (&self.execution, state) {
            (Execution::Cpu(m) | Execution::Hybrid { model: m, .. }, RuntimeSequence::Cpu(s)) => {
                m.prefill(s, tokens)
            }
            (Execution::Vulkan(m), RuntimeSequence::Vulkan(s)) => m.prefill(s, tokens),
            _ => bail!("sequence belongs to another execution backend"),
        }
    }
    pub fn future_ram_reserve_bytes(&self) -> u64 {
        match &self.execution {
            Execution::Vulkan(model) => {
                let stats = model.memory_stats();
                if stats["kv_placement"] == "ram" {
                    self.host_plan.vulkan_host_budget_bytes.saturating_sub(stats["host_live_allocation_bytes"].as_u64().unwrap_or(0))
                } else { 0 }
            },
            _ => 0,
        }
    }
    pub fn report(&self) -> serde_json::Value {
        let execution = match &self.execution {
            Execution::Cpu(_) => {
                serde_json::json!({"backend":"cpu","fallback":if self.selection.automatic {"auto_selection_cpu"} else {"none"}})
            }
            Execution::Vulkan(m) => {
                serde_json::json!({"backend":"vulkan","fallback":"no_implicit_backend_or_external_runtime_fallback", "expert_budget_policy":"MoE execution retains all selected experts; bounded GPU caches and CPU expert executions are counted in memory statistics", "adapter":m.adapter_info(),"memory":m.memory_stats()})
            }
            Execution::Hybrid { executor, .. } => {
                serde_json::json!({"backend":"vulkan_hybrid","execution":"CPU model state and explicitly selected Vulkan/CPU projection policy","statistics":executor.stats()})
            }
        };
        serde_json::json!({"engine":crate::ENGINE_ID,"version":crate::ENGINE_VERSION,"config":self.config(),
            "backend_selection":self.selection,
            "host_memory_plan":self.host_plan,"expert_warmup":self.warmup,
            "future_ram_reserve_bytes":self.future_ram_reserve_bytes(),
            "model_load_timings_ms":self.load_timings_ms,
            "execution":execution,"mapped_weight_bytes":self.source.mapped_bytes(),"shards":self.source.shard_paths(),
            "expert_cache":self.source.expert_cache_stats().map(|s|serde_json::to_value(s).unwrap_or_default()).unwrap_or_else(|e|serde_json::json!({"error":e.to_string()})),"kv_precision":execution["memory"]["kv_precision"].as_str().unwrap_or("F32")})
    }
}

pub fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
        "empty or non-finite logits"
    );
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(u32::try_from(best)?)
}

#[cfg(test)]
mod backend_selection_tests {
    use super::*;

    #[test]
    fn direct_expert_cli_preserves_automatic_and_explicit_modes() {
        use clap::Parser;
        #[derive(Parser)] struct Options { #[command(flatten)] backend: BackendArgs }
        let automatic=Options::parse_from(["nextora","--backend","auto"]).backend;
        assert_eq!(automatic.direct_experts,None);
        for (text,value) in [("true",true),("false",false)] {
            let requested=Options::parse_from(["nextora","--backend","vulkan","--direct-experts",text]).backend;
            assert_eq!(requested.direct_experts,Some(value));
            assert_eq!(requested.device_options().unwrap().direct_experts,Some(value));
        }
        let mut forced=args(Backend::Auto); forced.direct_experts=Some(true);
        assert!(forced.resolve_with_probe(|_|Ok(None)).is_err());
        forced.backend=Some(Backend::Cpu);assert!(forced.selected().is_err());
        forced.direct_experts=Some(false);assert!(forced.selected().is_ok());
    }

    fn args(backend: Backend) -> BackendArgs {
        BackendArgs {
            cpu: false,
            backend: Some(backend),
            device: None,
            allow_software_vulkan: false,
            fast_math: false,
            direct_experts: None,
            kv_precision: KvPrecision::F32,
            kv_placement: KvPlacement::Gpu,
            kv_ram_mib: None,
            vram_mib: 0,
            expert_ram_mib: Some(0),
            ram_margin_mib: 0,
            expert_warmup: false,
            state_slots: 1,
        }
    }

    #[test]
    fn auto_uses_compatible_adapter_and_pins_its_index() {
        let result = args(Backend::Auto)
            .resolve_with_probe(|options| {
                assert!(!options.allow_software);
                assert_eq!(options.device_index, None);
                Ok(Some(2))
            })
            .unwrap();
        assert_eq!(result.selection.requested, Backend::Auto);
        assert_eq!(result.selection.selected, Backend::Vulkan);
        assert_eq!(result.device.device_index, Some(2));
        assert!(result.selection.automatic);
    }

    #[test]
    fn auto_selects_cpu_when_no_compatible_adapter_exists() {
        let result = args(Backend::Auto)
            .resolve_with_probe(|_| Ok(None))
            .unwrap();
        assert_eq!(result.selection.selected, Backend::Cpu);
        assert!(result.selection.reason.contains("No compatible hardware"));
        assert!(result.selection.automatic);
    }

    #[test]
    fn auto_reports_loader_unavailability() {
        let result = args(Backend::Auto)
            .resolve_with_probe(|_| bail!("Vulkan loader is missing"))
            .unwrap();
        assert_eq!(result.selection.selected, Backend::Cpu);
        assert!(result.selection.reason.contains("Vulkan loader is missing"));
    }

    #[test]
    fn auto_never_discards_explicit_gpu_settings() {
        let mut explicit_device = args(Backend::Auto);
        explicit_device.device = Some(9);
        let mut explicit_budget = args(Backend::Auto);
        explicit_budget.vram_mib = 512;
        let mut explicit_software = args(Backend::Auto);
        explicit_software.allow_software_vulkan = true;
        let mut explicit_fast = args(Backend::Auto);
        explicit_fast.fast_math = true;
        let mut explicit_compact = args(Backend::Auto);
        explicit_compact.kv_precision = KvPrecision::Q4_64;
        for options in [explicit_device, explicit_budget, explicit_software, explicit_fast, explicit_compact] {
            assert!(options.resolve_with_probe(|_| Ok(None)).is_err());
            assert!(
                options
                    .resolve_with_probe(|_| bail!("invalid/unavailable GPU"))
                    .is_err()
            );
        }
    }

    #[test]
    fn explicit_backends_never_probe_for_an_automatic_fallback() {
        for backend in [Backend::Cpu, Backend::Vulkan, Backend::Hybrid] {
            let result = args(backend)
                .resolve_with_probe(|_| panic!("explicit backend must not run automatic selection"))
                .unwrap();
            assert_eq!(result.selection.requested, backend);
            assert_eq!(result.selection.selected, backend);
            assert!(!result.selection.automatic);
        }
    }

    #[test]
    fn cpu_gpu_option_conflicts_and_missing_policy_still_fail() {
        let mut cpu = args(Backend::Cpu);
        cpu.device = Some(0);
        assert!(cpu.selected().is_err());
        for backend in [Backend::Cpu, Backend::Hybrid] {
            let mut fast = args(backend);
            fast.fast_math = true;
            assert!(fast.selected().is_err());
            fast.fast_math = false;
            fast.kv_precision = KvPrecision::Q4_64;
            assert!(fast.selected().is_err());
        }
        let mut omitted = args(Backend::Auto);
        omitted.backend = None;
        assert!(omitted.selected().is_err());
        omitted.cpu = true;
        assert_eq!(omitted.selected().unwrap(), Backend::Cpu);
    }

    #[test]
    fn overflowing_memory_option_fails_before_probe() {
        let mut options = args(Backend::Auto);
        options.vram_mib = usize::MAX;
        let error = options
            .resolve_with_probe(|_| panic!("invalid budget must not reach discovery"))
            .err()
            .unwrap();
        assert!(error.to_string().contains("VRAM budget overflow"));
    }
}
