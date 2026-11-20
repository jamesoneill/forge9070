//! Typed selection of independently implemented execution paths.
use crate::{
    gguf::Gguf,
    model::{Config, Model, Sequence, Step},
    tokenizer::Tokenizer,
    vulkan::{DeviceOptions, GpuMatvecExecutor, GpuModel, GpuSequence},
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
        Ok(backend)
    }
    pub fn device_options(&self) -> Result<DeviceOptions> {
        Ok(DeviceOptions {
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
                self.device.is_none() && self.vram_mib == 0 && !self.allow_software_vulkan;
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
                    "auto found no compatible Vulkan adapter; explicit --device, --vram-mib or --allow-software-vulkan settings prevent CPU fallback"
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

impl RuntimeModel {
    pub fn load(path: &Path, context: usize, options: &BackendArgs) -> Result<Self> {
        // Reject invalid GGUF and unsupported tokenizer metadata before probing
        // drivers or allocating any Vulkan resources. This also prevents an
        // unnecessary hardware probe on CPU-only hosts with an invalid model.
        let load_start = std::time::Instant::now();
        let mut g = Gguf::open(path)?;
        let mapping_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        let tokenizer_start = std::time::Instant::now();
        let tokenizer = Tokenizer::from_gguf(&g)?;
        let tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;
        let backend_start = std::time::Instant::now();
        let resolved = options.resolve()?;
        let backend_selection_ms = backend_start.elapsed().as_secs_f64() * 1000.0;
        let backend = resolved.selection.selected;
        if resolved.selection.automatic {
            eprintln!("Nextora backend: {backend:?}. {}", resolved.selection.reason);
        }
        let routed_device_cache = backend != Backend::Vulkan
            || g.meta.get("general.architecture").and_then(|v| v.as_str()) == Some("qwen4exp");
        let memory_plan_start = std::time::Instant::now();
        let host_plan = crate::memory::plan(
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
        )?;
        g.set_expert_cache_capacity(if routed_device_cache {
            host_plan.expert_cache_capacity_bytes
        } else {
            0
        })?;
        let memory_plan_ms = memory_plan_start.elapsed().as_secs_f64() * 1000.0;
        let source = Arc::new(g);
        let runtime_start = std::time::Instant::now();
        let execution = match backend {
            Backend::Cpu => Execution::Cpu(Model::load(source.clone(), context)?),
            Backend::Vulkan => {
                Execution::Vulkan(GpuModel::load(source.clone(), context, resolved.device)?)
            }
            Backend::Hybrid => {
                let executor = Arc::new(GpuMatvecExecutor::new(resolved.device)?);
                let model = Model::load_with_executor(source.clone(), context, executor.clone())?;
                executor.pin_dense(&source)?;
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
            serde_json::to_value(source.warm_expert_cache()?)?
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
    pub fn report(&self) -> serde_json::Value {
        let execution = match &self.execution {
            Execution::Cpu(_) => {
                serde_json::json!({"backend":"cpu","fallback":if self.selection.automatic {"auto_selection_cpu"} else {"none"}})
            }
            Execution::Vulkan(m) => {
                serde_json::json!({"backend":"vulkan","fallback":"no_implicit_backend_or_external_runtime_fallback", "expert_budget_policy":"Qwen3.8 retains all selected experts; device-budget CPU expert executions are counted in memory statistics", "adapter":m.adapter_info(),"memory":m.memory_stats()})
            }
            Execution::Hybrid { executor, .. } => {
                serde_json::json!({"backend":"vulkan_hybrid","execution":"CPU model state and explicitly selected Vulkan/CPU projection policy","statistics":executor.stats()})
            }
        };
        serde_json::json!({"engine":crate::ENGINE_ID,"version":crate::ENGINE_VERSION,"config":self.config(),
            "backend_selection":self.selection,
            "host_memory_plan":self.host_plan,"expert_warmup":self.warmup,
            "model_load_timings_ms":self.load_timings_ms,
            "execution":execution,"mapped_weight_bytes":self.source.mapped_bytes(),"shards":self.source.shard_paths(),
            "expert_cache":self.source.expert_cache_stats().map(|s|serde_json::to_value(s).unwrap_or_default()).unwrap_or_else(|e|serde_json::json!({"error":e.to_string()})),"kv_precision":"F32"})
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

    fn args(backend: Backend) -> BackendArgs {
        BackendArgs {
            cpu: false,
            backend: Some(backend),
            device: None,
            allow_software_vulkan: false,
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
        for options in [explicit_device, explicit_budget, explicit_software] {
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
