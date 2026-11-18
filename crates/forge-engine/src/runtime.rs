//! Typed selection of independently implemented execution paths.
use crate::{
    gguf::Gguf,
    model::{Config, Model, Sequence, Step},
    tokenizer::Tokenizer,
    vulkan::{DeviceOptions, GpuMatvecExecutor, GpuModel, GpuSequence},
};
use anyhow::{Result, bail, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use std::{path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Vulkan,
    Hybrid,
}

#[derive(Args, Clone, Debug)]
pub struct BackendArgs {
    /// Explicit CPU execution (equivalent to --backend cpu).
    #[arg(long, conflicts_with = "backend")]
    pub cpu: bool,
    /// cpu: host; vulkan: resident GPU graph; hybrid: CPU state with cached Vulkan projections.
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
            self.backend.ok_or_else(||anyhow::anyhow!("Select --backend cpu, vulkan or hybrid (or --cpu). There is no implicit backend fallback."))?
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
    host_plan: crate::memory::HostMemoryPlan,
    warmup: serde_json::Value,
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
        let backend = options.selected()?;
        let mut g = Gguf::open(path)?;
        let routed_device_cache = backend != Backend::Vulkan
            || g.meta.get("general.architecture").and_then(|v| v.as_str()) == Some("qwen4exp");
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
        let source = Arc::new(g);
        let tokenizer = Tokenizer::from_gguf(&source)?;
        let execution = match backend {
            Backend::Cpu => Execution::Cpu(Model::load(source.clone(), context)?),
            Backend::Vulkan => Execution::Vulkan(GpuModel::load(
                source.clone(),
                context,
                options.device_options()?,
            )?),
            Backend::Hybrid => {
                let executor = Arc::new(GpuMatvecExecutor::new(options.device_options()?)?);
                let model = Model::load_with_executor(source.clone(), context, executor.clone())?;
                executor.pin_dense(&source)?;
                Execution::Hybrid { model, executor }
            }
        };
        let warmup = if options.expert_warmup && routed_device_cache {
            serde_json::to_value(source.warm_expert_cache()?)?
        } else {
            serde_json::json!({"enabled":false})
        };
        let result = Self {
            execution,
            source,
            tokenizer,
            backend,
            host_plan,
            warmup,
        };
        ensure!(
            result.tokenizer.vocab_size() == result.config().vocab_size,
            "tokenizer vocabulary and model disagree"
        );
        Ok(result)
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
            Execution::Cpu(_) => serde_json::json!({"backend":"cpu","fallback":"none"}),
            Execution::Vulkan(m) => {
                serde_json::json!({"backend":"vulkan","fallback":"no_implicit_backend_or_external_runtime_fallback", "expert_budget_policy":"Qwen3.8 retains all selected experts; device-budget CPU expert executions are counted in memory statistics", "adapter":m.adapter_info(),"memory":m.memory_stats()})
            }
            Execution::Hybrid { executor, .. } => {
                serde_json::json!({"backend":"vulkan_hybrid","execution":"CPU model state and explicitly selected Vulkan/CPU projection policy","statistics":executor.stats()})
            }
        };
        serde_json::json!({"engine":crate::ENGINE_ID,"version":crate::ENGINE_VERSION,"config":self.config(),
            "host_memory_plan":self.host_plan,"expert_warmup":self.warmup,
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
