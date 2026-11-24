//! Architecture dispatch for first-party GPU execution.
use super::{DeviceOptions, qwen38, resident};
use crate::{
    gguf::{Gguf, Value as GgufValue},
    model::{Config, Step},
};
use anyhow::{Result, bail};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
pub enum GpuModel {
    Ordinary(resident::GpuModel),
    Qwen38(qwen38::QwenGpuModel),
}
#[derive(Debug)]
pub enum GpuSequence {
    Ordinary(resident::GpuSequence),
    Qwen38(qwen38::QwenGpuSequence),
}
impl GpuModel {
    pub fn load(source: Arc<Gguf>, context: usize, options: DeviceOptions) -> Result<Self> {
        Self::load_with_progress(source, context, options, &|_, _, _, _, _| {})
    }

    pub(crate) fn load_with_progress(
        source: Arc<Gguf>,
        context: usize,
        options: DeviceOptions,
        progress: super::UploadProgress<'_>,
    ) -> Result<Self> {
        if source.meta.get("general.architecture").and_then(GgufValue::as_str) == Some("qwen35") {
            bail!("Qwen3.5/Qwen3.8 dense currently supports the native CPU or Hybrid backend; the resident Vulkan graph is not implemented");
        }
        if source
            .meta
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            == Some("qwen4exp")
        {
            Ok(Self::Qwen38(qwen38::QwenGpuModel::load_with_progress(
                source, context, options, progress,
            )?))
        } else {
            Ok(Self::Ordinary(resident::GpuModel::load(
                source, context, options, progress,
            )?))
        }
    }
    pub fn probe_capacity(&self) -> Result<Value> {
        match self {
            Self::Ordinary(model) => model.probe_capacity(),
            Self::Qwen38(model) => {
                // This graph eagerly allocates its complete KV/index/recurrent
                // state. A separate touch proof is not implemented here yet.
                let sequence = model.new_sequence()?;
                let report = serde_json::json!({"capacity_proof":"full_state_allocated_eagerly",
                    "context_tokens":model.context(),"state_touched":false,
                    "history_tokens_processed":0,"long_context_quality_verified":false,
                    "memory":model.memory_stats()});
                drop(sequence);
                Ok(report)
            }
        }
    }
    pub(crate) fn fit_memory_report(config: &Config, source: &Gguf, options: &DeviceOptions) -> Result<Value> {
        if matches!(config.architecture.as_str(), "qwen3" | "qwen3moe") {
            resident::GpuModel::fit_memory_report(config, source, options)
        } else {
            Ok(serde_json::json!({"estimate":"architecture_specific; accepted runtime load validates allocations",
                "architecture":config.architecture,"configured_context_tokens":config.context_length}))
        }
    }
    pub fn config(&self) -> &Config {
        match self {
            Self::Ordinary(m) => m.config(),
            Self::Qwen38(m) => m.config(),
        }
    }
    pub fn context(&self) -> usize {
        self.config().context_length
    }
    pub fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    pub fn adapter_info(&self) -> Value {
        match self {
            Self::Ordinary(m) => m.adapter_info(),
            Self::Qwen38(m) => m.adapter_info(),
        }
    }
    pub fn memory_stats(&self) -> Value {
        match self {
            Self::Ordinary(m) => m.memory_stats(),
            Self::Qwen38(m) => m.memory_stats(),
        }
    }
    pub fn new_sequence(&self) -> Result<GpuSequence> {
        match self {
            Self::Ordinary(m) => Ok(GpuSequence::Ordinary(m.new_sequence()?)),
            Self::Qwen38(m) => Ok(GpuSequence::Qwen38(m.new_sequence()?)),
        }
    }
    pub fn step_into(&self, s: &mut GpuSequence, token: u32, out: &mut Step) -> Result<()> {
        match (self, s) {
            (Self::Ordinary(m), GpuSequence::Ordinary(s)) => m.step_into(s, token, out),
            (Self::Qwen38(m), GpuSequence::Qwen38(s)) => m.step_into(s, token, out),
            _ => bail!("GPU model and sequence architectures differ"),
        }
    }
    pub fn prefill_into(&self, s: &mut GpuSequence, tokens: &[u32], out: &mut Step) -> Result<()> {
        match (self, s) {
            (Self::Ordinary(m), GpuSequence::Ordinary(s)) => m.prefill_into(s, tokens, out),
            (Self::Qwen38(m), GpuSequence::Qwen38(s)) => m.prefill_into(s, tokens, out),
            _ => bail!("GPU model and sequence architectures differ"),
        }
    }
    pub fn step(&self, s: &mut GpuSequence, token: u32) -> Result<Step> {
        let mut out = Step::default();
        self.step_into(s, token, &mut out)?;
        Ok(out)
    }
    pub fn prefill(&self, s: &mut GpuSequence, tokens: &[u32]) -> Result<Step> {
        let mut out = Step::default();
        self.prefill_into(s, tokens, &mut out)?;
        Ok(out)
    }
}
impl GpuSequence {
    pub fn position(&self) -> usize {
        match self {
            Self::Ordinary(s) => s.position(),
            Self::Qwen38(s) => s.position(),
        }
    }
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Self::Ordinary(s) => s.reset(),
            Self::Qwen38(s) => s.reset(),
        }
    }
    pub fn read_kv(&mut self, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Ordinary(s) => s.read_kv(layer),
            Self::Qwen38(s) => s.read_kv(layer),
        }
    }
    pub fn read_recurrent(&mut self, layer: usize) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        match self {
            Self::Ordinary(_) => Ok(None),
            Self::Qwen38(s) => s.read_recurrent(layer),
        }
    }
    pub fn read_indexer_keys(&mut self, layer: usize) -> Result<Option<Vec<f32>>> {
        match self {
            Self::Ordinary(_) => Ok(None),
            Self::Qwen38(s) => s.read_indexer_keys(layer),
        }
    }
    pub fn read_ple_history(&mut self) -> Result<Option<(Vec<f32>, [i64; 2])>> {
        match self {
            Self::Ordinary(_) => Ok(None),
            Self::Qwen38(s) => s.read_ple_history(),
        }
    }
}
