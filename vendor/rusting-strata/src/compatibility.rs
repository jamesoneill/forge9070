//! Bounded GGUF metadata checks shared by Forge9070's local and Hub routes.
//!
//! Acceptance means that the architecture, sparse-expert layout and tensor
//! types match implemented paths. It does not prove GPU memory fit, numerical
//! correctness or a token rate; those require actual loading and execution.

use crate::{gguf::{Gguf, TensorInfo, Value}, quant::GgmlType};
use anyhow::{Context, bail, ensure};
use serde::Serialize;
use std::{collections::{BTreeMap, BTreeSet}, io::Read, path::Path};

pub const HEADER_LIMIT: usize = 64 * 1024 * 1024;
pub const MAX_SHARDS: usize = 64;
const ARCHITECTURES: &[&str] = &["qwen3", "qwen3moe", "qwen3next", "qwen35", "qwen35moe", "qwen4exp", "flint"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy { Supported, MoeOnly }
impl Policy {
    pub fn name(self) -> &'static str {
        match self { Self::Supported => "supported", Self::MoeOnly => "sparse_moe_only" }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub policy: String,
    pub architecture: String,
    pub layers: u64,
    pub experts_per_layer: u64,
    pub active_experts_per_token: u64,
    pub shards: usize,
    pub tensors: usize,
    pub tensor_weight_bytes: u64,
    pub weight_memory: WeightMemory,
    pub quantization_histogram: BTreeMap<String, usize>,
    pub compression: CompressionReport,
    pub supported_metadata: bool,
    pub issues: Vec<String>,
    pub caveat: String,
}

/// File payload categories, not a demand to keep every tensor resident.
/// In qwen4exp the PLE table is gathered by row and remains lazy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct WeightMemory {
    pub routed_expert_bytes: u64,
    pub lazy_embedding_bytes: u64,
    pub other_weight_bytes: u64,
}

pub fn weight_memory(tensors: &BTreeMap<String, TensorInfo>) -> anyhow::Result<WeightMemory> {
    let mut memory = WeightMemory::default();
    for tensor in tensors.values() {
        let name = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
        let target = if [".ffn_gate_exps", ".ffn_up_exps", ".ffn_down_exps"].iter().any(|suffix| name.ends_with(suffix)) {
            &mut memory.routed_expert_bytes
        } else if matches!(name, "per_layer_token_embd" | "per_layer_tok_embd") {
            &mut memory.lazy_embedding_bytes
        } else {
            &mut memory.other_weight_bytes
        };
        *target = target.checked_add(tensor_bytes(tensor)?).context("weight memory category overflow")?;
    }
    Ok(memory)
}

/// Observed compression, not a measured accuracy score. All declared tensors
/// are counted, including any embedded draft head. Element counts avoid giving
/// a tiny normalization vector the same apparent importance as an expert bank.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CompressionReport {
    pub tensor_elements: u64,
    pub sub4bit_tensors: usize,
    pub sub4bit_elements: u64,
    pub sub4bit_formats: Vec<String>,
    pub routed_expert_tensors: usize,
    pub routed_expert_elements: u64,
    pub sub4bit_routed_expert_tensors: usize,
    pub sub4bit_routed_expert_elements: u64,
    pub unknown_type_tensors: usize,
}

fn compression_report(tensors: &BTreeMap<String, TensorInfo>) -> anyhow::Result<CompressionReport> {
    use GgmlType::*;
    let mut report = CompressionReport::default();
    let mut formats = BTreeSet::new();
    let add = |total: &mut u64, n: u64| -> anyhow::Result<()> {
        *total = total.checked_add(n).context("compression element count overflow")?;
        Ok(())
    };
    for tensor in tensors.values() {
        let elements = tensor.dims.iter().try_fold(1u64, |n, d| n.checked_mul(*d))
            .context("compression tensor geometry overflow")?;
        add(&mut report.tensor_elements, elements)?;
        let name = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
        let expert = [".ffn_gate_exps", ".ffn_up_exps", ".ffn_down_exps"].iter()
            .any(|suffix| name.ends_with(suffix));
        if expert {
            report.routed_expert_tensors += 1;
            add(&mut report.routed_expert_elements, elements)?;
        }
        let Ok(kind) = GgmlType::from_u32(tensor.ty) else {
            report.unknown_type_tensors += 1;
            continue;
        };
        // Nominal weight precision, not bytes-per-element including block
        // scales. Mixed GGUFs are inspected tensor by tensor, not by file name.
        if matches!(kind, Q2_0 | TQ1_0 | TQ2_0 | Q2K | Q3K | IQ1S | IQ1M |
            IQ2XXS | IQ2XS | IQ2S | IQ3XXS | IQ3S) {
            report.sub4bit_tensors += 1;
            add(&mut report.sub4bit_elements, elements)?;
            formats.insert(kind.name().to_owned());
            if expert {
                report.sub4bit_routed_expert_tensors += 1;
                add(&mut report.sub4bit_routed_expert_elements, elements)?;
            }
        }
    }
    report.sub4bit_formats = formats.into_iter().collect();
    Ok(report)
}

impl Report {
    pub fn ensure_supported(&self) -> anyhow::Result<()> {
        ensure!(self.supported_metadata, "model rejected by {} policy: {}", self.policy, self.issues.join("; "));
        Ok(())
    }
}

/// Only scalar metadata is retained by a remote scan. Arrays are bounded and
/// skipped; their lengths are represented by U64 for checks such as vocab size.
pub(crate) struct Header {
    pub meta: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    data_offset: u64,
    file_size: u64,
    token_count: Option<u64>,
}

#[derive(Debug)]
struct Incomplete;
impl std::fmt::Display for Incomplete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("GGUF header incomplete") }
}
impl std::error::Error for Incomplete {}

struct Cur<'a> { bytes: &'a [u8], at: usize }
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self.at.checked_add(n).context("GGUF header size overflow")?;
        ensure!(end <= HEADER_LIMIT, "GGUF metadata exceeds the 64 MiB verification limit");
        let result = self.bytes.get(self.at..end).ok_or(Incomplete)?;
        self.at = end;
        Ok(result)
    }
    fn u32(&mut self) -> anyhow::Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> anyhow::Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn string(&mut self) -> anyhow::Result<String> {
        let len = usize::try_from(self.u64()?).context("GGUF string length overflow")?;
        Ok(std::str::from_utf8(self.take(len)?).context("GGUF contains invalid UTF-8 metadata")?.to_owned())
    }
    fn skip(&mut self, ty: u32, depth: usize) -> anyhow::Result<()> {
        ensure!(depth <= 4, "GGUF nested arrays exceed the verification limit");
        match ty {
            0 | 1 | 7 => { self.take(1)?; },
            2 | 3 => { self.take(2)?; },
            4 | 5 | 6 => { self.take(4)?; },
            10 | 11 | 12 => { self.take(8)?; },
            8 => { let len = usize::try_from(self.u64()?)?; self.take(len)?; },
            9 => {
                let element = self.u32()?;
                let n = self.u64()?;
                ensure!(n <= HEADER_LIMIT as u64, "GGUF array count exceeds verification limit");
                for _ in 0..n { self.skip(element, depth + 1)?; }
            },
            _ => bail!("unknown GGUF metadata type {ty}"),
        }
        Ok(())
    }
    fn value(&mut self, ty: u32) -> anyhow::Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]), 1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?), 5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)), 7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let element = self.u32()?;
                let n = self.u64()?;
                ensure!(n <= HEADER_LIMIT as u64, "GGUF array count exceeds verification limit");
                for _ in 0..n { self.skip(element, 0)?; }
                Value::U64(n)
            },
            10 => Value::U64(self.u64()?), 11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => bail!("unknown GGUF metadata type {ty}"),
        })
    }
}

/// None means that a longer prefix is required. No weight payload is read.
pub(crate) fn parse_prefix(bytes: &[u8], file_size: u64) -> anyhow::Result<Option<Header>> {
    let parse = || -> anyhow::Result<Header> {
        let mut cursor = Cur { bytes, at: 0 };
        ensure!(cursor.take(4)? == b"GGUF", "file is not GGUF");
        ensure!(matches!(cursor.u32()?, 2 | 3), "only GGUF v2/v3 are supported");
        let (tensor_count, metadata_count) = (cursor.u64()?, cursor.u64()?);
        ensure!(tensor_count <= 200_000, "GGUF tensor count outside verification limits");
        ensure!(metadata_count <= 100_000, "GGUF metadata count outside verification limits");
        let mut meta = BTreeMap::new();
        let mut token_count = None;
        for _ in 0..metadata_count {
            let key = cursor.string()?;
            let ty = cursor.u32()?;
            let value_start = cursor.at;
            let value = cursor.value(ty)?;
            if key == "tokenizer.ggml.tokens" && ty == 9
                && u32::from_le_bytes(bytes[value_start..value_start+4].try_into()?) == 8 {
                token_count = value.as_u64();
            }
            ensure!(meta.insert(key.clone(), value).is_none(), "duplicate GGUF metadata key {key}");
        }
        // GGUF writers can put only metadata in the first split. Accept that
        // part provisionally; inspect_headers still requires the complete,
        // nonempty tensor set and checks its declared combined count.
        if tensor_count == 0 {
            let split = meta.get("split.count").and_then(Value::as_u64);
            ensure!(split.is_some_and(|n| (2..=MAX_SHARDS as u64).contains(&n))
                && meta.get("split.no").and_then(Value::as_u64) == Some(0)
                && meta.get("split.tensors.count").and_then(Value::as_u64).is_some_and(|n| n > 0),
                "zero-tensor GGUF must be the metadata-only first shard of a nonempty split model");
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = cursor.string()?;
            let n = cursor.u32()?;
            ensure!((1..=4).contains(&n), "tensor {name} needs 1 to 4 dimensions");
            let dims = (0..n).map(|_| cursor.u64()).collect::<anyhow::Result<Vec<_>>>()?;
            let (ty, offset) = (cursor.u32()?, cursor.u64()?);
            ensure!(tensors.insert(name.clone(), TensorInfo { name: name.clone(), dims, ty, offset, file: 0 }).is_none(), "duplicate tensor {name}");
        }
        let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
        ensure!(align > 0 && align.is_power_of_two() && align <= 1024 * 1024, "unsupported GGUF alignment {align}");
        let header_end = cursor.at as u64;
        // A metadata-only split may end immediately after its metadata: there
        // is no tensor-data section to align, and writers can omit its padding.
        let data_offset = if tensor_count == 0 && header_end == file_size { header_end }
            else { header_end.checked_next_multiple_of(align).context("GGUF data offset overflow")? };
        ensure!(data_offset <= file_size, "GGUF header lies past the declared file size");
        if tensor_count == 0 {
            ensure!(data_offset == file_size, "metadata-only GGUF shard has unexpected trailing payload");
        }
        Ok(Header { meta, tensors, data_offset, file_size, token_count })
    };
    match parse() {
        Ok(header) => Ok(Some(header)),
        Err(error) if error.downcast_ref::<Incomplete>().is_some() => Ok(None),
        Err(error) => Err(error),
    }
}

fn issue(issues: &mut Vec<String>, message: impl Into<String>) {
    if issues.len() < 40 { issues.push(message.into()); }
}

fn assess(meta: &BTreeMap<String, Value>, tensors: &BTreeMap<String, TensorInfo>, name: &str,
          policy: Policy, shards: usize, token_count: Option<u64>, mut issues: Vec<String>) -> Report {
    let architecture = meta.get("general.architecture").and_then(Value::as_str).unwrap_or("").to_owned();
    let number = |suffix: &str| meta.get(&format!("{architecture}.{suffix}")).and_then(Value::as_u64).unwrap_or(0);
    let total_layers = number("block_count");
    let nextn = number("nextn_predict_layers");
    let layers = total_layers.saturating_sub(nextn);
    let embd = number("embedding_length");
    let heads = number("attention.head_count");
    let kv_heads = number("attention.head_count_kv");
    let experts = number("expert_count");
    let active = number("expert_used_count");
    let dense = number("leading_dense_block_count");
    let hc = number("hyper_connection.count");
    let hc_rank = number("hyper_connection.low_rank");
    let ple = number("embedding_length_per_layer_input");
    if !ARCHITECTURES.contains(&architecture.as_str()) { issue(&mut issues, format!("architecture '{architecture}' is not supported")); }
    if name.to_ascii_lowercase().contains("iq3_k") { issue(&mut issues, "Avalon/ik_llama IQ3_K or IQ3_KS filename is not supported"); }
    if layers == 0 || layers > 4096 || nextn >= total_layers { issue(&mut issues, "invalid block_count/nextn_predict_layers"); }
    if embd == 0 || heads == 0 || kv_heads == 0 || kv_heads > heads || heads % kv_heads.max(1) != 0 {
        issue(&mut issues, "embedding and attention head dimensions must be positive and compatible");
    }
    let head_dim = number("attention.key_length");
    if head_dim == 0 && embd % heads.max(1) != 0 { issue(&mut issues, "embedding_length must divide evenly into attention heads"); }
    if !meta.get(&format!("{architecture}.attention.layer_norm_rms_epsilon")).and_then(Value::as_f32).is_some_and(|v| v.is_finite() && v > 0.0) {
        issue(&mut issues, "missing or invalid attention.layer_norm_rms_epsilon");
    }
    if experts > 512 || (experts > 0 && (active == 0 || active > experts)) { issue(&mut issues, "invalid expert count; engine supports at most 512 with valid active count"); }
    if dense > layers { issue(&mut issues, "leading_dense_block_count exceeds base layer count"); }
    if policy == Policy::MoeOnly && (experts < 2 || active == 0 || active >= experts || dense >= layers) {
        issue(&mut issues, "speed-focused catalogue requires sparse MoE: at least two experts, fewer active experts, and routed layers");
    }
    if matches!(architecture.as_str(), "qwen3next" | "qwen35" | "qwen35moe" | "qwen4exp") {
        let interval = number("full_attention_interval").max(1);
        // A missing interval means four in the engine. Only all-attention fixtures avoid SSM state.
        let has_delta = layers > 0 && (number("full_attention_interval") == 0 || interval > 1);
        if has_delta {
            let nk = number("ssm.group_count"); let nv = number("ssm.time_step_rank"); let inner = number("ssm.inner_size");
            if nk == 0 || nv == 0 || inner == 0 || nv % nk.max(1) != 0 || inner % nv.max(1) != 0
                || number("ssm.state_size") == 0 || number("ssm.conv_kernel") == 0 {
                issue(&mut issues, "missing or incompatible DeltaNet SSM dimensions");
            }
        }
    }
    let embedding = tensors.get("token_embd.weight");
    let vocab = embedding.and_then(|t| t.dims.get(1)).copied().unwrap_or(0);
    if embedding.is_none_or(|t| t.dims != [embd, vocab] || vocab == 0) { issue(&mut issues, "token_embd.weight must match [embedding_length,vocabulary]"); }
    if ple == 0 && tensors.get("output_norm.weight").is_none_or(|t| t.dims != [embd]) { issue(&mut issues, "missing or incompatible output_norm.weight"); }
    if hc > 0 {
        if hc_rank == 0 || experts == 0 { issue(&mut issues, "hyper-connections require a positive low rank and a MoE model"); }
        for suffix in ["norm", "down", "up"] {
            if !tensors.contains_key(&format!("output_hc_{suffix}.weight")) { issue(&mut issues, format!("missing output_hc_{suffix}.weight")); }
        }
    }
    if ple > 0 {
        if number("ple.ngram_size") != 3 || number("ple.heads_per_ngram") != 8 || number("ple.conv_kernel") != 4
            || ple.checked_mul(16) != Some(embd) || !tensors.contains_key("per_layer_token_embd.weight") {
            issue(&mut issues, "missing or unsupported per-layer embedding geometry");
        }
    }
    if number("attention.indexer.top_k") > 0 && (number("attention.indexer.head_count") != 4 || number("attention.indexer.key_length") != 128) {
        issue(&mut issues, "Vulkan QSA indexer requires four heads of 128 dimensions");
    }
    if tensors.get("output.weight").is_some_and(|t| t.dims != [embd, vocab]) { issue(&mut issues, "output.weight shape differs from embedding/vocabulary"); }
    if token_count != Some(vocab) || vocab == 0 { issue(&mut issues, "tokenizer string-array token count differs from embedding vocabulary"); }
    if meta.get("tokenizer.ggml.model").and_then(Value::as_str).is_none() { issue(&mut issues, "missing tokenizer.ggml.model"); }
    if layers <= 4096 {
        let hybrid = matches!(architecture.as_str(), "qwen3next" | "qwen35" | "qwen35moe" | "qwen4exp");
        let interval = match number("full_attention_interval") { 0 => 4, n => n };
        let hd = if head_dim > 0 { head_dim } else { embd / heads.max(1) };
        for layer in 0..layers {
            let tensor = |suffix: &str| tensors.get(&format!("blk.{layer}.{suffix}"));
            if hc > 0 {
                for group in ["hc_attn_", "hc_ffn_"] {
                    for suffix in ["norm", "down", "up"] {
                        let name = format!("{group}{suffix}.weight");
                        if tensor(&name).is_none() { issue(&mut issues, format!("layer {layer}: missing {name}")); }
                    }
                }
            } else {
                for suffix in ["attn_norm.weight", if matches!(architecture.as_str(), "qwen3" | "qwen3moe" | "flint") { "ffn_norm.weight" } else { "post_attention_norm.weight" }] {
                    if tensor(suffix).is_none_or(|t| t.dims != [embd]) { issue(&mut issues, format!("layer {layer}: missing/incompatible {suffix}")); }
                }
            }
            if hybrid && (layer + 1) % interval != 0 {
                for suffix in ["attn_qkv.weight", "attn_gate.weight", "ssm_dt.bias", "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight"] {
                    if tensor(suffix).is_none() { issue(&mut issues, format!("layer {layer}: missing {suffix}")); }
                }
                if tensor("ssm_ba.weight").is_none() && (tensor("ssm_beta.weight").is_none() || tensor("ssm_alpha.weight").is_none()) {
                    issue(&mut issues, format!("layer {layer}: missing separate or combined SSM beta/alpha weights"));
                }
            } else {
                let q_width = heads.checked_mul(hd).and_then(|n| n.checked_mul(if hybrid { 2 } else { 1 }));
                let kv_width = kv_heads.checked_mul(hd);
                for (suffix, rows) in [("attn_q.weight", q_width), ("attn_k.weight", kv_width), ("attn_v.weight", kv_width)] {
                    if rows.is_none() || tensor(suffix).is_none_or(|t| t.dims != [embd, rows.unwrap_or(0)]) {
                        issue(&mut issues, format!("layer {layer}: missing/incompatible {suffix}"));
                    }
                }
                if tensor("attn_output.weight").is_none_or(|t| Some(t.dims.clone()) != heads.checked_mul(hd).map(|n| vec![n,embd])) {
                    issue(&mut issues, format!("layer {layer}: missing/incompatible attn_output.weight"));
                }
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    if tensor(suffix).is_none_or(|t| t.dims.first().copied() != Some(hd)) {
                        issue(&mut issues, format!("layer {layer}: missing/incompatible {suffix}"));
                    }
                }
            }
            if experts == 0 || layer < dense {
                let ff = tensor("ffn_gate.weight").and_then(|t| t.dims.get(1)).copied().unwrap_or(0);
                if number("feed_forward_length") != ff { issue(&mut issues, format!("layer {layer}: dense FFN metadata does not match its runtime buffer dimensions")); }
                if ff == 0 || tensor("ffn_gate.weight").is_none_or(|t| t.dims != [embd,ff])
                    || tensor("ffn_up.weight").is_none_or(|t| t.dims != [embd,ff])
                    || tensor("ffn_down.weight").is_none_or(|t| t.dims != [ff,embd]) {
                    issue(&mut issues, format!("layer {layer}: missing/incompatible dense FFN tensors"));
                }
            }
            if let Some(up) = tensor("ffn_up_shexp.weight") {
                let ff = up.dims.get(1).copied().unwrap_or(0);
                if number("expert_shared_feed_forward_length") != ff { issue(&mut issues, format!("layer {layer}: shared expert metadata does not match its runtime buffer dimensions")); }
                if ff == 0 || up.dims != [embd,ff]
                    || tensor("ffn_gate_shexp.weight").is_none_or(|t| t.dims != [embd,ff])
                    || tensor("ffn_down_shexp.weight").is_none_or(|t| t.dims != [ff,embd]) {
                    issue(&mut issues, format!("layer {layer}: missing/incompatible shared expert tensors"));
                }
                if tensor("ffn_gate_inp_shexp.weight").is_some_and(|t| t.dims != [embd] && t.dims != [embd,1]) {
                    issue(&mut issues, format!("layer {layer}: incompatible shared expert gate"));
                }
            }
        }
    }
    if experts > 0 && layers <= 4096 && dense < layers {
        for layer in dense..layers {
            let tensor = |suffix: &str| tensors.get(&format!("blk.{layer}.{suffix}.weight"));
            let gate = tensor("ffn_gate_exps");
            let ff = gate.and_then(|t| t.dims.get(1)).copied().unwrap_or(0);
            if number("expert_feed_forward_length") != ff { issue(&mut issues, format!("layer {layer}: routed expert metadata does not match its runtime buffer dimensions")); }
            if tensor("ffn_gate_inp").is_none_or(|t| t.dims != [embd, experts]) {
                issue(&mut issues, format!("layer {layer}: missing/incompatible MoE router"));
            }
            if ff == 0 || gate.is_none_or(|t| t.dims != [embd, ff, experts])
                || tensor("ffn_up_exps").is_none_or(|t| t.dims != [embd, ff, experts])
                || tensor("ffn_down_exps").is_none_or(|t| t.dims != [ff, embd, experts]) {
                issue(&mut issues, format!("layer {layer}: missing/incompatible routed expert tensors"));
            }
        }
    }
    let mut histogram = BTreeMap::new();
    let mut bytes = 0u64;
    let alignment = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 1024 * 1024 { issue(&mut issues, "invalid tensor alignment"); }
    for tensor in tensors.values() {
        if shards == 1 && (alignment == 0 || tensor.offset % alignment.max(1) != 0) {
            issue(&mut issues, format!("{}: tensor offset violates GGUF alignment", tensor.name));
        }
        let kind = match GgmlType::from_u32(tensor.ty) {
            Ok(kind) => kind,
            Err(_) => {
                *histogram.entry(format!("UNSUPPORTED_ID_{}", tensor.ty)).or_default() += 1;
                issue(&mut issues, format!("{}: unsupported quantization type {}", tensor.name, tensor.ty));
                continue;
            }
        };
        *histogram.entry(kind.name().to_owned()).or_default() += 1;
        match tensor_bytes(tensor) {
            Ok(size) => match bytes.checked_add(size) {
                Some(total) => bytes = total,
                None => issue(&mut issues, "tensor weight byte count overflow"),
            },
            Err(error) => issue(&mut issues, format!("{}: {error}", tensor.name)),
        }
    }
    let compression = match compression_report(tensors) {
        Ok(report) => report,
        Err(error) => { issue(&mut issues, format!("{error:#}")); CompressionReport::default() },
    };
    let weight_memory = match weight_memory(tensors) {
        Ok(memory) => memory,
        Err(error) => { issue(&mut issues, format!("weight memory inventory: {error:#}")); WeightMemory::default() },
    };
    Report { policy: policy.name().into(), architecture, layers, experts_per_layer: experts,
        active_experts_per_token: active, shards, tensors: tensors.len(), tensor_weight_bytes: bytes,
        weight_memory, quantization_histogram: histogram, compression, supported_metadata: issues.is_empty(), issues,
        caveat: "Metadata verified against implemented architecture/expert/quantization paths. Compression is reported from actual tensor types, not evaluated answer quality. GPU memory fit, numerical fidelity, model accuracy and speed require separate execution/evaluation.".into() }
}

fn tensor_bytes(tensor: &TensorInfo) -> anyhow::Result<u64> {
    ensure!(!tensor.dims.is_empty() && tensor.dims.len() <= 4 && !tensor.dims.contains(&0), "invalid tensor dimensions");
    let kind = GgmlType::from_u32(tensor.ty)?;
    ensure!(tensor.dims[0] % kind.block_elems() as u64 == 0, "row is not aligned to quantization blocks");
    let row = (tensor.dims[0] / kind.block_elems() as u64).checked_mul(kind.block_bytes() as u64).context("tensor row size overflow")?;
    tensor.dims.iter().skip(1).try_fold(row, |size, dim| size.checked_mul(*dim).context("tensor byte size overflow"))
}

pub fn inspect_gguf(g: &Gguf, name: &str, policy: Policy) -> Report {
    let shards = g.meta.get("split.count").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
    let issues = if shards > MAX_SHARDS { vec!["model exceeds the 64-shard verification limit".into()] } else { Vec::new() };
    let token_count = g.meta.get("tokenizer.ggml.tokens").and_then(Value::as_arr)
        .filter(|values| values.iter().all(|v| v.as_str().is_some())).map(|values| values.len() as u64);
    assess(&g.meta, &g.tensors, name, policy, shards, token_count, issues)
}

pub(crate) fn inspect_headers(headers: &[Header], name: &str, policy: Policy) -> anyhow::Result<Report> {
    let first = headers.first().context("model has no GGUF headers")?;
    ensure!(headers.len() <= MAX_SHARDS, "model exceeds the 64-shard verification limit");
    let expected = first.meta.get("split.count").and_then(Value::as_u64).unwrap_or(1).max(1);
    let mut issues = Vec::new();
    if expected != headers.len() as u64 { issue(&mut issues, "GGUF split.count differs from the complete shard group"); }
    let mut tensors = BTreeMap::new();
    for (index, header) in headers.iter().enumerate() {
        if header.meta.get("split.no").and_then(Value::as_u64).is_some_and(|no| no != index as u64) {
            issue(&mut issues, "GGUF split.no differs from shard order");
        }
        if header.meta.get("split.count").and_then(Value::as_u64).is_some_and(|count| count != expected) {
            issue(&mut issues, "GGUF shards disagree on split.count");
        }
        if let Some(arch) = header.meta.get("general.architecture").and_then(Value::as_str) {
            if Some(arch) != first.meta.get("general.architecture").and_then(Value::as_str) { issue(&mut issues, "GGUF shards disagree on architecture"); }
        }
        for (name, tensor) in &header.tensors {
            let alignment = header.meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
            if tensor.offset % alignment != 0 { issue(&mut issues, format!("{name}: tensor offset violates GGUF alignment")); }
            if let Ok(size) = tensor_bytes(tensor) {
                if tensor.offset.checked_add(size).is_none_or(|end| end > header.file_size.saturating_sub(header.data_offset)) {
                    issue(&mut issues, format!("{name}: tensor extends beyond declared shard size"));
                }
            }
            if tensors.insert(name.clone(), tensor.clone()).is_some() { issue(&mut issues, format!("tensor {name} occurs in more than one shard")); }
        }
    }
    if first.meta.get("split.tensors.count").and_then(Value::as_u64).is_some_and(|count| count != tensors.len() as u64) {
        issue(&mut issues, "GGUF split.tensors.count differs from combined tensor count");
    }
    if tensors.is_empty() { issue(&mut issues, "GGUF model has no tensors across its shard group"); }
    Ok(assess(&first.meta, &tensors, name, policy, headers.len(), first.token_count, issues))
}

fn read_header(path: &Path) -> anyhow::Result<Header> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let size = file.metadata()?.len();
    let mut bytes = Vec::new();
    let mut chunk = vec![0; 1024 * 1024];
    loop {
        let remaining = HEADER_LIMIT.saturating_sub(bytes.len());
        ensure!(remaining > 0, "GGUF metadata exceeds the 64 MiB verification limit");
        let n = file.read(&mut chunk[..remaining.min(1024 * 1024)])?;
        ensure!(n > 0, "GGUF header truncated");
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(header) = parse_prefix(&bytes, size)? { return Ok(header); }
    }
}

/// Read only bounded headers, including every shard, without mapping model weights.
pub fn inspect_path(path: &Path, policy: Policy) -> anyhow::Result<Report> {
    let first = read_header(path)?;
    let count = first.meta.get("split.count").and_then(Value::as_u64).unwrap_or(1).max(1);
    ensure!(count <= MAX_SHARDS as u64, "model exceeds the 64-shard verification limit");
    let mut headers = vec![first];
    let name = path.file_name().and_then(|s| s.to_str()).context("model filename is not UTF-8")?;
    if count > 1 {
        let suffix = format!("-00001-of-{count:05}.gguf");
        let stem = name.strip_suffix(&suffix).context("select the first GGUF shard (-00001-of-XXXXX.gguf)")?;
        for part in 2..=count { headers.push(read_header(&path.with_file_name(format!("{stem}-{part:05}-of-{count:05}.gguf")))?); }
    }
    inspect_headers(&headers, name, policy)
}

/// Validate the ordinary next-token prediction head against the target model's
/// actual runtime geometry, before a GPU is opened or a working model is dropped.
/// Resolved Weights preserve embedded/companion precedence and normalisation
/// aliases. Geometry cannot establish training provenance or a useful draft
/// acceptance rate for a separately published checkpoint.
pub fn validate_mtp_geometry(model: &crate::model::Model<'_>, companion: Option<&Gguf>) -> anyhow::Result<()> {
    let c = &model.cfg;
    ensure!(c.n_nextn > 0, "model has no usable MTP head");
    ensure!(c.eagle3.is_none() && !companion.is_some_and(|g| g.tensors.contains_key("fc.weight")),
        "--mtp-file accepts ordinary matching MTP heads; EAGLE3 draft geometry is not enabled by this profile");
    ensure!(c.hc_n == 0 && c.n_embd_ple == 0 && c.idx_top_k == 0,
        "ordinary MTP validation does not support hyper-connections, per-layer embeddings or QSA");
    let n = c.n_embd;
    ensure!(n > 0 && c.n_head > 0 && c.n_head_kv > 0 && c.head_dim > 0,
        "MTP target has invalid embedding/attention dimensions");
    let multiply = |a: usize, b: usize| a.checked_mul(b).context("MTP geometry overflow");
    let q = multiply(c.n_head,c.head_dim)?;
    let kv = multiply(c.n_head_kv,c.head_dim)?;
    if let Some(aux) = companion {
        if let Some(arch) = aux.meta.get("general.architecture") {
            ensure!(arch.as_str() == Some(c.arch.as_str()), "MTP companion architecture differs from target {}",c.arch);
        }
        for (key, expected) in [
            ("embedding_length",n), ("attention.head_count",c.n_head),
            ("attention.head_count_kv",c.n_head_kv), ("attention.key_length",c.head_dim),
            ("expert_count",c.n_expert), ("expert_used_count",c.n_expert_used),
        ] {
            if let Some(value) = aux.meta.get(&format!("{}.{key}",c.arch)) {
                ensure!(value.as_u64() == Some(expected as u64), "MTP companion {key} differs from target ({expected})");
            }
        }
        for key in ["token_embd.weight","output.weight"] {
            if let Some(tensor) = aux.tensors.get(key) {
                ensure!(tensor.dims == [n as u64,c.n_vocab as u64], "MTP companion {key} vocabulary/embedding shape differs from target");
            }
        }
        if let Some(tokens) = aux.meta.get("tokenizer.ggml.tokens") {
            let tokens = tokens.as_arr().context("MTP companion tokenizer tokens are not an array")?;
            let target = model.w.g.meta.get("tokenizer.ggml.tokens").and_then(Value::as_arr).context("target tokenizer tokens missing")?;
            ensure!(tokens.len() == target.len() && tokens.iter().zip(target).all(|(a,b)| a.as_str().is_some() && a.as_str() == b.as_str()),
                "MTP companion tokenizer token IDs differ from target");
        }
        // Ordinary auxiliary weights are not materialised by Model::load. Check
        // their dimensions/types before Weights::get calculates mapped slices.
        for tensor in aux.tensors.values() { tensor_bytes(tensor).with_context(|| format!("MTP companion tensor {}",tensor.name))?; }
    }
    let layer = c.n_layer;
    let check = |suffix: &str, cols: usize, rows: usize| -> anyhow::Result<()> {
        ensure!(cols > 0 && rows > 0, "MTP expected dimensions for {suffix} are empty");
        let tensor = model.w.layer(layer,suffix).with_context(|| format!("MTP head layer {layer}: {suffix}"))?;
        ensure!(tensor.cols == cols && tensor.rows == rows,
            "MTP head {suffix}: shape {}x{} differs from target-required {cols}x{rows}",tensor.cols,tensor.rows);
        ensure!(cols % tensor.ty.block_elems() == 0, "MTP head {suffix}: misaligned quantization row");
        let expected = multiply(cols/tensor.ty.block_elems(),tensor.ty.block_bytes()).and_then(|row| multiply(row,rows))?;
        ensure!(tensor.data.len() == expected, "MTP head {suffix}: mapped byte length does not match its dimensions");
        Ok(())
    };
    for suffix in ["nextn.enorm.weight","nextn.hnorm.weight","nextn.shared_head_norm.weight","attn_norm.weight","post_attention_norm.weight"] {
        check(suffix,n,1)?;
    }
    check("nextn.eh_proj.weight",multiply(n,2)?,n)?;
    check("attn_q.weight",n,multiply(q,if c.q_gate { 2 } else { 1 })?)?;
    check("attn_k.weight",n,kv)?; check("attn_v.weight",n,kv)?;
    check("attn_output.weight",q,n)?;
    check("attn_q_norm.weight",c.head_dim,1)?; check("attn_k_norm.weight",c.head_dim,1)?;
    if c.is_moe_layer(layer) {
        ensure!(c.n_expert >= 2 && c.n_expert_used > 0 && c.n_expert_used <= c.n_expert && c.n_ff_exp > 0,
            "MTP target has invalid MoE buffer geometry");
        check("ffn_gate_inp.weight",n,c.n_expert)?;
        let ff_rows = multiply(c.n_ff_exp,c.n_expert)?;
        check("ffn_gate_exps.weight",n,ff_rows)?;
        check("ffn_up_exps.weight",n,ff_rows)?;
        check("ffn_down_exps.weight",c.n_ff_exp,multiply(n,c.n_expert)?)?;
        if model.w.layer(layer,"ffn_up_shexp.weight").is_ok() {
            ensure!(c.n_ff_shexp > 0, "MTP shared expert exceeds the target's shared-expert buffer geometry");
            check("ffn_gate_shexp.weight",n,c.n_ff_shexp)?;
            check("ffn_up_shexp.weight",n,c.n_ff_shexp)?;
            check("ffn_down_shexp.weight",c.n_ff_shexp,n)?;
            if model.w.layer(layer,"ffn_gate_inp_shexp.weight").is_ok() { check("ffn_gate_inp_shexp.weight",n,1)?; }
        }
    } else {
        check("ffn_gate.weight",n,c.n_ff)?; check("ffn_up.weight",n,c.n_ff)?;
        check("ffn_down.weight",c.n_ff,n)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_inventory_separates_experts_lazy_ple_and_dense_without_loading_payloads() {
        let tensor = |name: &str, dims: Vec<u64>, ty| TensorInfo { name:name.into(), dims, ty, offset:0, file:0 };
        let rows = [
            tensor("blk.0.ffn_gate_exps.weight", vec![256,2,4], 21),
            tensor("per_layer_token_embd.weight", vec![160,10], 20),
            tensor("output_norm.weight", vec![256], 0),
        ];
        let tensors = rows.into_iter().map(|t| (t.name.clone(),t)).collect();
        let m = weight_memory(&tensors).unwrap();
        assert_eq!(m.routed_expert_bytes, 880);
        assert_eq!(m.lazy_embedding_bytes, 900);
        assert_eq!(m.other_weight_bytes, 1024);
        let invalid = tensor("blk.0.ffn_up_exps.weight", vec![160,2,4], 21);
        assert!(weight_memory(&[(invalid.name.clone(),invalid)].into_iter().collect()).is_err());
    }
    fn dense_fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3.gguf")
    }
    struct MetadataSplit {
        dir: std::path::PathBuf,
        first: Vec<u8>,
        second: Vec<u8>,
    }
    impl MetadataSplit {
        fn path(&self, part: usize) -> std::path::PathBuf {
            self.dir.join(format!("metadata-{part:05}-of-00002.gguf"))
        }
    }
    impl Drop for MetadataSplit { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.dir); } }
    fn append_u64_meta(out: &mut Vec<u8>, key: &str, value: u64) {
        out.extend((key.len() as u64).to_le_bytes()); out.extend(key.as_bytes());
        out.extend(10u32.to_le_bytes()); out.extend(value.to_le_bytes());
    }
    fn set_u64_meta(bytes: &mut [u8], key: &str, value: u64) {
        let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let mut c = Cur { bytes, at: 24 };
        let mut at = None;
        for _ in 0..count {
            let name = c.string().unwrap(); let ty = c.u32().unwrap();
            if name == key { assert_eq!(ty,10); at = Some(c.at); break; }
            c.value(ty).unwrap();
        }
        let at = at.expect("fixture metadata field");
        bytes[at..at+8].copy_from_slice(&value.to_le_bytes());
    }
    // Split an existing valid tiny model, preserving its real tokenizer,
    // tensor descriptors and payload. The first file deliberately has no
    // tensors or alignment padding; every tensor belongs to the second map.
    fn metadata_split() -> MetadataSplit {
        use std::sync::atomic::{AtomicU64,Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf");
        let bytes = std::fs::read(path).unwrap();
        let header = parse_prefix(&bytes,bytes.len() as u64).unwrap().unwrap();
        let nt = header.tensors.len() as u64;
        let nkv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let mut c = Cur { bytes:&bytes, at:24 };
        for _ in 0..nkv { c.string().unwrap(); let ty = c.u32().unwrap(); c.value(ty).unwrap(); }
        let metadata_end = c.at;
        for _ in 0..nt { c.string().unwrap(); let nd = c.u32().unwrap(); c.take(nd as usize * 8 + 12).unwrap(); }
        let descriptors_end = c.at;
        let split = |out: &mut Vec<u8>, part| {
            for (key,value) in [("split.no",part),("split.count",2),("split.tensors.count",nt)] {
                append_u64_meta(out,key,value);
            }
        };
        let mut first = bytes[..metadata_end].to_vec();
        first[8..16].copy_from_slice(&0u64.to_le_bytes());
        first[16..24].copy_from_slice(&(nkv+3).to_le_bytes());
        split(&mut first,0);
        if first.len() % 32 == 0 {
            append_u64_meta(&mut first,"test.unpadded",1);
            first[16..24].copy_from_slice(&(nkv+4).to_le_bytes());
        }
        assert_ne!(first.len() % 32,0);
        let mut second = bytes[..24].to_vec();
        second[16..24].copy_from_slice(&3u64.to_le_bytes());
        split(&mut second,1);
        second.extend_from_slice(&bytes[metadata_end..descriptors_end]);
        second.resize(second.len().next_multiple_of(32),0);
        second.extend_from_slice(&bytes[header.data_offset as usize..]);
        let dir = std::env::temp_dir().join(format!("forge-metadata-split-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = MetadataSplit { dir, first, second };
        std::fs::write(fixture.path(1),&fixture.first).unwrap();
        std::fs::write(fixture.path(2),&fixture.second).unwrap();
        fixture
    }
    struct MtpTemp(std::path::PathBuf);
    impl Drop for MtpTemp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
    fn mtp_companion(base: &Gguf) -> MtpTemp {
        use std::sync::atomic::{AtomicU64,Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("forge-mtp-geometry-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).unwrap();
        let text = |out: &mut Vec<u8>, value: &str| { out.extend((value.len() as u64).to_le_bytes()); out.extend(value.as_bytes()); };
        let mut tensors: Vec<(String,Vec<u64>,u32,Vec<u8>)> = base.tensors.values()
            .filter_map(|t| t.name.strip_prefix("blk.0.").map(|suffix|
                (format!("blk.1.{suffix}"),t.dims.clone(),t.ty,base.bytes(t).to_vec()))).collect();
        for suffix in ["enorm","hnorm","shared_head_norm"] {
            let t = &base.tensors["output_norm.weight"];
            tensors.push((format!("blk.1.nextn.{suffix}.weight"),t.dims.clone(),t.ty,base.bytes(t).to_vec()));
        }
        tensors.push(("blk.1.nextn.eh_proj.weight".into(),vec![64,32],0,vec![0;64*32*4]));
        let mut header = b"GGUF".to_vec(); header.extend(3u32.to_le_bytes());
        header.extend((tensors.len() as u64).to_le_bytes()); header.extend(2u64.to_le_bytes());
        text(&mut header,"general.architecture"); header.extend(8u32.to_le_bytes()); text(&mut header,"qwen3moe");
        text(&mut header,"qwen3moe.embedding_length"); header.extend(4u32.to_le_bytes()); header.extend(32u32.to_le_bytes());
        let mut data = Vec::new();
        for (name,dims,ty,bytes) in tensors {
            data.resize(data.len().next_multiple_of(32),0);
            text(&mut header,&name); header.extend((dims.len() as u32).to_le_bytes());
            for dim in dims { header.extend(dim.to_le_bytes()); }
            header.extend(ty.to_le_bytes()); header.extend((data.len() as u64).to_le_bytes()); data.extend(bytes);
        }
        header.resize(header.len().next_multiple_of(32),0); header.extend(data);
        std::fs::write(dir.join("head.gguf"),header).unwrap(); MtpTemp(dir)
    }

    #[test]
    fn matching_standalone_mtp_geometry_accepts_normalization_aliases() {
        let base = Gguf::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf")).unwrap();
        let temp = mtp_companion(&base); let aux = Gguf::open(&temp.0.join("head.gguf")).unwrap();
        assert!(aux.tensors.contains_key("blk.1.ffn_norm.weight"));
        assert!(!aux.tensors.contains_key("blk.1.post_attention_norm.weight"));
        let model = crate::model::Model::load_with_aux(&base,Some(&aux),&[]).unwrap();
        validate_mtp_geometry(&model,Some(&aux)).unwrap();
    }

    #[test]
    fn mismatched_projection_and_target_metadata_are_rejected_before_gpu_use() {
        let base = Gguf::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf")).unwrap();
        let temp = mtp_companion(&base); let mut aux = Gguf::open(&temp.0.join("head.gguf")).unwrap();
        aux.tensors.get_mut("blk.1.nextn.eh_proj.weight").unwrap().dims[0] = 32;
        {
            let model = crate::model::Model::load_with_aux(&base,Some(&aux),&[]).unwrap();
            assert_eq!(model.cfg.n_nextn,1); // The old loader alone accepted this wrong shape.
            assert!(validate_mtp_geometry(&model,Some(&aux)).unwrap_err().to_string().contains("eh_proj"));
        }
        aux.tensors.get_mut("blk.1.nextn.eh_proj.weight").unwrap().dims[0] = 64;
        aux.meta.insert("qwen3moe.embedding_length".into(),Value::U32(64));
        let model = crate::model::Model::load_with_aux(&base,Some(&aux),&[]).unwrap();
        assert!(validate_mtp_geometry(&model,Some(&aux)).unwrap_err().to_string().contains("embedding_length"));
    }

    #[test]
    fn headless_missing_attention_and_wrong_token_ids_are_rejected() {
        let base = Gguf::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf")).unwrap();
        assert!(validate_mtp_geometry(&crate::model::Model::load(&base).unwrap(),None).is_err());
        let temp = mtp_companion(&base); let mut aux = Gguf::open(&temp.0.join("head.gguf")).unwrap();
        let mut tokens = base.meta["tokenizer.ggml.tokens"].as_arr().unwrap().to_vec();
        tokens[0] = Value::Str("different token".into()); aux.meta.insert("tokenizer.ggml.tokens".into(),Value::Arr(tokens));
        {
            let model = crate::model::Model::load_with_aux(&base,Some(&aux),&[]).unwrap();
            assert!(validate_mtp_geometry(&model,Some(&aux)).unwrap_err().to_string().contains("token IDs"));
        }
        aux.meta.remove("tokenizer.ggml.tokens"); aux.tensors.remove("blk.1.attn_k.weight");
        let model = crate::model::Model::load_with_aux(&base,Some(&aux),&[]).unwrap();
        assert!(validate_mtp_geometry(&model,Some(&aux)).unwrap_err().to_string().contains("attn_k"));
    }
    #[test]
    fn dense_fixture_is_diagnostic_only_under_sparse_policy() {
        let supported = inspect_path(&dense_fixture(), Policy::Supported).unwrap();
        supported.ensure_supported().unwrap();
        let rejected = inspect_path(&dense_fixture(), Policy::MoeOnly).unwrap();
        assert!(!rejected.supported_metadata);
        assert!(rejected.issues.iter().any(|s| s.contains("sparse MoE")));
    }
    #[test]
    fn prefix_scanner_needs_headers_not_weight_payload_and_checks_size() {
        let bytes = std::fs::read(dense_fixture()).unwrap();
        assert!(parse_prefix(&bytes[..20], bytes.len() as u64).unwrap().is_none());
        let full = parse_prefix(&bytes, bytes.len() as u64).unwrap().unwrap();
        let prefix = parse_prefix(&bytes[..full.data_offset as usize], bytes.len() as u64).unwrap().unwrap();
        inspect_headers(&[prefix], "fixture.gguf", Policy::Supported).unwrap().ensure_supported().unwrap();
        let short = parse_prefix(&bytes, bytes.len() as u64 - 1).unwrap().unwrap();
        assert!(!inspect_headers(&[short], "fixture.gguf", Policy::Supported).unwrap().supported_metadata);
    }
    #[test]
    fn metadata_only_first_shard_accepts_unpadded_eof_and_maps_later_tensors() {
        let fixture = metadata_split();
        let header = parse_prefix(&fixture.first,fixture.first.len() as u64).unwrap().unwrap();
        assert!(header.tensors.is_empty());
        assert_eq!(header.data_offset,fixture.first.len() as u64);
        let report = inspect_path(&fixture.path(1),Policy::MoeOnly).unwrap();
        report.ensure_supported().unwrap();
        assert_eq!(report.shards,2);
        let base = Gguf::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf")).unwrap();
        let mapped = Gguf::open(&fixture.path(1)).unwrap();
        assert_eq!(mapped.tensors.len(),base.tensors.len());
        for (name,tensor) in &mapped.tensors {
            assert_eq!(tensor.file,1);
            assert_eq!(mapped.bytes(tensor),base.bytes(&base.tensors[name]));
        }
        let runtime = inspect_gguf(&mapped,"metadata-00001-of-00002.gguf",Policy::MoeOnly);
        runtime.ensure_supported().unwrap();
        assert_eq!(report.tensor_weight_bytes,runtime.tensor_weight_bytes);
        assert_eq!(report.compression,runtime.compression);
    }
    #[test]
    fn metadata_only_first_shard_requires_a_complete_nonempty_model() {
        let fixture = metadata_split();
        let first = parse_prefix(&fixture.first,fixture.first.len() as u64).unwrap().unwrap();
        let incomplete = inspect_headers(&[first],"metadata-00001-of-00002.gguf",Policy::MoeOnly).unwrap();
        assert!(incomplete.ensure_supported().is_err());
        assert!(incomplete.issues.iter().any(|s| s.contains("complete shard group")));
        std::fs::remove_file(fixture.path(2)).unwrap();
        assert!(inspect_path(&fixture.path(1),Policy::MoeOnly).is_err());
        assert!(Gguf::open(&fixture.path(1)).is_err());

        let mut empty_second = fixture.first.clone();
        set_u64_meta(&mut empty_second,"split.no",1);
        std::fs::write(fixture.path(2),&empty_second).unwrap();
        assert!(inspect_path(&fixture.path(1),Policy::MoeOnly).is_err());
        let mut headers: Vec<_> = (0..2).map(|_| parse_prefix(&fixture.first,fixture.first.len() as u64).unwrap().unwrap()).collect();
        headers[1].meta.insert("split.no".into(),Value::U64(1));
        let empty = inspect_headers(&headers,"metadata-00001-of-00002.gguf",Policy::MoeOnly).unwrap();
        assert!(empty.ensure_supported().is_err());
        assert!(empty.issues.iter().any(|s| s.contains("no tensors")));

        for (key,value) in [("split.count",1),("split.count",0),("split.no",1),("split.tensors.count",0)] {
            let mut malformed = fixture.first.clone(); set_u64_meta(&mut malformed,key,value);
            assert!(parse_prefix(&malformed,malformed.len() as u64).is_err(),"{key}={value}");
        }
        let mut standalone = b"GGUF".to_vec(); standalone.extend(3u32.to_le_bytes()); standalone.extend([0;16]);
        assert!(parse_prefix(&standalone,standalone.len() as u64).is_err());
    }
    #[test]
    fn metadata_only_shards_preserve_header_bounds_alignment_and_combined_counts() {
        let fixture = metadata_split();
        let size = fixture.first.len() as u64;
        assert!(parse_prefix(&fixture.first[..fixture.first.len()-1],size).unwrap().is_none());
        assert!(parse_prefix(&fixture.first,size-1).is_err());
        let mut padded = fixture.first.clone(); padded.resize(padded.len().next_multiple_of(32),0);
        assert!(parse_prefix(&padded,padded.len() as u64).unwrap().is_some());
        padded.push(0);
        assert!(parse_prefix(&padded,padded.len() as u64).is_err());
        for alignment in [0,3,2*1024*1024] {
            let mut malformed = b"GGUF".to_vec(); malformed.extend(3u32.to_le_bytes());
            malformed.extend(0u64.to_le_bytes()); malformed.extend(4u64.to_le_bytes());
            for (key,value) in [("split.no",0),("split.count",2),("split.tensors.count",1),("general.alignment",alignment)] {
                append_u64_meta(&mut malformed,key,value);
            }
            assert!(parse_prefix(&malformed,malformed.len() as u64).err().unwrap().to_string().contains("alignment"));
            let path = fixture.dir.join("bad-alignment.gguf"); std::fs::write(&path,&malformed).unwrap();
            assert!(Gguf::open(&path).err().unwrap().to_string().contains("alignment"));
        }
        let mut headers = vec![
            parse_prefix(&fixture.first,size).unwrap().unwrap(),
            parse_prefix(&fixture.second,fixture.second.len() as u64).unwrap().unwrap(),
        ];
        let count = headers[1].tensors.len() as u64;
        headers[0].meta.insert("split.tensors.count".into(),Value::U64(count+1));
        let wrong_count = inspect_headers(&headers,"metadata-00001-of-00002.gguf",Policy::MoeOnly).unwrap();
        assert!(wrong_count.issues.iter().any(|s| s.contains("combined tensor count")));
        headers[0].meta.insert("split.tensors.count".into(),Value::U64(count));
        headers[1].file_size -= 1;
        let truncated = inspect_headers(&headers,"metadata-00001-of-00002.gguf",Policy::MoeOnly).unwrap();
        assert!(truncated.issues.iter().any(|s| s.contains("beyond declared shard size")));
        std::fs::write(fixture.path(2),&fixture.second[..fixture.second.len()-1]).unwrap();
        assert!(inspect_path(&fixture.path(1),Policy::MoeOnly).unwrap().ensure_supported().is_err());
        assert!(Gguf::open(&fixture.path(1)).is_err());
    }
    #[test]
    fn remote_and_local_metadata_checks_agree() {
        let path = dense_fixture(); let g = Gguf::open(&path).unwrap();
        let local = inspect_gguf(&g, "fixture.gguf", Policy::Supported);
        let header = inspect_path(&path, Policy::Supported).unwrap();
        assert_eq!(local.tensor_weight_bytes, header.tensor_weight_bytes);
        assert_eq!(local.quantization_histogram, header.quantization_histogram);
        assert_eq!(local.compression, header.compression);
        assert_eq!(local.supported_metadata, header.supported_metadata);
    }
    #[test]
    fn compression_reports_actual_mixed_types_and_routed_expert_elements() {
        let tensor = |name: &str, ty, dims| TensorInfo { name:name.into(), ty, dims, offset:0, file:0 };
        let values = [
            tensor("blk.0.ffn_gate_exps.weight",11,vec![256,16,8]), // Q3_K expert bank
            tensor("blk.0.ffn_up_exps.weight",12,vec![256,16,8]),   // Q4_K expert bank
            tensor("blk.0.ffn_down_exps.weight",13,vec![256,16,8]), // Q5_K expert bank
            tensor("output_norm.weight",0,vec![256]),
        ];
        let tensors = values.into_iter().map(|t| (t.name.clone(),t)).collect();
        let report = compression_report(&tensors).unwrap();
        assert_eq!(report.tensor_elements,3*32768+256);
        assert_eq!((report.sub4bit_tensors,report.sub4bit_elements),(1,32768));
        assert_eq!((report.routed_expert_tensors,report.routed_expert_elements),(3,3*32768));
        assert_eq!((report.sub4bit_routed_expert_tensors,report.sub4bit_routed_expert_elements),(1,32768));
        assert_eq!(report.sub4bit_formats,["Q3_K"]);
        assert_eq!(report.unknown_type_tensors,0);
    }
    #[test]
    fn higher_precision_experts_do_not_hide_low_bit_output_or_unknown_types() {
        let tensor = |name: &str, ty, dims| TensorInfo { name:name.into(), ty, dims, offset:0, file:0 };
        let values = [
            tensor("blk.0.ffn_up_exps",13,vec![256,16,8]), // name without .weight is also recognized
            tensor("output.weight",18,vec![256,512]),    // IQ3_XXS outside the expert banks
            tensor("unknown.weight",999,vec![256]),
        ];
        let tensors = values.into_iter().map(|t| (t.name.clone(),t)).collect();
        let report = compression_report(&tensors).unwrap();
        assert_eq!((report.sub4bit_tensors,report.sub4bit_elements),(1,131072));
        assert_eq!(report.routed_expert_tensors,1);
        assert_eq!(report.sub4bit_routed_expert_tensors,0);
        assert_eq!(report.sub4bit_formats,["IQ3_XXS"]);
        assert_eq!(report.unknown_type_tensors,1);
    }
    #[test]
    fn malformed_prefix_is_rejected_without_unbounded_allocation() {
        let mut bytes = b"GGUF".to_vec(); bytes.extend(3u32.to_le_bytes());
        bytes.extend(u64::MAX.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
        assert!(parse_prefix(&bytes, u64::MAX).is_err());
        let tensor = TensorInfo { name:"bad".into(),dims:vec![u64::MAX,2],ty:0,offset:0,file:0 };
        assert!(tensor_bytes(&tensor).is_err());
    }
    #[test]
    fn sparse_fixture_passes_but_missing_attention_and_custom_quant_do_not() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf");
        let report = inspect_path(&path, Policy::MoeOnly).unwrap();
        report.ensure_supported().unwrap();
        assert_eq!((report.experts_per_layer, report.active_experts_per_token), (4,2));
        let mut g = Gguf::open(&path).unwrap();
        g.tensors.remove("blk.0.attn_q.weight");
        assert!(!inspect_gguf(&g,"fixture.gguf",Policy::MoeOnly).supported_metadata);
        let mut g = Gguf::open(&path).unwrap();
        g.tensors.get_mut("blk.0.ffn_gate_exps.weight").unwrap().ty = 43;
        assert!(!inspect_gguf(&g,"fixture.gguf",Policy::MoeOnly).supported_metadata);
        assert!(!inspect_gguf(&Gguf::open(&path).unwrap(),"model-IQ3_KS.gguf",Policy::MoeOnly).supported_metadata);
    }
    #[test]
    fn partial_shared_expert_and_misaligned_offsets_are_rejected() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf");
        let mut g = Gguf::open(&path).unwrap();
        let mut shared = g.tensors["blk.0.ffn_up_exps.weight"].clone();
        shared.name = "blk.0.ffn_up_shexp.weight".into(); shared.dims = vec![32,64];
        g.tensors.insert(shared.name.clone(),shared);
        let report = inspect_gguf(&g,"fixture.gguf",Policy::MoeOnly);
        assert!(report.issues.iter().any(|s| s.contains("shared expert tensors")));
        let mut g = Gguf::open(&path).unwrap();
        g.tensors.get_mut("token_embd.weight").unwrap().offset = 1;
        let report = inspect_gguf(&g,"fixture.gguf",Policy::MoeOnly);
        assert!(report.issues.iter().any(|s| s.contains("alignment")));
        let mut g = Gguf::open(&path).unwrap();
        g.meta.insert("qwen3moe.expert_feed_forward_length".into(),Value::U32(1));
        let report = inspect_gguf(&g,"fixture.gguf",Policy::MoeOnly);
        assert!(report.issues.iter().any(|s| s.contains("runtime buffer")));
    }
}
