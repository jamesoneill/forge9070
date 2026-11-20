//! Independent Qwen model execution and an explicit matrix-executor boundary.
//!
//! This module uses the published model equations, not another inference engine:
//! https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3
//! https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe
//! GGUF names/layout follow https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
//! No code from those implementations or RustingStrata is copied into this module.
//!
//! Immutable weights remain mapped in their original storage format. Only small
//! normalization vectors are expanded at load. Each request owns its KV state and
//! reusable scratch buffers. Qwen3.8 text execution is implemented in qwen38.rs.
//! An explicit executor may accelerate matrix projections; sequence/nonlinear
//! operations here remain on the host. Speculative decoding is not implemented.

use crate::gguf::{Gguf, Tensor, Value};
use crate::math;
use crate::qwen38::{Qwen38, Qwen38State};
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Matrix projection boundary used by the CPU and explicit hybrid Vulkan paths.
/// Implementations may retain immutable tensors using Tensor::cache_key().
pub trait MatvecExecutor: Send + Sync + std::fmt::Debug {
    fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> Result<()>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Default)]
pub struct CpuMatvec;
impl MatvecExecutor for CpuMatvec {
    fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
        tensor.matvec_into(input, output)
    }
    fn name(&self) -> &'static str {
        "cpu"
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub architecture: String,
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rope_dimension_count: usize,
    pub rope_base: f32,
    pub rms_epsilon: f32,
    pub vocab_size: usize,
    pub context_length: usize,
    pub training_context_length: usize,
    pub feed_forward_length: usize,
    pub expert_count: usize,
    pub experts_per_token: usize,
    pub expert_feed_forward_length: usize,
    pub normalize_expert_weights: bool,
    pub expert_weights_scale: f32,
}

#[derive(Debug)]
enum FeedForward {
    Dense {
        gate: String,
        up: String,
        down: String,
    },
    Moe {
        router: String,
        gate: String,
        up: String,
        down: String,
    },
}

#[derive(Debug)]
struct Layer {
    attention_norm: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    feed_forward_norm: Vec<f32>,
    query: String,
    key: String,
    value: String,
    attention_output: String,
    feed_forward: FeedForward,
}

/// The model is immutable after load and can be shared between CPU requests.
/// CPU concurrency is ordinary host computation, not simultaneous GPU decoding.
#[derive(Debug)]
pub struct Model {
    source: Arc<Gguf>,
    executor: Arc<dyn MatvecExecutor>,
    qwen38: Option<Qwen38>,
    config: Config,
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    output_tensor: String,
    rope_inverse: Vec<f32>,
    owner: Arc<()>,
}

#[derive(Debug, Default)]
struct Kv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct Scratch {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    expert_output: Vec<f32>,
    router: Vec<f32>,
    selected: Vec<(usize, f32)>,
    router_probabilities: Vec<f32>,
    attention_scores: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Scratch {
    fn new(c: &Config) -> Self {
        let ffn = if c.expert_count == 0 {
            c.feed_forward_length
        } else {
            c.expert_feed_forward_length
        };
        Self {
            residual: vec![0.0; c.embedding_length],
            normalized: vec![0.0; c.embedding_length],
            query: vec![0.0; c.head_count * c.head_dim],
            key: vec![0.0; c.head_count_kv * c.head_dim],
            value: vec![0.0; c.head_count_kv * c.head_dim],
            attention: vec![0.0; c.head_count * c.head_dim],
            projected: vec![0.0; c.embedding_length],
            gate: vec![0.0; ffn],
            up: vec![0.0; ffn],
            expert_output: vec![0.0; c.embedding_length],
            router: vec![0.0; c.expert_count],
            selected: Vec::with_capacity(c.expert_count),
            router_probabilities: Vec::with_capacity(c.expert_count),
            attention_scores: Vec::new(),
            rope_cos: vec![0.0; c.rope_dimension_count / 2],
            rope_sin: vec![0.0; c.rope_dimension_count / 2],
        }
    }
}

/// Independent F32 KV cache plus scratch for one request. No KV or scratch is
/// shared with another sequence. Reset retains allocations for the next request.
#[derive(Debug)]
pub struct Sequence {
    owner: Arc<()>,
    position: usize,
    kv_width: usize,
    layers: Vec<Kv>,
    scratch: Scratch,
    qwen38: Option<Qwen38State>,
}

impl Sequence {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        self.rewind(0);
        if let Some(state) = &mut self.qwen38 {
            state.reset();
        }
    }

    /// The initialized cache prefix: token-major, then KV head, then coordinate.
    pub fn kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        if let Some(state) = &self.qwen38 {
            return state.kv(layer);
        }
        self.layers
            .get(layer)
            .map(|kv| (kv.keys.as_slice(), kv.values.as_slice()))
    }

    /// Recurrent convolution and matrix state, when this layer is DeltaNet.
    pub fn recurrent(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        self.qwen38.as_ref()?.recurrent(layer)
    }

    pub fn indexer_keys(&self, layer: usize) -> Option<&[f32]> {
        self.qwen38.as_ref()?.indexer_keys(layer)
    }
    pub fn ple_history(&self) -> Option<(&[f32], [i64; 2])> {
        self.qwen38.as_ref().map(Qwen38State::ple_history)
    }

    fn rewind(&mut self, position: usize) {
        for kv in &mut self.layers {
            kv.keys.truncate(position * self.kv_width);
            kv.values.truncate(position * self.kv_width);
        }
        self.position = position;
    }
}

#[derive(Clone, Debug, Default)]
pub struct Step {
    /// Full-vocabulary logits after the current token.
    pub logits: Vec<f32>,
    /// Final RMS-normalized residual used as the input to the language-model head.
    pub hidden: Vec<f32>,
}

fn integer(g: &Gguf, key: &str) -> Result<usize> {
    let n = g
        .meta
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("{key}: expected a nonnegative integer"))?;
    usize::try_from(n).with_context(|| format!("{key}: value exceeds addressable range"))
}

fn optional_integer(g: &Gguf, key: &str, default: usize) -> Result<usize> {
    if g.meta.contains_key(key) {
        integer(g, key)
    } else {
        Ok(default)
    }
}

fn optional_float(g: &Gguf, key: &str, default: f32) -> Result<f32> {
    match g.meta.get(key) {
        Some(v) => v
            .as_f32()
            .filter(|x| x.is_finite())
            .with_context(|| format!("{key}: expected a finite number")),
        None => Ok(default),
    }
}

fn required_positive_float(g: &Gguf, key: &str) -> Result<f32> {
    g.meta
        .get(key)
        .with_context(|| format!("{key}: required model metadata is missing"))?
        .as_f32()
        .filter(|x| x.is_finite() && *x > 0.0)
        .with_context(|| format!("{key}: expected a finite positive number"))
}

fn optional_bool(g: &Gguf, key: &str, default: bool) -> Result<bool> {
    match g.meta.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .or_else(|| match v.as_u64() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                _ => None,
            })
            .with_context(|| format!("{key}: expected a boolean or integer 0/1")),
    }
}

fn checked_width(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= 1_048_576)
        .with_context(|| format!("{name}: geometry exceeds the CPU reference limit"))
}

fn check_shape(
    g: &Gguf,
    used: &mut BTreeSet<String>,
    name: String,
    dims: &[usize],
) -> Result<String> {
    let t = g.tensor(&name)?;
    ensure!(
        t.dims.len() == dims.len() && t.dims.iter().zip(dims).all(|(a, b)| *a == *b as u64),
        "{name}: dimensions {:?}, expected {:?}",
        t.dims,
        dims
    );
    used.insert(name.clone());
    Ok(name)
}

fn norm_vector(g: &Gguf, used: &mut BTreeSet<String>, name: String, n: usize) -> Result<Vec<f32>> {
    check_shape(g, used, name.clone(), &[n])?;
    let vector = g.tensor(&name)?.row(0)?;
    math::finite(&vector, &name)?;
    Ok(vector)
}

fn reject_unsupported_features(g: &Gguf, arch: &str) -> Result<()> {
    // Architecture acceptance does not imply arbitrary combinations of features.
    // Reject variants explicitly, before any state is allocated or token decoded.
    for suffix in [
        "nextn_predict_layers",
        "leading_dense_block_count",
        "expert_shared_count",
        "expert_group_count",
        "expert_group_used_count",
        "expert_shared_feed_forward_length",
        "shared_expert_feed_forward_length",
        "attention.sliding_window",
        "attention.sliding_window_pattern",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.indexer.head_count",
        "attention.indexer.top_k",
        "hyperconnection.count",
        "hyperconnection.rank",
        "hyper_connection_count",
        "embedding_length_per_layer",
        "ssm.inner_size",
        "ssm.state_size",
        "ssm.conv_kernel",
        "full_attention_interval",
    ] {
        let key = format!("{arch}.{suffix}");
        ensure!(
            optional_integer(g, &key, 0)? == 0,
            "independent Forge CPU engine does not support {key}; use the explicit reference engine for this variant"
        );
    }
    for suffix in [
        "rope.dimension_sections",
        "rope.mrope_section",
        "attention.layer_types",
    ] {
        let key = format!("{arch}.{suffix}");
        ensure!(
            !g.meta.contains_key(&key),
            "independent Forge CPU engine does not support {key}"
        );
    }
    let scaling_key = format!("{arch}.rope.scaling.type");
    if let Some(value) = g.meta.get(&scaling_key) {
        ensure!(
            value.as_str() == Some("none"),
            "independent Forge CPU engine supports ordinary unscaled RoPE only ({scaling_key})"
        );
    }
    for suffix in [
        "rope.scaling.factor",
        "rope.freq_scale",
        "rope.scale_linear",
    ] {
        ensure!(
            optional_float(g, &format!("{arch}.{suffix}"), 1.0)? == 1.0,
            "independent Forge CPU engine does not support scaled RoPE ({suffix})"
        );
    }
    for suffix in [
        "attention.logit_softcapping",
        "final_logit_softcapping",
        "router.logit_softcapping",
    ] {
        ensure!(
            optional_float(g, &format!("{arch}.{suffix}"), 0.0)? == 0.0,
            "independent Forge CPU engine does not support {suffix}"
        );
    }
    for suffix in ["embedding_scale", "residual_scale", "attention.scale"] {
        ensure!(
            optional_float(g, &format!("{arch}.{suffix}"), 1.0)? == 1.0,
            "independent Forge CPU engine does not support custom {suffix}"
        );
    }
    let router_function = optional_integer(g, &format!("{arch}.expert_gating_func"), 0)?;
    ensure!(
        router_function <= 1,
        "independent Forge CPU engine supports softmax expert routing only"
    );
    ensure!(
        optional_bool(g, &format!("{arch}.attention.causal"), true)?,
        "independent Forge CPU engine supports causal attention only"
    );
    ensure!(
        !optional_bool(g, &format!("{arch}.attention.use_sliding_window"), false)?,
        "independent Forge CPU engine does not support sliding-window attention"
    );
    ensure!(
        !g.meta.contains_key("adapter.type"),
        "load a base model, not an adapter, in the independent Forge CPU engine"
    );
    if let Some(kind) = g.meta.get("general.type") {
        ensure!(
            kind.as_str() == Some("model"),
            "independent Forge CPU engine only loads ordinary model GGUF files"
        );
    }
    Ok(())
}

impl Model {
    pub fn load(source: Arc<Gguf>, context: usize) -> Result<Self> {
        Self::load_with_executor(source, context, Arc::new(CpuMatvec))
    }

    pub fn load_with_executor(
        source: Arc<Gguf>,
        context: usize,
        executor: Arc<dyn MatvecExecutor>,
    ) -> Result<Self> {
        let architecture = source
            .meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .context("general.architecture must be a string")?
            .to_owned();
        if architecture == "qwen4exp" {
            let qwen38 = Qwen38::load(Arc::clone(&source), context, Arc::clone(&executor))?;
            return Ok(Self {
                config: qwen38.common_config().clone(),
                qwen38: Some(qwen38),
                source,
                executor,
                layers: Vec::new(),
                output_norm: Vec::new(),
                output_tensor: String::new(),
                rope_inverse: Vec::new(),
                owner: Arc::new(()),
            });
        }
        ensure!(
            matches!(architecture.as_str(), "qwen3" | "qwen3moe"),
            "independent Forge model does not support architecture {architecture:?}; supported: qwen3, qwen3moe, qwen4exp"
        );
        reject_unsupported_features(&source, &architecture)?;
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let block_count = integer(&source, &key("block_count"))?;
        let embedding_length = integer(&source, &key("embedding_length"))?;
        let head_count = integer(&source, &key("attention.head_count"))?;
        let head_count_kv = integer(&source, &key("attention.head_count_kv"))?;
        ensure!(
            (1..=512).contains(&block_count),
            "CPU reference supports 1..=512 layers"
        );
        ensure!(
            (1..=65536).contains(&embedding_length),
            "embedding width is outside the CPU reference limit"
        );
        ensure!(
            head_count > 0 && head_count_kv > 0 && head_count % head_count_kv == 0,
            "attention head count must be a positive multiple of KV head count"
        );
        let head_dim = if source.meta.contains_key(&key("attention.key_length")) {
            integer(&source, &key("attention.key_length"))?
        } else {
            ensure!(
                embedding_length % head_count == 0,
                "embedding width is not divisible by heads and key_length is missing"
            );
            embedding_length / head_count
        };
        ensure!(
            head_dim > 0 && head_dim <= 65536,
            "invalid attention head dimension"
        );
        ensure!(
            optional_integer(&source, &key("attention.value_length"), head_dim)? == head_dim,
            "independent Forge CPU engine requires equal key and value head dimensions"
        );
        let query_width = checked_width(head_count, head_dim, "query projection")?;
        let kv_width = checked_width(head_count_kv, head_dim, "KV projection")?;
        let rope_dimension_count =
            optional_integer(&source, &key("rope.dimension_count"), head_dim)?;
        ensure!(
            rope_dimension_count == head_dim && rope_dimension_count % 2 == 0,
            "independent Qwen3 CPU inference requires a full even head rotary dimension; partial RoPE is unsupported"
        );
        // GGUF does not define missing-key defaults for these model parameters.
        let rope_base = required_positive_float(&source, &key("rope.freq_base"))?;
        let rms_epsilon =
            required_positive_float(&source, &key("attention.layer_norm_rms_epsilon"))?;
        let training_context_length = integer(&source, &key("context_length"))?;
        ensure!(
            context > 0 && context <= training_context_length && context <= 1_048_576,
            "requested context {context} must be positive, within model context {training_context_length}, and at most 1048576; RoPE extrapolation is unsupported"
        );
        let embeddings = source.tensor("token_embd.weight")?;
        ensure!(
            embeddings.dims.len() == 2 && embeddings.dims[0] == embedding_length as u64,
            "token_embd.weight must have [embedding_length, vocabulary] dimensions"
        );
        let vocab_size = usize::try_from(embeddings.dims[1]).context("vocabulary size overflow")?;
        ensure!(
            (1..=2_000_000).contains(&vocab_size),
            "vocabulary outside CPU reference limit"
        );
        if let Some(tokens) = source.meta.get("tokenizer.ggml.tokens") {
            ensure!(
                tokens.as_array().is_some_and(|v| v.len() == vocab_size),
                "tokenizer vocabulary and embedding rows disagree"
            );
        }
        let is_moe = architecture == "qwen3moe";
        let expert_count = optional_integer(&source, &key("expert_count"), 0)?;
        let experts_per_token = optional_integer(&source, &key("expert_used_count"), 0)?;
        let expert_feed_forward_length =
            optional_integer(&source, &key("expert_feed_forward_length"), 0)?;
        let feed_forward_length = optional_integer(&source, &key("feed_forward_length"), 0)?;
        if is_moe {
            ensure!(
                (1..=65536).contains(&expert_count)
                    && experts_per_token > 0
                    && experts_per_token <= expert_count,
                "qwen3moe requires a positive expert count and 1..=expert_count selected experts"
            );
            ensure!(
                (1..=1_048_576).contains(&expert_feed_forward_length),
                "invalid expert feed-forward width"
            );
        } else {
            ensure!(
                expert_count == 0 && experts_per_token == 0 && expert_feed_forward_length == 0,
                "qwen3 dense model contains MoE metadata; use the correct architecture"
            );
            ensure!(
                (1..=1_048_576).contains(&feed_forward_length),
                "invalid dense feed-forward width"
            );
        }
        let normalize_expert_weights = optional_bool(&source, &key("expert_weights_norm"), true)?;
        let stored_scale = optional_float(&source, &key("expert_weights_scale"), 0.0)?;
        ensure!(
            stored_scale >= 0.0,
            "expert weight scale cannot be negative"
        );
        // GGUF's zero default means no additional scaling, not zero contribution.
        let expert_weights_scale = if stored_scale == 0.0 {
            1.0
        } else {
            stored_scale
        };
        let config = Config {
            architecture,
            block_count,
            embedding_length,
            head_count,
            head_count_kv,
            head_dim,
            rope_dimension_count,
            rope_base,
            rms_epsilon,
            vocab_size,
            context_length: context,
            training_context_length,
            feed_forward_length,
            expert_count,
            experts_per_token,
            expert_feed_forward_length,
            normalize_expert_weights,
            expert_weights_scale,
        };
        let rope_inverse = math::rope_inverse_frequencies(rope_dimension_count, rope_base)?;
        let mut used = BTreeSet::new();
        check_shape(
            &source,
            &mut used,
            "token_embd.weight".into(),
            &[embedding_length, vocab_size],
        )?;
        let output_norm = norm_vector(
            &source,
            &mut used,
            "output_norm.weight".into(),
            embedding_length,
        )?;
        let output_tensor = if source.tensors.contains_key("output.weight") {
            check_shape(
                &source,
                &mut used,
                "output.weight".into(),
                &[embedding_length, vocab_size],
            )?
        } else {
            "token_embd.weight".into()
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer in 0..block_count {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}.weight");
            let attention_norm =
                norm_vector(&source, &mut used, name("attn_norm"), embedding_length)?;
            let query_norm = norm_vector(&source, &mut used, name("attn_q_norm"), head_dim)?;
            let key_norm = norm_vector(&source, &mut used, name("attn_k_norm"), head_dim)?;
            let feed_forward_norm =
                norm_vector(&source, &mut used, name("ffn_norm"), embedding_length)?;
            let query = check_shape(
                &source,
                &mut used,
                name("attn_q"),
                &[embedding_length, query_width],
            )?;
            let key = check_shape(
                &source,
                &mut used,
                name("attn_k"),
                &[embedding_length, kv_width],
            )?;
            let value = check_shape(
                &source,
                &mut used,
                name("attn_v"),
                &[embedding_length, kv_width],
            )?;
            let attention_output = check_shape(
                &source,
                &mut used,
                name("attn_output"),
                &[query_width, embedding_length],
            )?;
            let feed_forward = if is_moe {
                FeedForward::Moe {
                    router: check_shape(
                        &source,
                        &mut used,
                        name("ffn_gate_inp"),
                        &[embedding_length, expert_count],
                    )?,
                    gate: check_shape(
                        &source,
                        &mut used,
                        name("ffn_gate_exps"),
                        &[embedding_length, expert_feed_forward_length, expert_count],
                    )?,
                    up: check_shape(
                        &source,
                        &mut used,
                        name("ffn_up_exps"),
                        &[embedding_length, expert_feed_forward_length, expert_count],
                    )?,
                    down: check_shape(
                        &source,
                        &mut used,
                        name("ffn_down_exps"),
                        &[expert_feed_forward_length, embedding_length, expert_count],
                    )?,
                }
            } else {
                FeedForward::Dense {
                    gate: check_shape(
                        &source,
                        &mut used,
                        name("ffn_gate"),
                        &[embedding_length, feed_forward_length],
                    )?,
                    up: check_shape(
                        &source,
                        &mut used,
                        name("ffn_up"),
                        &[embedding_length, feed_forward_length],
                    )?,
                    down: check_shape(
                        &source,
                        &mut used,
                        name("ffn_down"),
                        &[feed_forward_length, embedding_length],
                    )?,
                }
            };
            layers.push(Layer {
                attention_norm,
                query_norm,
                key_norm,
                feed_forward_norm,
                query,
                key,
                value,
                attention_output,
                feed_forward,
            });
        }
        if let Some(name) = source.tensors.keys().find(|name| !used.contains(*name)) {
            bail!(
                "independent Forge CPU engine does not support extra tensor {name:?}; bias, adapters, shared experts and extended architectures cannot be silently ignored"
            );
        }
        Ok(Self {
            source,
            executor,
            qwen38: None,
            config,
            layers,
            output_norm,
            output_tensor,
            rope_inverse,
            owner: Arc::new(()),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn executor_name(&self) -> &'static str {
        self.executor.name()
    }
    pub fn qwen38_config(&self) -> Option<&crate::qwen38::Architecture> {
        self.qwen38.as_ref().map(Qwen38::architecture)
    }
    pub fn context(&self) -> usize {
        self.config.context_length
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn new_sequence(&self) -> Sequence {
        Sequence {
            owner: Arc::clone(&self.owner),
            position: 0,
            kv_width: self.config.head_count_kv * self.config.head_dim,
            layers: (0..self.config.block_count)
                .map(|_| Kv::default())
                .collect(),
            scratch: Scratch::new(&self.config),
            qwen38: self.qwen38.as_ref().map(Qwen38::new_state),
        }
    }

    fn validate_tokens(&self, sequence: &Sequence, tokens: &[u32]) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.owner, &sequence.owner),
            "sequence belongs to a different loaded model"
        );
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        ensure!(
            sequence
                .position
                .checked_add(tokens.len())
                .is_some_and(|end| end <= self.context()),
            "context full: {} existing + {} incoming tokens exceed {}",
            sequence.position,
            tokens.len(),
            self.context()
        );
        ensure!(
            tokens
                .iter()
                .all(|token| (*token as usize) < self.vocab_size()),
            "token ID is outside vocabulary of {} entries",
            self.vocab_size()
        );
        Ok(())
    }

    /// Decode one token and allocate the returned vectors. Use step_into to retain
    /// output buffers in a long-lived generation loop.
    pub fn step(&self, sequence: &mut Sequence, token: u32) -> Result<Step> {
        let mut out = Step::default();
        self.step_into(sequence, token, &mut out)?;
        Ok(out)
    }

    /// Decode one token into reusable caller-owned buffers. Any declared error
    /// restores the initialized KV prefix and position to their pre-call values.
    pub fn step_into(&self, sequence: &mut Sequence, token: u32, out: &mut Step) -> Result<()> {
        self.validate_tokens(sequence, &[token])?;
        if let Some(model) = &self.qwen38 {
            let state = sequence.qwen38.as_mut().context("missing Qwen3.8 state")?;
            model.step(state, token, true, true, out)?;
            sequence.position = state.position();
            return Ok(());
        }
        let start = sequence.position;
        let result = self
            .forward_token(sequence, token, true)
            .and_then(|()| self.project(sequence, out));
        if result.is_err() {
            sequence.rewind(start);
        }
        result
    }

    /// Process a prompt without projecting every intermediate token to the full
    /// vocabulary. All token IDs and the complete context budget are checked first.
    pub fn prefill(&self, sequence: &mut Sequence, tokens: &[u32]) -> Result<Step> {
        let mut out = Step::default();
        self.prefill_into(sequence, tokens, &mut out)?;
        Ok(out)
    }

    pub fn prefill_into(
        &self,
        sequence: &mut Sequence,
        tokens: &[u32],
        out: &mut Step,
    ) -> Result<()> {
        self.validate_tokens(sequence, tokens)?;
        if let Some(model) = &self.qwen38 {
            let state = sequence.qwen38.as_mut().context("missing Qwen3.8 state")?;
            let checkpoint = state.checkpoint();
            for (index, &token) in tokens.iter().enumerate() {
                if let Err(error) = model.step(state, token, index + 1 == tokens.len(), false, out)
                {
                    state.restore(checkpoint);
                    return Err(error);
                }
            }
            sequence.position = state.position();
            return Ok(());
        }
        let start = sequence.position;
        let result = (|| {
            for &token in tokens {
                self.forward_token(sequence, token, false)?;
            }
            self.project(sequence, out)
        })();
        if result.is_err() {
            sequence.rewind(start);
        }
        result
    }

    fn project(&self, sequence: &Sequence, out: &mut Step) -> Result<()> {
        out.hidden.resize(self.config.embedding_length, 0.0);
        out.logits.resize(self.vocab_size(), 0.0);
        math::rms_norm(
            &sequence.scratch.residual,
            &self.output_norm,
            self.config.rms_epsilon,
            &mut out.hidden,
        )?;
        self.executor.matvec(
            &self.source.tensor(&self.output_tensor)?,
            &out.hidden,
            &mut out.logits,
        )?;
        math::finite(&out.logits, "language-model logits")
    }

    fn forward_token(
        &self,
        sequence: &mut Sequence,
        token: u32,
        admit_experts: bool,
    ) -> Result<()> {
        let c = &self.config;
        // Reserve only the next initialized row; an oversized requested context
        // does not eagerly allocate every possible request's entire KV capacity.
        for kv in &mut sequence.layers {
            kv.keys
                .try_reserve(sequence.kv_width)
                .context("reserve key cache row")?;
            kv.values
                .try_reserve(sequence.kv_width)
                .context("reserve value cache row")?;
        }
        let s = &mut sequence.scratch;
        self.source
            .tensor("token_embd.weight")?
            .read_row(token as usize, &mut s.residual)?;
        math::rope_angles(
            sequence.position,
            &self.rope_inverse,
            &mut s.rope_cos,
            &mut s.rope_sin,
        )?;
        for (layer, kv) in self.layers.iter().zip(sequence.layers.iter_mut()) {
            math::rms_norm(
                &s.residual,
                &layer.attention_norm,
                c.rms_epsilon,
                &mut s.normalized,
            )?;
            self.executor.matvec(
                &self.source.tensor(&layer.query)?,
                &s.normalized,
                &mut s.query,
            )?;
            self.executor
                .matvec(&self.source.tensor(&layer.key)?, &s.normalized, &mut s.key)?;
            self.executor.matvec(
                &self.source.tensor(&layer.value)?,
                &s.normalized,
                &mut s.value,
            )?;
            math::rms_norm_heads(&mut s.query, &layer.query_norm, c.rms_epsilon)?;
            math::rms_norm_heads(&mut s.key, &layer.key_norm, c.rms_epsilon)?;
            math::rope_heads(&mut s.query, c.head_dim, &s.rope_cos, &s.rope_sin)?;
            math::rope_heads(&mut s.key, c.head_dim, &s.rope_cos, &s.rope_sin)?;
            kv.keys.extend_from_slice(&s.key);
            kv.values.extend_from_slice(&s.value);
            math::attention(
                &s.query,
                &kv.keys,
                &kv.values,
                sequence.position + 1,
                c.head_count,
                c.head_count_kv,
                c.head_dim,
                &mut s.attention_scores,
                &mut s.attention,
            )?;
            self.executor.matvec(
                &self.source.tensor(&layer.attention_output)?,
                &s.attention,
                &mut s.projected,
            )?;
            math::residual_add(&mut s.residual, &s.projected)?;
            math::rms_norm(
                &s.residual,
                &layer.feed_forward_norm,
                c.rms_epsilon,
                &mut s.normalized,
            )?;
            match &layer.feed_forward {
                FeedForward::Dense { gate, up, down } => {
                    self.executor
                        .matvec(&self.source.tensor(gate)?, &s.normalized, &mut s.gate)?;
                    self.executor
                        .matvec(&self.source.tensor(up)?, &s.normalized, &mut s.up)?;
                    math::swiglu(&mut s.gate, &s.up)?;
                    self.executor
                        .matvec(&self.source.tensor(down)?, &s.gate, &mut s.projected)?;
                }
                FeedForward::Moe {
                    router,
                    gate,
                    up,
                    down,
                } => {
                    self.executor.matvec(
                        &self.source.tensor(router)?,
                        &s.normalized,
                        &mut s.router,
                    )?;
                    math::route_experts(
                        &s.router,
                        c.experts_per_token,
                        c.normalize_expert_weights,
                        c.expert_weights_scale,
                        &mut s.selected,
                        &mut s.router_probabilities,
                    )?;

                    s.projected.fill(0.0);
                    for &(expert, weight) in &s.selected {
                        let gate_lease = if admit_experts {
                            self.source.expert_cached(gate, expert)?
                        } else {
                            self.source.expert_cached_for_prefill(gate, expert)?
                        };
                        self.executor
                            .matvec(&gate_lease.tensor(), &s.normalized, &mut s.gate)?;
                        drop(gate_lease);
                        let up_lease = if admit_experts {
                            self.source.expert_cached(up, expert)?
                        } else {
                            self.source.expert_cached_for_prefill(up, expert)?
                        };
                        self.executor
                            .matvec(&up_lease.tensor(), &s.normalized, &mut s.up)?;
                        drop(up_lease);
                        math::swiglu(&mut s.gate, &s.up)?;
                        let down_lease = if admit_experts {
                            self.source.expert_cached(down, expert)?
                        } else {
                            self.source.expert_cached_for_prefill(down, expert)?
                        };
                        self.executor.matvec(
                            &down_lease.tensor(),
                            &s.gate,
                            &mut s.expert_output,
                        )?;
                        drop(down_lease);
                        for (sum, contribution) in s.projected.iter_mut().zip(&s.expert_output) {
                            *sum += weight * contribution;
                        }
                    }
                }
            }
            math::residual_add(&mut s.residual, &s.projected)?;
        }
        sequence.position += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn source(moe: bool) -> Gguf {
        let name = if moe {
            "tiny-qwen3moe.gguf"
        } else {
            "tiny-qwen3.gguf"
        };
        Gguf::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures")
                .join(name),
        )
        .unwrap()
    }

    fn loaded(moe: bool, context: usize) -> Model {
        Model::load(Arc::new(source(moe)), context).unwrap()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-5, "{x} != {y}");
        }
    }

    #[test]
    fn prefill_matches_repeated_decode_for_dense_and_moe() {
        for moe in [false, true] {
            let m = loaded(moe, 16);
            let mut sequential = m.new_sequence();
            let mut batched = m.new_sequence();
            let prompt = [3, 8, 19, 77];
            let mut last = Step::default();
            for token in prompt {
                m.step_into(&mut sequential, token, &mut last).unwrap();
            }
            let prefill = m.prefill(&mut batched, &prompt).unwrap();
            close(&prefill.logits, &last.logits);
            close(&prefill.hidden, &last.hidden);
            assert_eq!(sequential.kv(0), batched.kv(0));
            assert_eq!(batched.position(), prompt.len());
        }
    }

    #[test]
    fn interleaved_requests_are_isolated_and_reset_replays_exactly() {
        let m = loaded(true, 16);
        let mut a = m.new_sequence();
        let mut b = m.new_sequence();
        let mut reference = m.new_sequence();
        m.step(&mut a, 2).unwrap();
        m.step(&mut b, 99).unwrap();
        m.step(&mut b, 101).unwrap();
        let a_result = m.step(&mut a, 3).unwrap();
        let expected = m.prefill(&mut reference, &[2, 3]).unwrap();
        assert_eq!(a_result.logits, expected.logits);
        assert_eq!(a.kv(0), reference.kv(0));
        assert_ne!(a.kv(0), b.kv(0));
        a.reset();
        assert_eq!(a.position(), 0);
        assert!(a.kv(0).unwrap().0.is_empty());
        assert_eq!(m.prefill(&mut a, &[2, 3]).unwrap().logits, expected.logits);
    }

    #[test]
    fn invalid_token_context_and_wrong_owner_do_not_change_state() {
        let m = loaded(false, 3);
        let other = loaded(false, 3);
        let mut state = m.new_sequence();
        m.step(&mut state, 7).unwrap();
        let original_keys = state.kv(0).unwrap().0.to_vec();
        assert!(m.prefill(&mut state, &[1, 2, 3]).is_err());
        assert!(m.prefill(&mut state, &[1, m.vocab_size() as u32]).is_err());
        assert!(m.prefill(&mut state, &[]).is_err());
        assert!(other.step(&mut state, 8).is_err());
        assert_eq!(state.position(), 1);
        assert_eq!(state.kv(0).unwrap().0, original_keys);
        m.prefill(&mut state, &[8, 9]).unwrap();
        assert!(m.step(&mut state, 10).is_err());
        assert_eq!(state.position(), 3);
    }

    #[test]
    fn incomplete_qwen38_and_extended_features_are_rejected_explicitly() {
        let mut g = source(true);
        g.meta.insert(
            "general.architecture".into(),
            Value::String("qwen4exp".into()),
        );
        let error = Model::load(Arc::new(g), 16).unwrap_err().to_string();
        assert!(error.contains("qwen4exp") || error.contains("required"));
        let mut g = source(false);
        g.meta
            .insert("qwen3.attention.sliding_window".into(), Value::U32(16));
        assert!(
            Model::load(Arc::new(g), 16)
                .unwrap_err()
                .to_string()
                .contains("sliding_window")
        );
        let mut g = source(false);
        g.meta.insert(
            "qwen3.rope.scaling.type".into(),
            Value::String("yarn".into()),
        );
        assert!(
            Model::load(Arc::new(g), 16)
                .unwrap_err()
                .to_string()
                .contains("unscaled RoPE")
        );
        let mut g = source(false);
        g.meta
            .insert("qwen3.rope.dimension_count".into(), Value::U32(16));
        assert!(
            Model::load(Arc::new(g), 16)
                .unwrap_err()
                .to_string()
                .contains("partial RoPE")
        );
        for moe in [false, true] {
            let arch = if moe { "qwen3moe" } else { "qwen3" };
            for suffix in ["rope.freq_base", "attention.layer_norm_rms_epsilon"] {
                let key = format!("{arch}.{suffix}");
                let mut g = source(moe);
                assert!(g.meta.remove(&key).is_some());
                let error = Model::load(Arc::new(g), 16).unwrap_err().to_string();
                assert!(error.contains(&key) && error.contains("required model metadata"));
                for invalid in [0.0, -1.0, f32::INFINITY, f32::NAN] {
                    let mut g = source(moe);
                    g.meta.insert(key.clone(), Value::F32(invalid));
                    let error = Model::load(Arc::new(g), 16).unwrap_err().to_string();
                    assert!(error.contains(&key) && error.contains("finite positive"));
                }
            }
            let key = format!("{arch}.rope.scale_linear");
            let mut g = source(moe);
            g.meta.insert(key.clone(), Value::F32(2.0));
            assert!(
                Model::load(Arc::new(g), 16)
                    .unwrap_err()
                    .to_string()
                    .contains("rope.scale_linear")
            );
            let mut g = source(moe);
            g.meta.insert(key, Value::F32(1.0));
            assert!(Model::load(Arc::new(g), 16).is_ok());
        }
    }

    #[test]
    fn unknown_tensors_and_invalid_router_geometry_are_not_ignored() {
        let mut g = source(false);
        let extra = g.tensors["output_norm.weight"].clone();
        g.tensors.insert("blk.0.attn_q.bias".into(), extra);
        assert!(
            Model::load(Arc::new(g), 16)
                .unwrap_err()
                .to_string()
                .contains("extra tensor")
        );
        let mut g = source(true);
        g.meta
            .insert("qwen3moe.expert_used_count".into(), Value::U32(5));
        assert!(Model::load(Arc::new(g), 16).is_err());
        assert!(Model::load(Arc::new(source(false)), 513).is_err());
    }

    #[test]
    fn projection_failure_rolls_back_appended_kv_for_step_and_prefill() {
        use std::io::Write;
        struct Temporary(PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let g = source(false);
        let mut bytes = std::fs::read(&g.path).unwrap();
        let start = g.tensors["output.weight"].file_offset;
        bytes[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temporary = Temporary(std::env::temp_dir().join(format!(
            "forge-projection-error-{}-{stamp}.gguf",
            std::process::id()
        )));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary.0)
            .unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        let m = Model::load(Arc::new(Gguf::open(&temporary.0).unwrap()), 16).unwrap();
        let mut state = m.new_sequence();
        // The decoder is sound; only the final vocabulary projection is corrupt.
        m.forward_token(&mut state, 7, true).unwrap();
        let prefix = state.kv(0).unwrap().0.to_vec();
        assert!(m.step(&mut state, 8).is_err());
        assert_eq!(state.position(), 1);
        assert_eq!(state.kv(0).unwrap().0, prefix);
        assert!(m.prefill(&mut state, &[8, 9, 10]).is_err());
        assert_eq!(state.position(), 1);
        assert_eq!(state.kv(0).unwrap().0, prefix);
    }
}
