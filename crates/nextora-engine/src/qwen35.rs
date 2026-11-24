//! Independent text-only Qwen3.5 / Qwen3.8 dense execution for GGUF `qwen35`.
//! Public HF equations and GGUF conversion layouts are references, not runtime
//! dependencies. GGUF has already folded +1 into zero-centered RMSNorm scales
//! and tiled DeltaNet value heads. MTP tensors are checked but never executed.
//! References: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
//! https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
//! https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py
use crate::{
    gguf::{Gguf, Value},
    math,
    model::{Config, MatvecExecutor, Step, tensor_dimensions},
};
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Debug, Serialize)]
pub struct Architecture {
    pub full_attention_interval: usize,
    pub recurrent_key_heads: usize,
    pub recurrent_value_heads: usize,
    pub recurrent_key_width: usize,
    pub recurrent_value_width: usize,
    pub recurrent_inner_width: usize,
    pub recurrent_convolution: usize,
    pub mtp_layers_validated_but_unused: usize,
    pub text_only: bool,
}
#[derive(Debug)]
pub(crate) struct Attention {
    pub(crate) query: String,
    pub(crate) key: String,
    pub(crate) value: String,
    pub(crate) output: String,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) key_norm: Vec<f32>,
}
#[derive(Debug)]
pub(crate) struct Delta {
    pub(crate) qkv: String,
    pub(crate) gate: String,
    pub(crate) beta: String,
    pub(crate) alpha: String,
    pub(crate) output: String,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) decay: Vec<f32>,
    pub(crate) convolution: Vec<f32>,
    pub(crate) norm: Vec<f32>,
}
#[derive(Debug)]
pub(crate) enum Mixer {
    Attention(Attention),
    Delta(Delta),
}
#[derive(Debug)]
pub(crate) struct Layer {
    pub(crate) input_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) mixer: Mixer,
    pub(crate) gate: String,
    pub(crate) up: String,
    pub(crate) down: String,
}
#[derive(Debug)]
pub struct Qwen35 {
    pub(crate) source: Arc<Gguf>,
    pub(crate) executor: Arc<dyn MatvecExecutor>,
    pub(crate) common: Config,
    pub(crate) architecture: Architecture,
    pub(crate) layers: Vec<Layer>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) output: String,
    pub(crate) frequencies: Vec<f32>,
    owner: Arc<()>,
}
#[derive(Debug, Clone)]
struct Recurrent {
    convolution: Vec<f32>,
    matrix: Vec<f32>,
}
#[derive(Debug)]
enum LayerState {
    Attention {
        key: Vec<f32>,
        value: Vec<f32>,
    },
    Delta {
        current: Recurrent,
        pending: Recurrent,
    },
}
#[derive(Debug)]
struct Work {
    residual: Vec<f32>,
    input: Vec<f32>,
    update: Vec<f32>,
    qg: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attention: Vec<f32>,
    scores: Vec<f32>,
    qkv: Vec<f32>,
    activated: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    recurrent_out: Vec<f32>,
    head: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}
#[derive(Debug)]
pub struct Qwen35State {
    owner: Arc<()>,
    position: usize,
    kv_width: usize,
    layers: Vec<LayerState>,
    work: Work,
}
#[derive(Debug)]
pub struct Checkpoint {
    position: usize,
    recurrent: Vec<Option<Recurrent>>,
}
impl Qwen35State {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            match layer {
                LayerState::Attention { key, value } => {
                    key.clear();
                    value.clear();
                }
                LayerState::Delta { current, pending } => {
                    for state in [current, pending] {
                        state.matrix.fill(0.0);
                        state.convolution.fill(0.0);
                    }
                }
            }
        }
    }
    pub fn kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match self.layers.get(layer)? {
            LayerState::Attention { key, value } => Some((key, value)),
            _ => None,
        }
    }
    pub fn recurrent(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match self.layers.get(layer)? {
            LayerState::Delta { current, .. } => Some((&current.convolution, &current.matrix)),
            _ => None,
        }
    }
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            position: self.position,
            recurrent: self
                .layers
                .iter()
                .map(|state| match state {
                    LayerState::Delta { current, .. } => Some(current.clone()),
                    _ => None,
                })
                .collect(),
        }
    }
    pub fn restore(&mut self, checkpoint: Checkpoint) {
        self.position = checkpoint.position;
        for (state, saved) in self.layers.iter_mut().zip(checkpoint.recurrent) {
            if let (LayerState::Delta { current, .. }, Some(saved)) = (state, saved) {
                *current = saved;
            }
        }
        self.truncate_attention();
    }
    fn truncate_attention(&mut self) {
        for state in &mut self.layers {
            if let LayerState::Attention { key, value } = state {
                key.truncate(self.position * self.kv_width);
                value.truncate(self.position * self.kv_width);
            }
        }
    }
}
fn number(g: &Gguf, key: &str) -> Result<usize> {
    usize::try_from(
        g.meta
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("{key}: required nonnegative integer"))?,
    )
    .context("model integer exceeds address space")
}
fn optional(g: &Gguf, key: &str, default: usize) -> Result<usize> {
    if g.meta.contains_key(key) {
        number(g, key)
    } else {
        Ok(default)
    }
}
fn positive(g: &Gguf, key: &str) -> Result<f32> {
    g.meta
        .get(key)
        .and_then(Value::as_f32)
        .filter(|v| v.is_finite() && *v > 0.0)
        .with_context(|| format!("{key}: required finite positive number"))
}
fn product(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= 32 * 1024 * 1024)
        .with_context(|| format!("{label}: excessive geometry"))
}
fn shape(g: &Gguf, used: &mut BTreeSet<String>, name: String, dims: &[usize]) -> Result<String> {
    let actual = tensor_dimensions(g, &name)?;
    ensure!(
        actual.len() == dims.len() && actual.iter().zip(dims).all(|(a, b)| *a == *b as u64),
        "{name}: shape {actual:?}, expected {dims:?}"
    );
    used.insert(name.clone());
    Ok(name)
}
fn vector(g: &Gguf, used: &mut BTreeSet<String>, name: String, width: usize) -> Result<Vec<f32>> {
    shape(g, used, name.clone(), &[width])?;
    if g.headers_only() {
        return Ok(Vec::new());
    }
    let mut out = vec![0.0; width];
    g.tensor(&name)?.read_row(0, &mut out)?;
    math::finite(&out, &name)?;
    Ok(out)
}
fn flat_vector(
    g: &Gguf,
    used: &mut BTreeSet<String>,
    name: String,
    dims: &[usize],
) -> Result<Vec<f32>> {
    shape(g, used, name.clone(), dims)?;
    if g.headers_only() {
        return Ok(Vec::new());
    }
    let t = g.tensor(&name)?;
    let mut out = vec![0.0; dims.iter().product()];
    for (row, values) in out.chunks_exact_mut(t.width()).enumerate() {
        t.read_row(row, values)?;
    }
    math::finite(&out, &name)?;
    Ok(out)
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        (1.0 + (-x).exp()).recip()
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

impl Qwen35 {
    pub fn load(
        source: Arc<Gguf>,
        context: usize,
        executor: Arc<dyn MatvecExecutor>,
    ) -> Result<Self> {
        ensure!(
            !source.headers_only(),
            "Qwen35 execution requires downloaded tensors"
        );
        Self::prepare(source, context, executor)
    }
    pub fn prepare(
        source: Arc<Gguf>,
        context: usize,
        executor: Arc<dyn MatvecExecutor>,
    ) -> Result<Self> {
        let g = &source;
        let key = |s: &str| format!("qwen35.{s}");
        let get = |s: &str| number(g, &key(s));
        ensure!(
            g.meta.get("general.architecture").and_then(Value::as_str) == Some("qwen35"),
            "Qwen35 requires qwen35 metadata"
        );
        ensure!(
            !g.meta.contains_key("adapter.type"),
            "Qwen35 adapter is not a base checkpoint"
        );
        if let Some(v) = g.meta.get("general.type") {
            ensure!(v.as_str() == Some("model"), "Qwen35 requires a model GGUF");
        }
        for suffix in [
            "expert_count",
            "expert_used_count",
            "attention.sliding_window",
            "hyper_connection.count",
            "attention.q_lora_rank",
            "attention.kv_lora_rank",
        ] {
            ensure!(
                optional(g, &key(suffix), 0)? == 0,
                "unsupported qwen35.{suffix}"
            );
        }
        for suffix in [
            "attention.layer_types",
            "attention.recurrent_layers",
            "rope.mrope_section",
            "attention.rope_pattern",
        ] {
            ensure!(
                !g.meta.contains_key(&key(suffix)),
                "unsupported qwen35.{suffix}"
            );
        }
        if let Some(v) = g.meta.get(&key("attention.causal")) {
            ensure!(
                v.as_bool() == Some(true) || v.as_u64() == Some(1),
                "Qwen35 requires causal attention"
            );
        }
        for suffix in [
            "attention.logit_softcapping",
            "final_logit_softcapping",
            "router.logit_softcapping",
        ] {
            if let Some(v) = g.meta.get(&key(suffix)) {
                ensure!(v.as_f32() == Some(0.0), "unsupported qwen35.{suffix}");
            }
        }
        for suffix in ["embedding_scale", "residual_scale", "attention.scale"] {
            if let Some(v) = g.meta.get(&key(suffix)) {
                ensure!(v.as_f32() == Some(1.0), "unsupported qwen35.{suffix}");
            }
        }
        let all_layers = get("block_count")?;
        let mtp = optional(g, &key("nextn_predict_layers"), 0)?;
        ensure!(
            mtp <= 4 && all_layers > mtp && all_layers <= 516,
            "invalid Qwen35 main/MTP layer count"
        );
        let blocks = all_layers - mtp;
        let n = get("embedding_length")?;
        let heads = get("attention.head_count")?;
        let kv = get("attention.head_count_kv")?;
        let hd = get("attention.key_length")?;
        let rot = get("rope.dimension_count")?;
        let ff = get("feed_forward_length")?;
        ensure!(
            (1..=65536).contains(&n)
                && heads > 0
                && kv > 0
                && heads % kv == 0
                && hd > 0
                && hd <= 1024,
            "invalid Qwen35 attention geometry"
        );
        ensure!(
            optional(g, &key("attention.value_length"), hd)? == hd
                && rot > 0
                && rot <= hd
                && rot % 2 == 0,
            "invalid Qwen35 rotary/value dimensions"
        );
        ensure!(
            (1..=1048576).contains(&ff),
            "invalid Qwen35 feed-forward width"
        );
        let sections = g
            .meta
            .get(&key("rope.dimension_sections"))
            .and_then(Value::as_array)
            .context("Qwen35 requires RoPE dimension sections")?;
        ensure!(
            sections.len() == 4
                && sections.iter().all(|v| v.as_u64().is_some())
                && sections
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u128)
                    .sum::<u128>()
                    == rot as u128 / 2,
            "invalid text MRoPE sections"
        );
        let train = get("context_length")?;
        ensure!(
            context > 0 && context <= train && context <= 1048576,
            "Qwen35 context {context} exceeds declared {train}"
        );
        let base = positive(g, &key("rope.freq_base"))?;
        let eps = positive(g, &key("attention.layer_norm_rms_epsilon"))?;
        let rope_scaling = crate::rope::from_gguf_metadata(&g.meta, "qwen35", train)?;
        let interval = optional(g, &key("full_attention_interval"), 4)?;
        ensure!(
            (1..=64).contains(&interval),
            "invalid Qwen35 attention interval"
        );
        let nk = get("ssm.group_count")?;
        let nv = get("ssm.time_step_rank")?;
        let dk = get("ssm.state_size")?;
        let inner = get("ssm.inner_size")?;
        let kernel = get("ssm.conv_kernel")?;
        ensure!(
            nk > 0
                && nv > 0
                && nv % nk == 0
                && inner > 0
                && inner % nv == 0
                && dk > 0
                && kernel >= 2
                && kernel <= 64,
            "invalid Qwen35 recurrent geometry"
        );
        let dv = inner / nv;
        let qk = product(nk, dk, "recurrent QK")?;
        let channels = qk
            .checked_mul(2)
            .and_then(|v| v.checked_add(inner))
            .context("recurrent channel overflow")?;
        product(inner, dk, "recurrent matrix")?;
        product(channels, kernel, "recurrent convolution")?;
        let embeddings = tensor_dimensions(g, "token_embd.weight")?;
        ensure!(
            embeddings.len() == 2 && embeddings[0] == n as u64,
            "invalid Qwen35 embedding shape"
        );
        let vocab = usize::try_from(embeddings[1])?;
        ensure!((1..=2000000).contains(&vocab), "invalid Qwen35 vocabulary");
        if let Some(v) = g.meta.get("tokenizer.ggml.tokens") {
            ensure!(
                v.as_array().is_some_and(|a| a.len() == vocab),
                "Qwen35 tokenizer/model vocabulary mismatch"
            );
        }
        let common = Config {
            architecture: "qwen35".into(),
            block_count: blocks,
            embedding_length: n,
            head_count: heads,
            head_count_kv: kv,
            head_dim: hd,
            rope_dimension_count: rot,
            rope_base: base,
            rope_scaling,
            rms_epsilon: eps,
            vocab_size: vocab,
            context_length: context,
            training_context_length: train,
            feed_forward_length: ff,
            expert_count: 0,
            experts_per_token: 0,
            expert_feed_forward_length: 0,
            normalize_expert_weights: false,
            expert_weights_scale: 1.0,
        };
        let architecture = Architecture {
            full_attention_interval: interval,
            recurrent_key_heads: nk,
            recurrent_value_heads: nv,
            recurrent_key_width: dk,
            recurrent_value_width: dv,
            recurrent_inner_width: inner,
            recurrent_convolution: kernel,
            mtp_layers_validated_but_unused: mtp,
            text_only: true,
        };
        let mut used = BTreeSet::new();
        shape(g, &mut used, "token_embd.weight".into(), &[n, vocab])?;
        let output = if g.tensors.contains_key("output.weight") {
            shape(g, &mut used, "output.weight".into(), &[n, vocab])?
        } else {
            "token_embd.weight".into()
        };
        let output_norm = vector(g, &mut used, "output_norm.weight".into(), n)?;
        let mut layers = Vec::with_capacity(blocks);
        for layer in 0..all_layers {
            let draft = layer >= blocks;
            let name = |s: &str| format!("blk.{layer}.{s}");
            let input_norm = if draft {
                if g.tensors.contains_key(&name("attn_norm.weight")) {
                    shape(g, &mut used, name("attn_norm.weight"), &[n])?;
                }
                Vec::new()
            } else {
                vector(g, &mut used, name("attn_norm.weight"), n)?
            };
            let ffn_norm = vector(g, &mut used, name("post_attention_norm.weight"), n)?;
            let gate = shape(g, &mut used, name("ffn_gate.weight"), &[n, ff])?;
            let up = shape(g, &mut used, name("ffn_up.weight"), &[n, ff])?;
            let down = shape(g, &mut used, name("ffn_down.weight"), &[ff, n])?;
            let mixer = if draft || (layer + 1) % interval == 0 {
                let qw = product(heads, hd, "attention query")?;
                let kw = product(kv, hd, "attention KV")?;
                Mixer::Attention(Attention {
                    query: shape(g, &mut used, name("attn_q.weight"), &[n, qw * 2])?,
                    key: shape(g, &mut used, name("attn_k.weight"), &[n, kw])?,
                    value: shape(g, &mut used, name("attn_v.weight"), &[n, kw])?,
                    output: shape(g, &mut used, name("attn_output.weight"), &[qw, n])?,
                    query_norm: vector(g, &mut used, name("attn_q_norm.weight"), hd)?,
                    key_norm: vector(g, &mut used, name("attn_k_norm.weight"), hd)?,
                })
            } else {
                let decay = vector(g, &mut used, name("ssm_a"), nv)?;
                ensure!(
                    decay.iter().all(|v| *v <= 0.0),
                    "Qwen35 ssm_a must contain converted nonpositive decay values"
                );
                Mixer::Delta(Delta {
                    qkv: shape(g, &mut used, name("attn_qkv.weight"), &[n, channels])?,
                    gate: shape(g, &mut used, name("attn_gate.weight"), &[n, inner])?,
                    beta: shape(g, &mut used, name("ssm_beta.weight"), &[n, nv])?,
                    alpha: shape(g, &mut used, name("ssm_alpha.weight"), &[n, nv])?,
                    output: shape(g, &mut used, name("ssm_out.weight"), &[inner, n])?,
                    dt_bias: vector(g, &mut used, name("ssm_dt.bias"), nv)?,
                    decay,
                    convolution: flat_vector(
                        g,
                        &mut used,
                        name("ssm_conv1d.weight"),
                        &[kernel, channels],
                    )?,
                    norm: vector(g, &mut used, name("ssm_norm.weight"), dv)?,
                })
            };
            if draft {
                shape(g, &mut used, name("nextn.eh_proj.weight"), &[n * 2, n])?;
                for suffix in [
                    "nextn.enorm.weight",
                    "nextn.hnorm.weight",
                    "nextn.shared_head_norm.weight",
                ] {
                    shape(g, &mut used, name(suffix), &[n])?;
                }
            } else {
                layers.push(Layer {
                    input_norm,
                    ffn_norm,
                    mixer,
                    gate,
                    up,
                    down,
                });
            }
        }
        for name in g.tensors.keys() {
            ensure!(
                used.contains(name),
                "unsupported Qwen35 extra tensor {name}"
            );
        }
        let frequencies = common.rope_frequencies()?;
        Ok(Self {
            source,
            executor,
            common,
            architecture,
            layers,
            output_norm,
            output,
            frequencies,
            owner: Arc::new(()),
        })
    }
    pub fn common_config(&self) -> &Config {
        &self.common
    }
    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }
    pub fn new_state(&self) -> Qwen35State {
        let c = &self.common;
        let a = &self.architecture;
        let n = c.embedding_length;
        let channels = 2 * a.recurrent_key_heads * a.recurrent_key_width + a.recurrent_inner_width;
        let layers = self
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Attention(_) => LayerState::Attention {
                    key: Vec::new(),
                    value: Vec::new(),
                },
                Mixer::Delta(_) => {
                    let current = Recurrent {
                        convolution: vec![0.0; (a.recurrent_convolution - 1) * channels],
                        matrix: vec![0.0; a.recurrent_inner_width * a.recurrent_key_width],
                    };
                    LayerState::Delta {
                        pending: current.clone(),
                        current,
                    }
                }
            })
            .collect();
        Qwen35State {
            owner: Arc::clone(&self.owner),
            position: 0,
            kv_width: c.head_count_kv * c.head_dim,
            layers,
            work: Work {
                residual: vec![0.0; n],
                input: vec![0.0; n],
                update: vec![0.0; n],
                qg: vec![0.0; 2 * c.head_count * c.head_dim],
                query: vec![0.0; c.head_count * c.head_dim],
                key: vec![0.0; c.head_count_kv * c.head_dim],
                value: vec![0.0; c.head_count_kv * c.head_dim],
                attention: vec![0.0; c.head_count * c.head_dim],
                scores: Vec::new(),
                qkv: vec![0.0; channels],
                activated: vec![0.0; channels],
                z: vec![0.0; a.recurrent_inner_width],
                beta: vec![0.0; a.recurrent_value_heads],
                alpha: vec![0.0; a.recurrent_value_heads],
                recurrent_out: vec![0.0; a.recurrent_inner_width],
                head: vec![0.0; a.recurrent_value_width],
                gate: vec![0.0; c.feed_forward_length],
                up: vec![0.0; c.feed_forward_length],
                cos: vec![0.0; c.rope_dimension_count / 2],
                sin: vec![0.0; c.rope_dimension_count / 2],
            },
        }
    }
    fn mv(&self, name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.executor
            .matvec(&self.source.tensor(name)?, input, output)
    }
    fn attention(
        &self,
        l: &Attention,
        w: &mut Work,
        keys: &mut Vec<f32>,
        values: &mut Vec<f32>,
        position: usize,
    ) -> Result<()> {
        let c = &self.common;
        let hd = c.head_dim;
        self.mv(&l.query, &w.input, &mut w.qg)?;
        for h in 0..c.head_count {
            w.query[h * hd..(h + 1) * hd].copy_from_slice(&w.qg[2 * h * hd..2 * h * hd + hd]);
        }
        self.mv(&l.key, &w.input, &mut w.key)?;
        self.mv(&l.value, &w.input, &mut w.value)?;
        math::rms_norm_heads(&mut w.query, &l.query_norm, c.rms_epsilon)?;
        math::rms_norm_heads(&mut w.key, &l.key_norm, c.rms_epsilon)?;
        math::rope_heads(&mut w.query, hd, &w.cos, &w.sin)?;
        math::rope_heads(&mut w.key, hd, &w.cos, &w.sin)?;
        keys.extend_from_slice(&w.key);
        values.extend_from_slice(&w.value);
        w.scores.resize(position + 1, 0.0);
        math::attention(
            &w.query,
            keys,
            values,
            position + 1,
            c.head_count,
            c.head_count_kv,
            hd,
            &mut w.scores,
            &mut w.attention,
        )?;
        for h in 0..c.head_count {
            for j in 0..hd {
                w.attention[h * hd + j] *= sigmoid(w.qg[2 * h * hd + hd + j]);
            }
        }
        self.mv(&l.output, &w.attention, &mut w.update)
    }
    fn delta(
        &self,
        l: &Delta,
        w: &mut Work,
        current: &Recurrent,
        pending: &mut Recurrent,
    ) -> Result<()> {
        let a = &self.architecture;
        let dk = a.recurrent_key_width;
        let dv = a.recurrent_value_width;
        let nk = a.recurrent_key_heads;
        let nv = a.recurrent_value_heads;
        let qk = nk * dk;
        let channels = w.qkv.len();
        let kernel = a.recurrent_convolution;
        self.mv(&l.qkv, &w.input, &mut w.qkv)?;
        self.mv(&l.gate, &w.input, &mut w.z)?;
        self.mv(&l.beta, &w.input, &mut w.beta)?;
        self.mv(&l.alpha, &w.input, &mut w.alpha)?;
        for channel in 0..channels {
            let mut value = w.qkv[channel] * l.convolution[channel * kernel + kernel - 1];
            for tap in 0..kernel - 1 {
                value += current.convolution[tap * channels + channel]
                    * l.convolution[channel * kernel + tap];
            }
            w.activated[channel] = silu(value);
        }
        if kernel > 2 {
            pending.convolution[..(kernel - 2) * channels]
                .copy_from_slice(&current.convolution[channels..]);
        }
        pending.convolution[(kernel - 2) * channels..].copy_from_slice(&w.qkv);
        for head in w.activated[..2 * qk].chunks_exact_mut(dk) {
            let inverse = (head.iter().map(|v| v * v).sum::<f32>() + 1e-6)
                .sqrt()
                .recip();
            for value in head {
                *value *= inverse;
            }
        }
        for h in 0..nv {
            // Converted GGUF V heads are tiled, not HF's grouped order.
            let kh = h % nk;
            let q = &w.activated[kh * dk..(kh + 1) * dk];
            let k = &w.activated[qk + kh * dk..qk + (kh + 1) * dk];
            let v = &w.activated[2 * qk + h * dv..2 * qk + (h + 1) * dv];
            let decay = (l.decay[h] * softplus(w.alpha[h] + l.dt_bias[h])).exp();
            let beta = sigmoid(w.beta[h]);
            for j in 0..dv {
                let offset = (h * dv + j) * dk;
                let old = &current.matrix[offset..offset + dk];
                let new = &mut pending.matrix[offset..offset + dk];
                let mut prediction = 0.0;
                for i in 0..dk {
                    new[i] = old[i] * decay;
                    prediction += new[i] * k[i];
                }
                let error = beta * (v[j] - prediction);
                for i in 0..dk {
                    new[i] += error * k[i];
                }
                w.head[j] = new.iter().zip(q).map(|(s, q)| s * q).sum::<f32>() / (dk as f32).sqrt();
            }
            let output = &mut w.recurrent_out[h * dv..(h + 1) * dv];
            math::rms_norm(&w.head, &l.norm, self.common.rms_epsilon, output)?;
            for (j, value) in output.iter_mut().enumerate() {
                *value *= silu(w.z[h * dv + j]);
            }
        }
        math::finite(&pending.matrix, "Qwen35 recurrent state")?;
        self.mv(&l.output, &w.recurrent_out, &mut w.update)
    }
    pub fn step(
        &self,
        state: &mut Qwen35State,
        token: u32,
        project: bool,
        out: &mut Step,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.owner, &state.owner),
            "Qwen35 state belongs to another model"
        );
        ensure!(
            state.position < self.common.context_length
                && (token as usize) < self.common.vocab_size,
            "invalid Qwen35 token/context"
        );
        for layer in &mut state.layers {
            if let LayerState::Attention { key, value } = layer {
                key.try_reserve(state.kv_width)?;
                value.try_reserve(state.kv_width)?;
            }
        }
        let result = (|| {
            let w = &mut state.work;
            self.source
                .tensor("token_embd.weight")?
                .read_row(token as usize, &mut w.residual)?;
            math::rope_angles_scaled(
                state.position,
                &self.frequencies,
                self.common.rope_attention_factor(),
                &mut w.cos,
                &mut w.sin,
            )?;
            for (layer, slot) in self.layers.iter().zip(&mut state.layers) {
                math::rms_norm(
                    &w.residual,
                    &layer.input_norm,
                    self.common.rms_epsilon,
                    &mut w.input,
                )?;
                match (&layer.mixer, slot) {
                    (Mixer::Attention(l), LayerState::Attention { key, value }) => {
                        self.attention(l, w, key, value, state.position)?
                    }
                    (Mixer::Delta(l), LayerState::Delta { current, pending }) => {
                        self.delta(l, w, current, pending)?
                    }
                    _ => bail!("Qwen35 layer state mismatch"),
                }
                math::residual_add(&mut w.residual, &w.update)?;
                math::rms_norm(
                    &w.residual,
                    &layer.ffn_norm,
                    self.common.rms_epsilon,
                    &mut w.input,
                )?;
                self.mv(&layer.gate, &w.input, &mut w.gate)?;
                self.mv(&layer.up, &w.input, &mut w.up)?;
                math::swiglu(&mut w.gate, &w.up)?;
                self.mv(&layer.down, &w.gate, &mut w.update)?;
                math::residual_add(&mut w.residual, &w.update)?;
            }
            if project {
                out.hidden.resize(self.common.embedding_length, 0.0);
                math::rms_norm(
                    &w.residual,
                    &self.output_norm,
                    self.common.rms_epsilon,
                    &mut out.hidden,
                )?;
                out.logits.resize(self.common.vocab_size, 0.0);
                self.mv(&self.output, &out.hidden, &mut out.logits)?;
                math::finite(&out.logits, "Qwen35 logits")?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                for layer in &mut state.layers {
                    if let LayerState::Delta { current, pending } = layer {
                        std::mem::swap(current, pending);
                    }
                }
                state.position += 1;
                Ok(())
            }
            Err(error) => {
                state.truncate_attention();
                Err(error)
            }
        }
    }
    pub fn prefill(&self, state: &mut Qwen35State, tokens: &[u32], out: &mut Step) -> Result<()> {
        ensure!(
            !tokens.is_empty()
                && state
                    .position
                    .checked_add(tokens.len())
                    .is_some_and(|end| end <= self.common.context_length)
                && tokens
                    .iter()
                    .all(|t| (*t as usize) < self.common.vocab_size),
            "invalid Qwen35 prompt/context"
        );
        let checkpoint = state.checkpoint();
        for (i, &token) in tokens.iter().enumerate() {
            if let Err(error) = self.step(state, token, i + 1 == tokens.len(), out) {
                state.restore(checkpoint);
                return Err(error);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CpuMatvec;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicBool, Ordering},
    };

    fn source(mtp: bool) -> Gguf {
        Gguf::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures")
                .join(if mtp {
                    "tiny-qwen35-mtp.gguf"
                } else {
                    "tiny-qwen35.gguf"
                }),
        )
        .unwrap()
    }
    fn loaded(mtp: bool) -> Qwen35 {
        Qwen35::load(Arc::new(source(mtp)), 32, Arc::new(CpuMatvec)).unwrap()
    }
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 3e-5, "element {i}: {a} != {e}");
        }
    }
    fn reference_array(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }

    #[test]
    fn mixed_delta_attention_dense_ffn_matches_independent_scalar_reference() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/tiny-qwen35.reference.json"
        ))
        .unwrap();
        for mtp in [false, true] {
            let model = loaded(mtp);
            let mut state = model.new_state();
            let mut out = Step::default();
            for (index, token) in reference["prompt"].as_array().unwrap().iter().enumerate() {
                model
                    .step(&mut state, token.as_u64().unwrap() as u32, true, &mut out)
                    .unwrap();
                close(
                    &out.hidden,
                    &reference_array(&reference["steps"][index]["hidden"]),
                );
                close(
                    &out.logits[..16],
                    &reference_array(&reference["steps"][index]["logits"]),
                );
            }
            for layer in 0..3 {
                let (convolution, matrix) = state.recurrent(layer).unwrap();
                close(
                    convolution,
                    &reference_array(&reference["recurrent"][layer]["conv"]),
                );
                close(
                    matrix,
                    &reference_array(&reference["recurrent"][layer]["matrix"]),
                );
            }
            let (keys, values) = state.kv(3).unwrap();
            close(keys, &reference_array(&reference["keys"]));
            close(values, &reference_array(&reference["values"]));
            assert_eq!(model.common.block_count, 4);
            assert_eq!(
                model.architecture.mtp_layers_validated_but_unused,
                usize::from(mtp)
            );
        }
    }

    #[test]
    fn mixed_prefill_reset_and_request_isolation_replay_identically() {
        let model = loaded(false);
        let mut sequential = model.new_state();
        let mut batched = model.new_state();
        let mut out = Step::default();
        for token in [3, 8, 19, 77] {
            model.step(&mut sequential, token, true, &mut out).unwrap();
        }
        let mut other = Step::default();
        model
            .prefill(&mut batched, &[3, 8, 19, 77], &mut other)
            .unwrap();
        assert_eq!(out.logits, other.logits);
        assert_eq!(sequential.recurrent(0), batched.recurrent(0));
        assert_eq!(sequential.kv(3), batched.kv(3));
        batched.reset();
        model
            .prefill(&mut batched, &[3, 8, 19, 77], &mut other)
            .unwrap();
        assert_eq!(out.logits, other.logits);
        let mut isolated = model.new_state();
        model
            .prefill(&mut isolated, &[99, 101], &mut other)
            .unwrap();
        assert_eq!(sequential.position(), 4);
        assert!(
            model
                .step(&mut loaded(false).new_state(), 3, true, &mut other)
                .is_err()
        );
    }

    #[derive(Debug)]
    struct FailProjection(AtomicBool);
    impl MatvecExecutor for FailProjection {
        fn matvec(
            &self,
            t: &crate::gguf::Tensor<'_>,
            input: &[f32],
            out: &mut [f32],
        ) -> Result<()> {
            ensure!(
                !(out.len() == 261 && self.0.load(Ordering::Relaxed)),
                "injected projection failure"
            );
            t.matvec_into(input, out)
        }
        fn name(&self) -> &'static str {
            "failure_test"
        }
    }
    #[test]
    fn failure_rolls_back_attention_and_all_recurrent_layers_for_step_and_prefill() {
        let executor = Arc::new(FailProjection(AtomicBool::new(false)));
        let model = Qwen35::load(Arc::new(source(false)), 16, executor.clone()).unwrap();
        let mut state = model.new_state();
        let mut out = Step::default();
        model.prefill(&mut state, &[3, 8], &mut out).unwrap();
        let saved: Vec<_> = (0..3)
            .map(|i| {
                let (c, m) = state.recurrent(i).unwrap();
                (c.to_vec(), m.to_vec())
            })
            .collect();
        let keys = state.kv(3).unwrap().0.to_vec();
        executor.0.store(true, Ordering::Relaxed);
        assert!(model.step(&mut state, 19, true, &mut out).is_err());
        assert!(model.prefill(&mut state, &[19, 77], &mut out).is_err());
        assert_eq!(state.position(), 2);
        assert_eq!(state.kv(3).unwrap().0, keys);
        for (i, (c, m)) in saved.iter().enumerate() {
            assert_eq!(state.recurrent(i).unwrap(), (c.as_slice(), m.as_slice()));
        }
        executor.0.store(false, Ordering::Relaxed);
        model.prefill(&mut state, &[19, 77], &mut out).unwrap();
        let reference = loaded(false);
        let mut rs = reference.new_state();
        let mut expected = Step::default();
        reference
            .prefill(&mut rs, &[3, 8, 19, 77], &mut expected)
            .unwrap();
        assert_eq!(out.logits, expected.logits);
    }

    #[test]
    fn headers_and_unsupported_semantics_are_checked_without_execution() {
        let g = source(true);
        let bytes = std::fs::read(&g.path).unwrap();
        let name = g.path.file_name().unwrap().to_str().unwrap();
        let header =
            crate::gguf::RemoteHeader::parse(name, &bytes[..g.data_offset], bytes.len() as u64)
                .unwrap();
        let header = Arc::new(Gguf::from_remote_headers(vec![header]).unwrap());
        let model = Qwen35::prepare(header.clone(), 512, Arc::new(CpuMatvec)).unwrap();
        assert_eq!(model.common.block_count, 4);
        assert!(model.layers[0].input_norm.is_empty());
        assert!(Qwen35::load(header, 512, Arc::new(CpuMatvec)).is_err());
        for (key, value) in [
            ("qwen35.rope.dimension_count", Value::U32(9)),
            ("qwen35.ssm.group_count", Value::U32(3)),
            ("qwen35.attention.head_count_kv", Value::U32(3)),
            ("qwen35.nextn_predict_layers", Value::U32(5)),
            ("qwen35.rope.scaling.type", Value::String("dynamic".into())),
        ] {
            let mut invalid = source(false);
            invalid.meta.insert(key.into(), value);
            assert!(
                Qwen35::load(Arc::new(invalid), 16, Arc::new(CpuMatvec)).is_err(),
                "{key}"
            );
        }
        let mut extra = source(false);
        let info = extra.tensors["output_norm.weight"].clone();
        extra.tensors.insert("unknown.weight".into(), info);
        assert!(
            Qwen35::load(Arc::new(extra), 16, Arc::new(CpuMatvec))
                .unwrap_err()
                .to_string()
                .contains("extra tensor")
        );
        assert!(Qwen35::load(Arc::new(source(false)), 513, Arc::new(CpuMatvec)).is_err());
    }

    #[derive(Debug)]
    struct ForceGpu(crate::vulkan::GpuMatvecExecutor);
    impl MatvecExecutor for ForceGpu {
        fn matvec(
            &self,
            t: &crate::gguf::Tensor<'_>,
            input: &[f32],
            out: &mut [f32],
        ) -> Result<()> {
            self.0.matvec_gpu(t, input, out)
        }
        fn name(&self) -> &'static str {
            "test_forced_native_gpu"
        }
    }
    #[test]
    #[ignore = "requires explicit Vulkan projections; recurrent/attention state stays on CPU"]
    fn software_vulkan_qwen35_projections_match_cpu_mixed_layer_outputs() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let executor = Arc::new(ForceGpu(
            crate::vulkan::GpuMatvecExecutor::new(crate::vulkan::DeviceOptions {
                allow_software: true,
                memory_budget_bytes: 64 * 1024 * 1024,
                ..Default::default()
            })
            .unwrap(),
        ));
        let source = Arc::new(source(false));
        let cpu = Qwen35::load(source.clone(), 16, Arc::new(CpuMatvec)).unwrap();
        let gpu = Qwen35::load(source, 16, executor.clone()).unwrap();
        let mut cs = cpu.new_state();
        let mut gs = gpu.new_state();
        let mut expected = Step::default();
        let mut actual = Step::default();
        for token in [3, 8, 19, 77] {
            cpu.step(&mut cs, token, true, &mut expected).unwrap();
            gpu.step(&mut gs, token, true, &mut actual).unwrap();
            close(&actual.hidden, &expected.hidden);
            close(&actual.logits, &expected.logits);
            for layer in 0..3 {
                close(
                    gs.recurrent(layer).unwrap().1,
                    cs.recurrent(layer).unwrap().1,
                );
            }
        }
        assert!(executor.0.stats()["gpu_projection_calls"].as_u64().unwrap() > 100);
    }
    #[test]
    #[ignore = "requires explicitly cached real GGUF header, never model weights"]
    fn real_27b_header_passes_qwen35_shape_mtp_and_quant_validation() {
        let manifest = std::env::var("NEXTORA_QWEN35_HEADER_MANIFEST")
            .expect("provide cached header manifest");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
        let bytes = std::fs::read(manifest["header_path"].as_str().unwrap()).unwrap();
        let parsed = crate::gguf::RemoteHeader::parse(
            manifest["filename"].as_str().unwrap(),
            &bytes,
            manifest["file_bytes"].as_u64().unwrap(),
        )
        .unwrap();
        let source = Arc::new(Gguf::from_remote_headers(vec![parsed]).unwrap());
        println!("pretokenizer={:?}", source.meta.get("tokenizer.ggml.pre"));
        let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&source).unwrap();
        let template = crate::protocol::ChatTemplate::new(&source).unwrap();
        let request = crate::protocol::ChatRequest::parse(
            &serde_json::json!({
                "model":"header-test",
                "messages":[{"role":"user","content":"Explain fn square(x: i32) -> i32."}],
                "max_tokens":16
            }),
            16,
        )
        .unwrap();
        let rendered = template.render(&request).unwrap();
        let tokens = tokenizer.encode(&rendered).unwrap();
        assert!(!tokens.is_empty() && tokens.iter().all(|token| *token < 248320));
        let model = Qwen35::prepare(source, 1, Arc::new(CpuMatvec)).unwrap();
        assert_eq!(model.common.block_count, 64);
        assert_eq!(model.layers.len(), 64);
        assert_eq!(model.architecture.mtp_layers_validated_but_unused, 1);
        assert_eq!(
            model
                .layers
                .iter()
                .filter(|l| matches!(l.mixer, Mixer::Attention(_)))
                .count(),
            16
        );
        assert_eq!(model.common.head_dim, 256);
        assert_eq!(model.common.rope_dimension_count, 64);
        assert_eq!(model.common.vocab_size, 248320);
        println!("{}", serde_json::to_string(&model.common).unwrap());
    }
}
