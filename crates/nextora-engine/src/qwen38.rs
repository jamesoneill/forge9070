//! Independent text execution for the GGUF `qwen4exp` architecture.
//!
//! Implements hyper-connection streams, gated attention/DeltaNet, sparse QSA,
//! sparse n-gram embedding lookup and routed/shared experts. Matrix projections
//! use an explicit executor; nonlinear operations and sequence state are host
//! operations on this path. This is not a fully device-resident implementation.
//! Published format/model descriptions and the existing engine are validation
//! references only; this module does not import or call that engine.

use crate::{
    gguf::{Gguf, Value},
    math,
    model::{Config, MatvecExecutor, Step},
};
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::{collections::BTreeSet, sync::Arc};

const POOL: usize = 4;
const HASH_HEADS: usize = 16;
const HISTORY: usize = 9;

#[derive(Debug, Clone, Serialize)]
pub struct Architecture {
    pub hyper_streams: usize,
    pub hyper_rank: usize,
    pub full_attention_interval: usize,
    pub recurrent_key_heads: usize,
    pub recurrent_value_heads: usize,
    pub recurrent_key_width: usize,
    pub recurrent_value_width: usize,
    pub recurrent_inner_width: usize,
    pub recurrent_convolution: usize,
    pub indexer_heads: usize,
    pub indexer_head_width: usize,
    pub indexer_token_budget: usize,
    pub embedding_width_per_hash: usize,
    pub shared_expert_width: usize,
    pub text_only: bool,
}

#[derive(Debug)]
pub(crate) struct Hc {
    pub(crate) norm: Vec<f32>,
    pub(crate) down: String,
    pub(crate) up: String,
    pub(crate) inject: Option<String>,
}
#[derive(Debug)]
pub(crate) struct Indexer {
    pub(crate) query: String,
    pub(crate) key: String,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) key_norm: Vec<f32>,
}
#[derive(Debug)]
pub(crate) struct Attn {
    pub(crate) query: String,
    pub(crate) key: String,
    pub(crate) value: String,
    pub(crate) output: String,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) key_norm: Vec<f32>,
    pub(crate) indexer: Option<Indexer>,
}
#[derive(Debug)]
pub(crate) enum Gates {
    Split { beta: String, alpha: String },
    Combined(String),
}
#[derive(Debug)]
pub(crate) struct Delta {
    pub(crate) qkv: String,
    pub(crate) gate: String,
    pub(crate) gates: Gates,
    pub(crate) output: String,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) decay: Vec<f32>,
    pub(crate) convolution: Vec<f32>,
    pub(crate) norm: Vec<f32>,
}
#[derive(Debug)]
pub(crate) enum Mixer {
    Attn(Attn),
    Delta(Delta),
}
#[derive(Debug)]
pub(crate) struct Experts {
    pub(crate) router: String,
    pub(crate) gate: String,
    pub(crate) up: String,
    pub(crate) down: String,
    pub(crate) shared: Option<(String, String, String, Option<String>)>,
}
#[derive(Debug)]
pub(crate) struct PleLayer {
    pub(crate) key: String,
    pub(crate) value: String,
    pub(crate) key_norm: Vec<f32>,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) convolution_norm: Vec<f32>,
    pub(crate) convolution: Vec<f32>,
}
#[derive(Debug)]
pub(crate) struct Layer {
    pub(crate) read_attn: Hc,
    pub(crate) read_ffn: Hc,
    pub(crate) mixer: Mixer,
    pub(crate) experts: Experts,
    pub(crate) ple: Option<PleLayer>,
}
#[derive(Debug)]
pub(crate) struct HashEmbedding {
    pub(crate) multipliers: [u64; 3],
    pub(crate) offsets: [u64; HASH_HEADS],
    pub(crate) vocabulary: [u64; HASH_HEADS],
    pub(crate) eos: i64,
}

#[derive(Debug)]
pub struct Qwen38 {
    pub(crate) source: Arc<Gguf>,
    pub(crate) executor: Arc<dyn MatvecExecutor>,
    pub(crate) common: Config,
    pub(crate) architecture: Architecture,
    pub(crate) layers: Vec<Layer>,
    pub(crate) output: Hc,
    pub(crate) hash: Option<HashEmbedding>,
    pub(crate) frequencies: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Recurrent {
    convolution: Vec<f32>,
    matrix: Vec<f32>,
}
#[derive(Debug)]
enum LayerState {
    Attn {
        key: Vec<f32>,
        value: Vec<f32>,
        indexer: Vec<f32>,
        pooled: Vec<f32>,
    },
    Delta {
        current: Recurrent,
        pending: Recurrent,
    },
}
#[derive(Debug, Default)]
struct Work {
    streams: Vec<f32>,
    stream_norm: Vec<f32>,
    low_rank: Vec<f32>,
    mixing: Vec<f32>,
    inject: Vec<f32>,
    input: Vec<f32>,
    update: Vec<f32>,
    qg: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    iq: Vec<f32>,
    ik: Vec<f32>,
    pooled: Vec<f32>,
    pool_norm: Vec<f32>,
    blocks: Vec<(usize, f32)>,
    cells: Vec<usize>,
    qkv: Vec<f32>,
    activated: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    ba: Vec<f32>,
    recurrent_out: Vec<f32>,
    head: Vec<f32>,
    router: Vec<f32>,
    selection: Vec<(usize, f32)>,
    probabilities: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    expert_out: Vec<f32>,
    embedding: Vec<f32>,
    ple_key: Vec<f32>,
    ple_query: Vec<f32>,
    ple_value: Vec<f32>,
    ple_gated: Vec<f32>,
    ple_norm: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    pool_cos: Vec<f32>,
    pool_sin: Vec<f32>,
}
impl Work {
    fn new(c: &Config, a: &Architecture) -> Self {
        let n = c.embedding_length;
        let streams = n * a.hyper_streams;
        let channels = a.recurrent_inner_width + 2 * a.recurrent_key_heads * a.recurrent_key_width;
        let ff = c.expert_feed_forward_length.max(a.shared_expert_width);
        Self {
            streams: vec![0.; streams],
            stream_norm: vec![0.; streams],
            low_rank: vec![0.; a.hyper_rank],
            mixing: vec![0.; streams],
            inject: vec![0.; a.hyper_streams],
            input: vec![0.; n],
            update: vec![0.; n],
            qg: vec![0.; c.head_count * c.head_dim * 2],
            query: vec![0.; c.head_count * c.head_dim],
            key: vec![0.; c.head_count_kv * c.head_dim],
            value: vec![0.; c.head_count_kv * c.head_dim],
            attn: vec![0.; c.head_count * c.head_dim],
            scores: Vec::new(),
            iq: vec![0.; a.indexer_heads * a.indexer_head_width],
            ik: vec![0.; a.indexer_head_width],
            pooled: vec![0.; a.indexer_head_width],
            pool_norm: vec![0.; a.indexer_head_width],
            blocks: Vec::new(),
            cells: Vec::new(),
            qkv: vec![0.; channels],
            activated: vec![0.; channels],
            z: vec![0.; a.recurrent_inner_width],
            beta: vec![0.; a.recurrent_value_heads],
            alpha: vec![0.; a.recurrent_value_heads],
            ba: vec![0.; a.recurrent_value_heads * 2],
            recurrent_out: vec![0.; a.recurrent_inner_width],
            head: vec![0.; a.recurrent_value_width],
            router: vec![0.; c.expert_count],
            selection: Vec::with_capacity(c.expert_count),
            probabilities: Vec::new(),
            ffn_gate: vec![0.; ff],
            ffn_up: vec![0.; ff],
            expert_out: vec![0.; n],
            embedding: vec![0.; n],
            ple_key: vec![0.; streams],
            ple_query: vec![0.; streams],
            ple_value: vec![0.; n],
            ple_gated: vec![0.; streams],
            ple_norm: vec![0.; streams],
            cos: vec![0.; c.rope_dimension_count / 2],
            sin: vec![0.; c.rope_dimension_count / 2],
            pool_cos: vec![0.; c.rope_dimension_count / 2],
            pool_sin: vec![0.; c.rope_dimension_count / 2],
        }
    }
}

#[derive(Debug)]
pub struct Qwen38State {
    position: usize,
    kv_width: usize,
    index_width: usize,
    layers: Vec<LayerState>,
    history: Vec<f32>,
    pending_history: Vec<f32>,
    previous: [i64; 2],
    work: Work,
}
#[derive(Debug)]
pub struct Checkpoint {
    position: usize,
    recurrent: Vec<Option<Recurrent>>,
    history: Vec<f32>,
    previous: [i64; 2],
}
impl Qwen38State {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) {
        self.position = 0;
        self.previous = [-1, -1];
        self.history.fill(0.);
        self.pending_history.fill(0.);
        for state in &mut self.layers {
            match state {
                LayerState::Attn {
                    key,
                    value,
                    indexer,
                    pooled,
                } => {
                    key.clear();
                    value.clear();
                    indexer.clear();
                    pooled.clear();
                }
                LayerState::Delta { current, pending } => {
                    current.matrix.fill(0.);
                    current.convolution.fill(0.);
                    pending.matrix.fill(0.);
                    pending.convolution.fill(0.);
                }
            }
        }
    }
    pub fn kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match self.layers.get(layer)? {
            LayerState::Attn { key, value, .. } => Some((key, value)),
            _ => None,
        }
    }
    pub fn recurrent(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match self.layers.get(layer)? {
            LayerState::Delta { current, .. } => Some((&current.convolution, &current.matrix)),
            _ => None,
        }
    }
    pub fn indexer_keys(&self, layer: usize) -> Option<&[f32]> {
        match self.layers.get(layer)? {
            LayerState::Attn { indexer, .. } => Some(indexer),
            _ => None,
        }
    }
    pub fn ple_history(&self) -> (&[f32], [i64; 2]) {
        (&self.history, self.previous)
    }
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            position: self.position,
            history: self.history.clone(),
            previous: self.previous,
            recurrent: self
                .layers
                .iter()
                .map(|s| match s {
                    LayerState::Delta { current, .. } => Some(current.clone()),
                    _ => None,
                })
                .collect(),
        }
    }
    pub fn restore(&mut self, checkpoint: Checkpoint) {
        self.position = checkpoint.position;
        self.history = checkpoint.history;
        self.previous = checkpoint.previous;
        for (state, saved) in self.layers.iter_mut().zip(checkpoint.recurrent) {
            if let (LayerState::Delta { current, .. }, Some(saved)) = (state, saved) {
                *current = saved;
            }
        }
        self.truncate_attention();
    }
    fn truncate_attention(&mut self) {
        for state in &mut self.layers {
            if let LayerState::Attn {
                key,
                value,
                indexer,
                pooled,
            } = state
            {
                key.truncate(self.position * self.kv_width);
                value.truncate(self.position * self.kv_width);
                indexer.truncate(self.position * self.index_width);
                pooled.truncate((self.position / POOL) * self.index_width);
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
fn opt(g: &Gguf, key: &str, default: usize) -> Result<usize> {
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
        .filter(|v| v.is_finite() && *v > 0.)
        .with_context(|| format!("{key}: required finite positive value"))
}
fn product(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= 32 * 1024 * 1024)
        .with_context(|| format!("{label}: invalid or excessive geometry"))
}
fn shape(g: &Gguf, used: &mut BTreeSet<String>, name: String, dims: &[usize]) -> Result<String> {
    let t = g.tensor(&name)?;
    ensure!(
        t.dimensions().len() == dims.len()
            && t.dimensions()
                .iter()
                .zip(dims)
                .all(|(a, b)| *a == *b as u64),
        "{name}: shape {:?}, expected {dims:?}",
        t.dimensions()
    );
    used.insert(name.clone());
    Ok(name)
}
fn single_row(g: &Gguf, used: &mut BTreeSet<String>, name: String, width: usize) -> Result<String> {
    let t = g.tensor(&name)?;
    ensure!(
        t.width() == width && t.rows() == 1 && t.dimensions().len() <= 2,
        "{name}: expected a vector or single-row matrix"
    );
    used.insert(name.clone());
    Ok(name)
}
fn vector(
    g: &Gguf,
    used: &mut BTreeSet<String>,
    name: String,
    elements: usize,
) -> Result<Vec<f32>> {
    let t = g.tensor(&name)?;
    ensure!(
        t.elements() == elements,
        "{name}: expected {elements} elements, got {}",
        t.elements()
    );
    let mut values = vec![0.; elements];
    for (row, out) in values.chunks_exact_mut(t.width()).enumerate() {
        t.read_row(row, out)?;
    }
    used.insert(name);
    Ok(values)
}
fn hc(
    g: &Gguf,
    used: &mut BTreeSet<String>,
    prefix: &str,
    n: usize,
    streams: usize,
    rank: usize,
    inject: bool,
) -> Result<Hc> {
    let width = product(n, streams, "hyper streams")?;
    Ok(Hc {
        norm: vector(g, used, format!("{prefix}norm.weight"), width)?,
        down: shape(g, used, format!("{prefix}down.weight"), &[width, rank])?,
        up: shape(g, used, format!("{prefix}up.weight"), &[rank, width])?,
        inject: if inject {
            Some(shape(
                g,
                used,
                format!("{prefix}inject.weight"),
                &[width, streams],
            )?)
        } else {
            None
        },
    })
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20. { x } else { x.exp().ln_1p() }
}
fn grouped_norm(
    input: &[f32],
    weights: &[f32],
    width: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<()> {
    ensure!(
        input.len() == weights.len() && input.len() == out.len() && input.len() % width == 0,
        "hyper norm dimensions"
    );
    for ((x, w), y) in input
        .chunks_exact(width)
        .zip(weights.chunks_exact(width))
        .zip(out.chunks_exact_mut(width))
    {
        math::rms_norm(x, w, eps, y)?;
    }
    Ok(())
}

impl Qwen38 {
    pub fn load(
        source: Arc<Gguf>,
        context: usize,
        executor: Arc<dyn MatvecExecutor>,
    ) -> Result<Self> {
        let g = &source;
        let key = |s: &str| format!("qwen4exp.{s}");
        let get = |s: &str| number(g, &key(s));
        let optional = |s: &str, d| opt(g, &key(s), d);
        ensure!(
            g.meta.get("general.architecture").and_then(Value::as_str) == Some("qwen4exp"),
            "Qwen3.8 requires qwen4exp metadata"
        );
        ensure!(
            optional("nextn_predict_layers", 0)? == 0,
            "independent Qwen3.8 path does not load MTP layers yet"
        );
        ensure!(
            optional("leading_dense_block_count", 0)? == 0,
            "Qwen3.8 HC path requires routed experts in every layer"
        );
        ensure!(
            !g.meta.contains_key("adapter.type"),
            "adapter GGUF is not a base Qwen3.8 checkpoint"
        );
        if let Some(v) = g.meta.get("general.type") {
            ensure!(v.as_str() == Some("model"), "Qwen3.8 requires a model GGUF");
        }
        if let Some(v) = g.meta.get(&key("attention.causal")) {
            ensure!(
                v.as_bool() == Some(true) || v.as_u64() == Some(1),
                "Qwen3.8 requires causal attention"
            );
        }
        for suffix in [
            "attention.use_sliding_window",
            "attention.layer_types",
            "attention.rope_pattern",
            "rope.mrope_section",
        ] {
            ensure!(
                !g.meta.contains_key(&key(suffix)),
                "unsupported qwen4exp.{suffix}"
            );
        }
        for suffix in [
            "attention.sliding_window",
            "expert_group_count",
            "expert_group_used_count",
            "attention.q_lora_rank",
            "attention.kv_lora_rank",
        ] {
            ensure!(optional(suffix, 0)? == 0, "unsupported qwen4exp.{suffix}");
        }
        if let Some(v) = g.meta.get(&key("rope.scaling.type")) {
            ensure!(
                v.as_str() == Some("none"),
                "Qwen3.8 scaled RoPE is unsupported"
            );
        }
        for suffix in [
            "rope.scaling.factor",
            "rope.freq_scale",
            "rope.scale_linear",
            "embedding_scale",
            "residual_scale",
            "attention.scale",
        ] {
            if let Some(v) = g.meta.get(&key(suffix)) {
                ensure!(v.as_f32() == Some(1.), "unsupported qwen4exp.{suffix}");
            }
        }
        for suffix in [
            "attention.logit_softcapping",
            "final_logit_softcapping",
            "router.logit_softcapping",
        ] {
            if let Some(v) = g.meta.get(&key(suffix)) {
                ensure!(v.as_f32() == Some(0.), "unsupported qwen4exp.{suffix}");
            }
        }
        ensure!(
            optional("expert_gating_func", 1)? <= 1,
            "only softmax Qwen3.8 routing is supported"
        );
        if let Some(v) = g.meta.get(&key("expert_weights_norm")) {
            ensure!(
                v.as_bool() == Some(true) || v.as_u64() == Some(1),
                "Qwen3.8 requires normalized selected expert weights"
            );
        }
        let block_count = get("block_count")?;
        let n = get("embedding_length")?;
        let heads = get("attention.head_count")?;
        let kv_heads = get("attention.head_count_kv")?;
        ensure!(
            (1..=512).contains(&block_count) && (1..=65536).contains(&n),
            "invalid Qwen3.8 model dimensions"
        );
        ensure!(
            heads > 0 && kv_heads > 0 && heads % kv_heads == 0,
            "invalid Qwen3.8 attention head geometry"
        );
        let hd = optional("attention.key_length", n / heads)?;
        let rot = optional("rope.dimension_count", hd)?;
        ensure!(
            hd > 0
                && rot > 0
                && rot % 2 == 0
                && rot <= hd
                && optional("attention.value_length", hd)? == hd,
            "invalid Qwen3.8 head/rotary dimensions"
        );
        product(heads, hd, "query width")?;
        product(kv_heads, hd, "KV width")?;
        let eps = positive(g, &key("attention.layer_norm_rms_epsilon"))?;
        let base = positive(g, &key("rope.freq_base"))?;
        let train = get("context_length")?;
        ensure!(
            context > 0 && context <= train && context <= 1_048_576,
            "context {context} outside Qwen3.8 declared context {train}"
        );
        if let Some(v) = g.meta.get(&key("rope.dimension_sections")) {
            let s = v
                .as_array()
                .context("rope.dimension_sections must be an array")?;
            ensure!(
                s.len() == 4
                    && s.iter()
                        .map(|x| x.as_u64())
                        .collect::<Option<Vec<_>>>()
                        .is_some_and(|a| a
                            .iter()
                            .try_fold(0u64, |sum, value| sum.checked_add(*value))
                            == Some(rot as u64 / 2)),
                "invalid text M-RoPE sections"
            );
        }
        let embeddings = g.tensor("token_embd.weight")?;
        ensure!(
            embeddings.dimensions().len() == 2 && embeddings.width() == n,
            "invalid token embeddings"
        );
        let vocabulary = embeddings.rows();
        ensure!(
            (1..=2_000_000).contains(&vocabulary),
            "invalid Qwen3.8 vocabulary size"
        );
        if let Some(v) = g.meta.get("tokenizer.ggml.tokens") {
            ensure!(
                v.as_array().is_some_and(|a| a.len() == vocabulary),
                "Qwen3.8 tokenizer/embedding vocabulary mismatch"
            );
        }
        let experts = get("expert_count")?;
        let active = get("expert_used_count")?;
        let ff = get("expert_feed_forward_length")?;
        ensure!(
            experts > 0
                && experts <= 512
                && active > 0
                && active <= experts
                && ff > 0
                && ff <= 1_048_576,
            "invalid Qwen3.8 expert geometry"
        );
        let scale = match g.meta.get(&key("expert_weights_scale")) {
            None => 1.,
            Some(v) => {
                let x = v.as_f32().context("invalid expert scale")?;
                ensure!(x >= 0., "negative expert scale");
                if x == 0. { 1. } else { x }
            }
        };
        let streams = get("hyper_connection.count")?;
        let rank = get("hyper_connection.low_rank")?;
        ensure!(
            (1..=64).contains(&streams) && rank > 0 && rank <= 65536,
            "Qwen3.8 requires bounded positive HC geometry"
        );
        let stream_width = product(n, streams, "HC stream width")?;
        let interval = optional("full_attention_interval", 4)?;
        ensure!(interval > 0, "full attention interval must be positive");
        let delta = interval > 1;
        let nk = if delta { get("ssm.group_count")? } else { 0 };
        let nv = if delta { get("ssm.time_step_rank")? } else { 0 };
        let dk = if delta { get("ssm.state_size")? } else { 0 };
        let inner = if delta { get("ssm.inner_size")? } else { 0 };
        let kernel = if delta { get("ssm.conv_kernel")? } else { 0 };
        ensure!(
            !delta
                || (nk > 0
                    && nv > 0
                    && nv % nk == 0
                    && dk > 0
                    && inner > 0
                    && inner % nv == 0
                    && (1..=64).contains(&kernel)),
            "invalid DeltaNet dimensions"
        );
        let dv = if delta { inner / nv } else { 0 };
        if delta {
            product(inner, dk, "recurrent state")?;
            product(
                kernel,
                inner + 2 * product(nk, dk, "key channels")?,
                "convolution history",
            )?;
        }
        let budget = optional("attention.indexer.top_k", 0)?;
        let ih = optional("attention.indexer.head_count", 0)?;
        let id = optional("attention.indexer.key_length", 0)?;
        ensure!(
            (budget == 0 && ih == 0 && id == 0)
                || (budget >= POOL && budget % POOL == 0 && ih > 0 && id >= rot),
            "invalid QSA pool/head/budget geometry"
        );
        product(ih, id, "QSA query width")?;
        if let Some(v) = g.meta.get(&key("attention.compress_ratios")) {
            let ratios = v
                .as_array()
                .context("attention.compress_ratios must be an array")?;
            ensure!(
                ratios.len() == block_count
                    && ratios.iter().enumerate().all(|(i, v)| v.as_u64()
                        == Some(if (i + 1) % interval == 0 && budget > 0 {
                            POOL as u64
                        } else {
                            0
                        })),
                "unsupported QSA compression ratios"
            );
        }
        let ple_width = optional("embedding_length_per_layer_input", 0)?;
        let shared = optional("expert_shared_feed_forward_length", 0)?;
        ensure!(shared <= 1_048_576, "shared expert width too large");
        let a = Architecture {
            hyper_streams: streams,
            hyper_rank: rank,
            full_attention_interval: interval,
            recurrent_key_heads: nk,
            recurrent_value_heads: nv,
            recurrent_key_width: dk,
            recurrent_value_width: dv,
            recurrent_inner_width: inner,
            recurrent_convolution: kernel,
            indexer_heads: ih,
            indexer_head_width: id,
            indexer_token_budget: budget,
            embedding_width_per_hash: ple_width,
            shared_expert_width: shared,
            text_only: true,
        };
        let common = Config {
            architecture: "qwen4exp".into(),
            block_count,
            embedding_length: n,
            head_count: heads,
            head_count_kv: kv_heads,
            head_dim: hd,
            rope_dimension_count: rot,
            rope_base: base,
            rms_epsilon: eps,
            vocab_size: vocabulary,
            context_length: context,
            training_context_length: train,
            feed_forward_length: 0,
            expert_count: experts,
            experts_per_token: active,
            expert_feed_forward_length: ff,
            normalize_expert_weights: true,
            expert_weights_scale: scale,
        };
        let mut used = BTreeSet::new();
        shape(g, &mut used, "token_embd.weight".into(), &[n, vocabulary])?;
        shape(g, &mut used, "output.weight".into(), &[n, vocabulary])?;
        let output = hc(g, &mut used, "output_hc_", n, streams, rank, false)?;
        let hash = if ple_width > 0 {
            ensure!(
                ple_width.checked_mul(HASH_HEADS) == Some(n)
                    && get("ple.ngram_size")? == 3
                    && get("ple.heads_per_ngram")? == 8
                    && get("ple.conv_kernel")? == 4,
                "unsupported PLE hash/convolution geometry"
            );
            let arr = |s: &str, count: usize| -> Result<Vec<u64>> {
                let values = g
                    .meta
                    .get(&key(s))
                    .and_then(Value::as_array)
                    .with_context(|| format!("missing {s}"))?;
                ensure!(values.len() == count, "invalid {s} length");
                values
                    .iter()
                    .map(|v| v.as_u64().with_context(|| format!("invalid {s} entry")))
                    .collect()
            };
            let multipliers: [u64; 3] = arr("ple.layer_multipliers", 3)?.try_into().unwrap();
            let offsets: [u64; HASH_HEADS] =
                arr("ple.head_offsets", HASH_HEADS)?.try_into().unwrap();
            let vocab: [u64; HASH_HEADS] =
                arr("ple.head_vocab_sizes", HASH_HEADS)?.try_into().unwrap();
            let table = g.tensor("per_layer_token_embd.weight")?;
            ensure!(
                table.dimensions().len() == 2 && table.width() == ple_width,
                "invalid PLE table dimensions"
            );
            for (&off, &len) in offsets.iter().zip(&vocab) {
                ensure!(
                    len > 0
                        && off
                            .checked_add(len)
                            .is_some_and(|end| end <= table.rows() as u64),
                    "PLE hash range outside table"
                );
            }
            used.insert("per_layer_token_embd.weight".into());
            let eos = get("ple.eos_token_id")?;
            ensure!(eos < vocabulary, "PLE EOS outside vocabulary");
            Some(HashEmbedding {
                multipliers,
                offsets,
                vocabulary: vocab,
                eos: eos as i64,
            })
        } else {
            None
        };
        let mut layers = Vec::with_capacity(block_count);
        let mut ple_layers = 0;
        let mut ple_ids = Vec::new();
        for layer in 0..block_count {
            let p = |s: &str| format!("blk.{layer}.{s}");
            let m = |s: &str| p(&format!("{s}.weight"));
            let read_attn = hc(g, &mut used, &p("hc_attn_"), n, streams, rank, true)?;
            let read_ffn = hc(g, &mut used, &p("hc_ffn_"), n, streams, rank, true)?;
            let mixer = if (layer + 1) % interval == 0 {
                let indexer = if budget > 0 {
                    Some(Indexer {
                        query: shape(g, &mut used, m("indexer.q_proj"), &[n, ih * id])?,
                        key: shape(g, &mut used, m("indexer.k_proj"), &[n, id])?,
                        query_norm: vector(g, &mut used, m("indexer.q_norm"), id)?,
                        key_norm: vector(g, &mut used, m("indexer.k_norm"), id)?,
                    })
                } else {
                    None
                };
                Mixer::Attn(Attn {
                    query: shape(g, &mut used, m("attn_q"), &[n, heads * hd * 2])?,
                    key: shape(g, &mut used, m("attn_k"), &[n, kv_heads * hd])?,
                    value: shape(g, &mut used, m("attn_v"), &[n, kv_heads * hd])?,
                    output: shape(g, &mut used, m("attn_output"), &[heads * hd, n])?,
                    query_norm: vector(g, &mut used, m("attn_q_norm"), hd)?,
                    key_norm: vector(g, &mut used, m("attn_k_norm"), hd)?,
                    indexer,
                })
            } else {
                let channels = inner + 2 * nk * dk;
                let gates = if g.tensors.contains_key(&m("ssm_ba")) {
                    Gates::Combined(shape(g, &mut used, m("ssm_ba"), &[n, 2 * nv])?)
                } else {
                    Gates::Split {
                        beta: shape(g, &mut used, m("ssm_beta"), &[n, nv])?,
                        alpha: shape(g, &mut used, m("ssm_alpha"), &[n, nv])?,
                    }
                };
                Mixer::Delta(Delta {
                    qkv: shape(g, &mut used, m("attn_qkv"), &[n, channels])?,
                    gate: shape(g, &mut used, m("attn_gate"), &[n, inner])?,
                    gates,
                    output: shape(g, &mut used, m("ssm_out"), &[inner, n])?,
                    dt_bias: vector(g, &mut used, p("ssm_dt.bias"), nv)?,
                    decay: vector(g, &mut used, p("ssm_a"), nv)?,
                    convolution: vector(g, &mut used, m("ssm_conv1d"), channels * kernel)?,
                    norm: vector(g, &mut used, m("ssm_norm"), dv)?,
                })
            };
            let shared_weights = if shared > 0 {
                Some((
                    shape(g, &mut used, m("ffn_gate_shexp"), &[n, shared])?,
                    shape(g, &mut used, m("ffn_up_shexp"), &[n, shared])?,
                    shape(g, &mut used, m("ffn_down_shexp"), &[shared, n])?,
                    if g.tensors.contains_key(&m("ffn_gate_inp_shexp")) {
                        Some(single_row(g, &mut used, m("ffn_gate_inp_shexp"), n)?)
                    } else {
                        None
                    },
                ))
            } else {
                None
            };
            let experts = Experts {
                router: shape(g, &mut used, m("ffn_gate_inp"), &[n, experts])?,
                gate: shape(g, &mut used, m("ffn_gate_exps"), &[n, ff, experts])?,
                up: shape(g, &mut used, m("ffn_up_exps"), &[n, ff, experts])?,
                down: shape(g, &mut used, m("ffn_down_exps"), &[ff, n, experts])?,
                shared: shared_weights,
            };
            let ple = if g.tensors.contains_key(&m("ple_key")) {
                ensure!(hash.is_some(), "PLE block requires hash metadata and table");
                ple_layers += 1;
                ple_ids.push(layer);
                Some(PleLayer {
                    key: shape(g, &mut used, m("ple_key"), &[n, stream_width])?,
                    value: shape(g, &mut used, m("ple_value"), &[n, n])?,
                    key_norm: vector(g, &mut used, m("ple_norm_key"), stream_width)?,
                    query_norm: vector(g, &mut used, m("ple_norm_query"), stream_width)?,
                    convolution_norm: vector(g, &mut used, m("ple_norm_conv"), stream_width)?,
                    convolution: vector(g, &mut used, m("ple_conv1d"), 4 * stream_width)?,
                })
            } else {
                None
            };
            layers.push(Layer {
                read_attn,
                read_ffn,
                mixer,
                experts,
                ple,
            });
        }
        ensure!(
            (hash.is_none() && ple_layers == 0) || (hash.is_some() && ple_layers == 1),
            "Qwen3.8 PLE requires exactly one injection layer; multiple PLE histories are unsupported"
        );
        if let Some(v) = g.meta.get(&key("ple.layers")) {
            let ids = v.as_array().context("ple.layers must be an array")?;
            ensure!(
                ids.len() == ple_ids.len()
                    && ids
                        .iter()
                        .zip(&ple_ids)
                        .all(|(v, id)| v.as_u64() == Some(*id as u64)),
                "PLE layer metadata and tensors disagree"
            );
        }
        if let Some(name) = g.tensors.keys().find(|name| !used.contains(*name)) {
            bail!("unsupported Qwen3.8 extra tensor {name:?}; no tensor may be silently ignored");
        }
        Ok(Self {
            source,
            executor,
            common,
            architecture: a,
            layers,
            output,
            hash,
            frequencies: math::rope_inverse_frequencies(rot, base)?,
        })
    }
    pub fn common_config(&self) -> &Config {
        &self.common
    }
    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }
    pub fn new_state(&self) -> Qwen38State {
        let a = &self.architecture;
        let c = &self.common;
        let channels = a.recurrent_inner_width + 2 * a.recurrent_key_heads * a.recurrent_key_width;
        let layers = self
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Attn(_) => LayerState::Attn {
                    key: Vec::new(),
                    value: Vec::new(),
                    indexer: Vec::new(),
                    pooled: Vec::new(),
                },
                Mixer::Delta(_) => {
                    let current = Recurrent {
                        convolution: vec![0.; (a.recurrent_convolution - 1) * channels],
                        matrix: vec![0.; a.recurrent_inner_width * a.recurrent_key_width],
                    };
                    LayerState::Delta {
                        pending: current.clone(),
                        current,
                    }
                }
            })
            .collect();
        let history = if self.hash.is_some() {
            vec![0.; HISTORY * c.embedding_length * a.hyper_streams]
        } else {
            Vec::new()
        };
        Qwen38State {
            position: 0,
            kv_width: c.head_count_kv * c.head_dim,
            index_width: a.indexer_head_width,
            layers,
            pending_history: history.clone(),
            history,
            previous: [-1, -1],
            work: Work::new(c, a),
        }
    }
    fn mv(&self, name: &str, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.executor.matvec(&self.source.tensor(name)?, x, y)
    }
    fn read_hc(&self, h: &Hc, w: &mut Work) -> Result<()> {
        let n = self.common.embedding_length;
        let hc = self.architecture.hyper_streams as f32;
        grouped_norm(
            &w.streams,
            &h.norm,
            n,
            self.common.rms_epsilon,
            &mut w.stream_norm,
        )?;
        self.mv(&h.down, &w.stream_norm, &mut w.low_rank)?;
        for x in &mut w.low_rank {
            *x = silu(*x / hc);
        }
        self.mv(&h.up, &w.low_rank, &mut w.mixing)?;
        w.input.fill(0.);
        for stream in 0..self.architecture.hyper_streams {
            for j in 0..n {
                let i = stream * n + j;
                w.input[j] += w.stream_norm[i] * sigmoid(w.mixing[i]) / hc;
            }
        }
        if let Some(name) = &h.inject {
            self.mv(name, &w.stream_norm, &mut w.inject)?;
        }
        math::finite(&w.input, "HC mixed input")
    }
    fn write_hc(&self, w: &mut Work) -> Result<()> {
        let n = self.common.embedding_length;
        let scale = self.architecture.hyper_streams as f32;
        for (stream, &inject) in w.streams.chunks_exact_mut(n).zip(&w.inject) {
            let gate = 2. * sigmoid(inject / scale);
            for (x, delta) in stream.iter_mut().zip(&w.update) {
                *x += gate * delta;
            }
        }
        math::finite(&w.streams, "HC residual streams")
    }
    /// Pure PLE lookup plan for device-resident execution. The caller advances
    /// previous-token state only after the token has completed successfully.
    pub(crate) fn ple_rows(
        &self,
        previous: [i64; 2],
        token: u32,
    ) -> Result<Option<[u64; HASH_HEADS]>> {
        ensure!(
            (token as usize) < self.common.vocab_size,
            "PLE token outside vocabulary"
        );
        let Some(hash) = &self.hash else {
            return Ok(None);
        };
        let mut context = [token as i64, hash.eos, hash.eos];
        let mut ended = false;
        for i in 0..2 {
            ended |= previous[i] < 0 || previous[i] == hash.eos;
            if !ended {
                context[i + 1] = previous[i];
            }
        }
        let mut value = (context[0] as u64).wrapping_mul(hash.multipliers[0]);
        let mut rows = [0; HASH_HEADS];
        for order in 1..3 {
            value ^= (context[order] as u64).wrapping_mul(hash.multipliers[order]);
            for head in (order - 1) * 8..order * 8 {
                rows[head] = value % hash.vocabulary[head] + hash.offsets[head];
            }
        }
        Ok(Some(rows))
    }

    pub(crate) fn hash_embedding(
        &self,
        previous: [i64; 2],
        token: u32,
        output: &mut [f32],
    ) -> Result<()> {
        let Some(rows) = self.ple_rows(previous, token)? else {
            return Ok(());
        };
        let width = self.architecture.embedding_width_per_hash;
        let table = self.source.tensor("per_layer_token_embd.weight")?;
        for (head, row) in rows.into_iter().enumerate() {
            table.read_row(row as usize, &mut output[head * width..(head + 1) * width])?;
        }
        Ok(())
    }
    fn ple(
        &self,
        layer: &PleLayer,
        w: &mut Work,
        history: &[f32],
        pending: &mut [f32],
    ) -> Result<()> {
        let n = self.common.embedding_length;
        let d = w.streams.len();
        let eps = self.common.rms_epsilon;
        self.mv(&layer.key, &w.embedding, &mut w.ple_key)?;
        // Normalize one stream in-place using the shared primitive's one-head path.
        for (x, weights) in w
            .ple_key
            .chunks_exact_mut(n)
            .zip(layer.key_norm.chunks_exact(n))
        {
            math::rms_norm_heads(x, weights, eps)?;
        }
        grouped_norm(&w.streams, &layer.query_norm, n, eps, &mut w.ple_query)?;
        self.mv(&layer.value, &w.embedding, &mut w.ple_value)?;
        for stream in 0..self.architecture.hyper_streams {
            let start = stream * n;
            let score = w.ple_key[start..start + n]
                .iter()
                .zip(&w.ple_query[start..start + n])
                .map(|(x, y)| x * y)
                .sum::<f32>()
                / (n as f32).sqrt();
            ensure!(score.is_finite(), "non-finite PLE gate score");
            let root = score.abs().max(1e-6).sqrt();
            let gate = sigmoid(if score > 0. {
                root
            } else if score < 0. {
                -root
            } else {
                0.
            });
            for j in 0..n {
                w.ple_gated[start + j] = w.ple_value[j] * gate;
            }
        }
        grouped_norm(
            &w.ple_gated,
            &layer.convolution_norm,
            n,
            eps,
            &mut w.ple_norm,
        )?;
        for channel in 0..d {
            let mut convolution = layer.convolution[channel * 4 + 3] * w.ple_norm[channel];
            for tap in 0..3 {
                convolution +=
                    layer.convolution[channel * 4 + tap] * history[tap * 3 * d + channel];
            }
            w.streams[channel] += w.ple_gated[channel] + silu(convolution);
        }
        pending[..(HISTORY - 1) * d].copy_from_slice(&history[d..]);
        pending[(HISTORY - 1) * d..].copy_from_slice(&w.ple_norm);
        math::finite(&w.streams, "PLE stream update")
    }
    fn delta(
        &self,
        layer: &Delta,
        w: &mut Work,
        current: &Recurrent,
        pending: &mut Recurrent,
    ) -> Result<()> {
        let a = &self.architecture;
        let eps = self.common.rms_epsilon;
        let dk = a.recurrent_key_width;
        let nv = a.recurrent_value_heads;
        let nk = a.recurrent_key_heads;
        let dv = a.recurrent_value_width;
        let qk = nk * dk;
        let channels = w.qkv.len();
        let kernel = a.recurrent_convolution;
        self.mv(&layer.qkv, &w.input, &mut w.qkv)?;
        self.mv(&layer.gate, &w.input, &mut w.z)?;
        match &layer.gates {
            Gates::Split { beta, alpha } => {
                self.mv(beta, &w.input, &mut w.beta)?;
                self.mv(alpha, &w.input, &mut w.alpha)?;
            }
            Gates::Combined(name) => {
                self.mv(name, &w.input, &mut w.ba)?;
                let group = nv / nk;
                for h in 0..nv {
                    let offset = h / group * 2 * group + h % group;
                    w.beta[h] = w.ba[offset];
                    w.alpha[h] = w.ba[offset + group];
                }
            }
        }
        for channel in 0..channels {
            let mut v = 0.;
            for tap in 0..kernel - 1 {
                v += current.convolution[tap * channels + channel]
                    * layer.convolution[channel * kernel + tap];
            }
            v += w.qkv[channel] * layer.convolution[channel * kernel + kernel - 1];
            w.activated[channel] = silu(v);
        }
        if kernel > 1 {
            if kernel > 2 {
                pending.convolution[..(kernel - 2) * channels]
                    .copy_from_slice(&current.convolution[channels..]);
            }
            pending.convolution[(kernel - 2) * channels..].copy_from_slice(&w.qkv);
        }
        for head in w.activated[..2 * qk].chunks_exact_mut(dk) {
            let sq = head.iter().map(|x| x * x).sum::<f32>();
            let inv = sq.sqrt().max(eps).recip();
            for x in head {
                *x *= inv;
            }
        }
        for head in 0..nv {
            let key_head = head % nk;
            let query = &w.activated[key_head * dk..(key_head + 1) * dk];
            let key = &w.activated[qk + key_head * dk..qk + (key_head + 1) * dk];
            let value = &w.activated[2 * qk + head * dv..2 * qk + (head + 1) * dv];
            let decay = (softplus(w.alpha[head] + layer.dt_bias[head]) * layer.decay[head]).exp();
            let beta = sigmoid(w.beta[head]);
            for j in 0..dv {
                let base = (head * dv + j) * dk;
                let old = &current.matrix[base..base + dk];
                let new = &mut pending.matrix[base..base + dk];
                let mut prediction = 0.;
                for i in 0..dk {
                    new[i] = old[i] * decay;
                    prediction += new[i] * key[i];
                }
                let error = beta * (value[j] - prediction);
                for i in 0..dk {
                    new[i] += error * key[i];
                }
                w.head[j] =
                    new.iter().zip(query).map(|(x, y)| x * y).sum::<f32>() / (dk as f32).sqrt();
            }
            let output = &mut w.recurrent_out[head * dv..(head + 1) * dv];
            math::rms_norm(&w.head, &layer.norm, eps, output)?;
            for (j, x) in output.iter_mut().enumerate() {
                *x *= sigmoid(w.z[head * dv + j]);
            }
        }
        math::finite(&pending.matrix, "DeltaNet recurrent state")?;
        self.mv(&layer.output, &w.recurrent_out, &mut w.update)
    }
    fn qsa(
        &self,
        indexer: &Indexer,
        w: &mut Work,
        history: &mut Vec<f32>,
        pooled_keys: &mut Vec<f32>,
        position: usize,
    ) -> Result<()> {
        let a = &self.architecture;
        let d = a.indexer_head_width;
        let eps = self.common.rms_epsilon;
        self.mv(&indexer.key, &w.input, &mut w.ik)?;
        history.extend_from_slice(&w.ik);
        let complete = (position + 1) / POOL;
        let keep = a.indexer_token_budget / POOL;
        w.cells.clear();
        // Each complete key block is pooled/normed/rotated once, then shared by
        // every subsequent decode position. Raw keys remain available for state
        // validation and the incomplete tail never enters this immutable cache.
        while pooled_keys.len() / d < complete {
            let block = pooled_keys.len() / d;
            w.pooled.fill(0.);
            for row in history[block * POOL * d..(block + 1) * POOL * d].chunks_exact(d) {
                for (y, x) in w.pooled.iter_mut().zip(row) {
                    *y += *x / POOL as f32;
                }
            }
            math::rms_norm(&w.pooled, &indexer.key_norm, eps, &mut w.pool_norm)?;
            math::rope_angles(
                block * POOL,
                &self.frequencies,
                &mut w.pool_cos,
                &mut w.pool_sin,
            )?;
            math::rope_heads(&mut w.pool_norm, d, &w.pool_cos, &w.pool_sin)?;
            pooled_keys.extend_from_slice(&w.pool_norm);
        }
        if complete <= keep {
            w.cells.extend(0..position + 1);
            return Ok(());
        }
        self.mv(&indexer.query, &w.input, &mut w.iq)?;
        math::rms_norm_heads(&mut w.iq, &indexer.query_norm, eps)?;
        math::rope_heads(&mut w.iq, d, &w.cos, &w.sin)?;
        w.blocks.clear();
        for block in 0..complete {
            let key = &pooled_keys[block * d..(block + 1) * d];
            let mut score = 0.0;
            for query in w.iq.chunks_exact(d) {
                let head_score = query.iter().zip(key).map(|(x, y)| x * y).sum::<f32>();
                ensure!(head_score.is_finite(), "non-finite QSA head score");
                score += head_score.max(0.0);
            }
            score /= (d as f32).sqrt();
            ensure!(score.is_finite(), "non-finite QSA block score");
            w.blocks.push((block, score));
        }
        w.blocks.sort_unstable_by(|a, b| {
            if a.1 == b.1 {
                a.0.cmp(&b.0)
            } else {
                b.1.total_cmp(&a.1)
            }
        });
        w.blocks.truncate(keep);
        w.blocks.sort_unstable_by_key(|b| b.0);
        for (block, _) in &w.blocks {
            w.cells.extend(block * POOL..(block + 1) * POOL);
        }
        w.cells.extend(complete * POOL..position + 1);
        Ok(())
    }
    fn attention(
        &self,
        layer: &Attn,
        w: &mut Work,
        keys: &mut Vec<f32>,
        values: &mut Vec<f32>,
        indexer_keys: &mut Vec<f32>,
        pooled_keys: &mut Vec<f32>,
        position: usize,
    ) -> Result<()> {
        let c = &self.common;
        let hd = c.head_dim;
        let kw = c.head_count_kv * hd;
        if let Some(indexer) = &layer.indexer {
            self.qsa(indexer, w, indexer_keys, pooled_keys, position)?;
        } else {
            w.cells.clear();
            w.cells.extend(0..position + 1);
        }
        self.mv(&layer.query, &w.input, &mut w.qg)?;
        for head in 0..c.head_count {
            w.query[head * hd..(head + 1) * hd]
                .copy_from_slice(&w.qg[head * hd * 2..head * hd * 2 + hd]);
        }
        self.mv(&layer.key, &w.input, &mut w.key)?;
        self.mv(&layer.value, &w.input, &mut w.value)?;
        math::rms_norm_heads(&mut w.query, &layer.query_norm, c.rms_epsilon)?;
        math::rms_norm_heads(&mut w.key, &layer.key_norm, c.rms_epsilon)?;
        math::rope_heads(&mut w.query, hd, &w.cos, &w.sin)?;
        math::rope_heads(&mut w.key, hd, &w.cos, &w.sin)?;
        keys.extend_from_slice(&w.key);
        values.extend_from_slice(&w.value);
        w.scores.resize(w.cells.len(), 0.);
        w.attn.fill(0.);
        for head in 0..c.head_count {
            let kh = head / (c.head_count / c.head_count_kv);
            let query = &w.query[head * hd..(head + 1) * hd];
            for (s, &cell) in w.scores.iter_mut().zip(&w.cells) {
                let key = &keys[cell * kw + kh * hd..cell * kw + (kh + 1) * hd];
                *s = query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt();
            }
            math::softmax_in_place(&mut w.scores)?;
            for (&weight, &cell) in w.scores.iter().zip(&w.cells) {
                let value = &values[cell * kw + kh * hd..cell * kw + (kh + 1) * hd];
                for (j, x) in value.iter().enumerate() {
                    w.attn[head * hd + j] += weight * x;
                }
            }
            for j in 0..hd {
                w.attn[head * hd + j] *= sigmoid(w.qg[head * hd * 2 + hd + j]);
            }
        }
        self.mv(&layer.output, &w.attn, &mut w.update)
    }
    fn moe(&self, layer: &Experts, w: &mut Work, admit_experts: bool) -> Result<()> {
        let c = &self.common;
        let ff = c.expert_feed_forward_length;
        self.mv(&layer.router, &w.input, &mut w.router)?;
        math::route_experts(
            &w.router,
            c.experts_per_token,
            true,
            c.expert_weights_scale,
            &mut w.selection,
            &mut w.probabilities,
        )?;
        w.update.fill(0.);
        for &(expert, weight) in &w.selection {
            let gate = if admit_experts {
                self.source.expert_cached(&layer.gate, expert)?
            } else {
                self.source.expert_cached_for_prefill(&layer.gate, expert)?
            };
            self.executor
                .matvec(&gate.tensor(), &w.input, &mut w.ffn_gate[..ff])?;
            drop(gate);
            let up = if admit_experts {
                self.source.expert_cached(&layer.up, expert)?
            } else {
                self.source.expert_cached_for_prefill(&layer.up, expert)?
            };
            self.executor
                .matvec(&up.tensor(), &w.input, &mut w.ffn_up[..ff])?;
            drop(up);
            math::swiglu(&mut w.ffn_gate[..ff], &w.ffn_up[..ff])?;
            let down = if admit_experts {
                self.source.expert_cached(&layer.down, expert)?
            } else {
                self.source.expert_cached_for_prefill(&layer.down, expert)?
            };
            self.executor
                .matvec(&down.tensor(), &w.ffn_gate[..ff], &mut w.expert_out)?;
            drop(down);
            for (y, x) in w.update.iter_mut().zip(&w.expert_out) {
                *y += weight * x;
            }
        }
        if let Some((gate, up, down, scale)) = &layer.shared {
            let width = self.architecture.shared_expert_width;
            let mut weight = [0.];
            let weight = if let Some(scale) = scale {
                self.mv(scale, &w.input, &mut weight)?;
                sigmoid(weight[0])
            } else {
                1.
            };
            self.mv(gate, &w.input, &mut w.ffn_gate[..width])?;
            self.mv(up, &w.input, &mut w.ffn_up[..width])?;
            math::swiglu(&mut w.ffn_gate[..width], &w.ffn_up[..width])?;
            self.mv(down, &w.ffn_gate[..width], &mut w.expert_out)?;
            for (y, x) in w.update.iter_mut().zip(&w.expert_out) {
                *y += weight * x;
            }
        }
        math::finite(&w.update, "expert combined output")
    }
    /// Recurrent updates are calculated into persistent alternate buffers. They
    /// become visible only after the complete token (including output projection)
    /// succeeds. Attention appends are truncated if any operation fails.
    pub fn step(
        &self,
        state: &mut Qwen38State,
        token: u32,
        project: bool,
        admit_experts: bool,
        out: &mut Step,
    ) -> Result<()> {
        ensure!(
            state.position < self.common.context_length
                && (token as usize) < self.common.vocab_size,
            "invalid Qwen3.8 token/context"
        );
        let old_previous = state.previous;
        for layer in &mut state.layers {
            if let LayerState::Attn {
                key,
                value,
                indexer,
                ..
            } = layer
            {
                key.try_reserve(state.kv_width)
                    .context("reserve key cache")?;
                value
                    .try_reserve(state.kv_width)
                    .context("reserve value cache")?;
                indexer
                    .try_reserve(state.index_width)
                    .context("reserve indexer cache")?;
            }
        }
        let result = (|| {
            let w = &mut state.work;
            let n = self.common.embedding_length;
            self.source
                .tensor("token_embd.weight")?
                .read_row(token as usize, &mut w.input)?;
            for stream in w.streams.chunks_exact_mut(n) {
                stream.copy_from_slice(&w.input);
            }
            math::rope_angles(state.position, &self.frequencies, &mut w.cos, &mut w.sin)?;
            self.hash_embedding(old_previous, token, &mut w.embedding)?;
            for (layer, layer_state) in self.layers.iter().zip(&mut state.layers) {
                if let Some(ple) = &layer.ple {
                    self.ple(ple, w, &state.history, &mut state.pending_history)?;
                }
                self.read_hc(&layer.read_attn, w)?;
                match (&layer.mixer, layer_state) {
                    (
                        Mixer::Attn(weights),
                        LayerState::Attn {
                            key,
                            value,
                            indexer,
                            pooled,
                        },
                    ) => self.attention(weights, w, key, value, indexer, pooled, state.position)?,
                    (Mixer::Delta(weights), LayerState::Delta { current, pending }) => {
                        self.delta(weights, w, current, pending)?
                    }
                    _ => bail!("Qwen3.8 layer state type mismatch"),
                }
                self.write_hc(w)?;
                self.read_hc(&layer.read_ffn, w)?;
                self.moe(&layer.experts, w, admit_experts)?;
                self.write_hc(w)?;
            }
            if project {
                self.read_hc(&self.output, w)?;
                out.hidden.resize(n, 0.);
                out.hidden.copy_from_slice(&w.input);
                out.logits.resize(self.common.vocab_size, 0.);
                self.mv("output.weight", &out.hidden, &mut out.logits)?;
                math::finite(&out.logits, "Qwen3.8 logits")?;
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
                if self.hash.is_some() {
                    std::mem::swap(&mut state.history, &mut state.pending_history);
                    state.previous = [token as i64, old_previous[0]];
                }
                state.position += 1;
                Ok(())
            }
            Err(e) => {
                state.truncate_attention();
                Err(e)
            }
        }
    }
}
