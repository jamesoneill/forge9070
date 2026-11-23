//! Stateful Qwen3.8 Vulkan graph with bounded packed expert residency.
//!
//! Dense/state work remains on-device between layers. Only routing decisions
//! and a small FFN input are read at layer boundaries; the latter permits exact
//! CPU execution of an expert when its packed working set cannot fit the GPU
//! budget. Cold PLE and token-embedding rows are gathered lazily on the host.
use super::{
    GpuMatvecExecutor,
    context::{
        Buffer, Commands, Device, DeviceOptions, Operation, Push, bytes_of_f32, bytes_of_f32_mut,
        float_bytes,
    },
};
use crate::{
    gguf::{Gguf, Tensor, TensorKey},
    math,
    model::{Config, CpuMatvec, Step},
    qwen38::{Gates, Hc, Mixer, Qwen38},
};
use anyhow::{Context, Result, ensure};
use ash::vk;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Debug)]
struct Part {
    buffer: Arc<Buffer>,
    rows: usize,
    start: usize,
}
#[derive(Debug)]
struct Matrix {
    width: usize,
    ty: u32,
    stride: usize,
    parts: Vec<Part>,
}
#[derive(Clone, Copy, Debug)]
enum Dynamic {
    Fixed,
    Position(usize),
    Tokens(usize),
    RawIndex,
    Pool,
    Blocks,
    QsaSelect,
    SparseAttention,
    DenseScores,
}
#[derive(Debug)]
struct Op {
    operation: Operation,
    dynamic: Dynamic,
}
type Graph = Vec<Op>;

#[derive(Debug)]
enum LayerState {
    Attention {
        key: Arc<Buffer>,
        value: Arc<Buffer>,
        indexer: Option<Arc<Buffer>>,
        pooled: Option<Arc<Buffer>>,
    },
    Delta {
        convolution: Arc<Buffer>,
        matrix: Arc<Buffer>,
    },
}
#[derive(Debug)]
pub struct QwenGpuModel {
    plan: Qwen38,
    cache: GpuMatvecExecutor,
    device: Arc<Device>,
    dense: BTreeMap<String, Matrix>,
    extra: BTreeMap<String, Arc<Buffer>>,
    owner: Arc<()>,
    fixed_bytes: usize,
    cpu_experts: AtomicU64,
    gpu_experts: AtomicU64,
    cpu_gpu_overlap_batches: AtomicU64,
    tokens: AtomicU64,
}
#[derive(Debug)]
pub struct QwenGpuSequence {
    initial_commands: Commands,
    expert_commands: Commands,
    owner: Arc<()>,
    position: usize,
    previous: [i64; 2],
    poisoned: bool,
    scratch: BTreeMap<&'static str, Arc<Buffer>>,
    layers: Vec<LayerState>,
    history: Option<Arc<Buffer>>,
    initial: Graph,
    stages: Vec<Graph>,
    shared: Vec<Graph>,
    finish: Vec<Graph>,
    output: Graph,
    upload: Arc<Buffer>,
    readback: Arc<Buffer>,
    expert_upload: Arc<Buffer>,
    cpu_input: Vec<f32>,
    cpu_embeddings: Vec<f32>,
    cpu_outputs: Vec<f32>,
    cpu_gate: Vec<f32>,
    cpu_up: Vec<f32>,
    cpu_expert: Vec<f32>,
    cpu_routes: Vec<f32>,
    selected: Vec<(usize, f32)>,
    config: Config,
    architecture: crate::qwen38::Architecture,
}
impl QwenGpuSequence {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) -> Result<()> {
        self.initial_commands.begin()?;
        for layer in &self.layers {
            if let LayerState::Delta {
                convolution,
                matrix,
            } = layer
            {
                unsafe {
                    self.initial_commands.device.raw.cmd_fill_buffer(
                        self.initial_commands.raw,
                        convolution.raw,
                        0,
                        convolution.size,
                        0,
                    );
                    self.initial_commands.device.raw.cmd_fill_buffer(
                        self.initial_commands.raw,
                        matrix.raw,
                        0,
                        matrix.size,
                        0,
                    );
                }
            }
        }
        if let Some(history) = &self.history {
            unsafe {
                self.initial_commands.device.raw.cmd_fill_buffer(
                    self.initial_commands.raw,
                    history.raw,
                    0,
                    history.size,
                    0,
                )
            }
        }
        self.initial_commands.transfer_to_compute();
        self.initial_commands.submit()?;
        self.position = 0;
        self.previous = [-1, -1];
        self.poisoned = false;
        Ok(())
    }
    fn read_buffer(&mut self, buffer: Arc<Buffer>, count: usize) -> Result<Vec<f32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = float_bytes(count)?;
        ensure!(
            bytes as u64 <= buffer.size,
            "diagnostic state range exceeds buffer"
        );
        let host = self.initial_commands.device.host(bytes)?;
        self.initial_commands.begin()?;
        self.initial_commands.compute_to_transfer();
        unsafe {
            self.initial_commands.device.raw.cmd_copy_buffer(
                self.initial_commands.raw,
                buffer.raw,
                host.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                }],
            )
        };
        self.initial_commands.transfer_to_host();
        self.initial_commands.submit()?;
        let mut output = vec![0.0; count];
        host.read(0, bytes_of_f32_mut(&mut output))?;
        self.initial_commands
            .device
            .readbacks
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(output)
    }
    pub fn read_kv(&mut self, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let LayerState::Attention { key, value, .. } =
            self.layers.get(layer).context("layer outside Qwen model")?
        else {
            anyhow::bail!("layer uses recurrent state, not KV");
        };
        let key = Arc::clone(key);
        let value = Arc::clone(value);
        let count = self.position * self.config.head_count_kv * self.config.head_dim;
        Ok((
            self.read_buffer(key, count)?,
            self.read_buffer(value, count)?,
        ))
    }
    pub fn read_recurrent(&mut self, layer: usize) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let Some(LayerState::Delta {
            convolution,
            matrix,
        }) = self.layers.get(layer)
        else {
            return Ok(None);
        };
        let convolution = Arc::clone(convolution);
        let matrix = Arc::clone(matrix);
        let c = convolution.size as usize / 4;
        let m = matrix.size as usize / 4;
        Ok(Some((
            self.read_buffer(convolution, c)?,
            self.read_buffer(matrix, m)?,
        )))
    }
    pub fn read_indexer_keys(&mut self, layer: usize) -> Result<Option<Vec<f32>>> {
        let Some(LayerState::Attention {
            indexer: Some(keys),
            ..
        }) = self.layers.get(layer)
        else {
            return Ok(None);
        };
        let keys = Arc::clone(keys);
        let count = self.position * self.architecture.indexer_head_width;
        Ok(Some(self.read_buffer(keys, count)?))
    }
    pub fn read_ple_history(&mut self) -> Result<Option<(Vec<f32>, [i64; 2])>> {
        let Some(history) = &self.history else {
            return Ok(None);
        };
        let history = Arc::clone(history);
        let count = history.size as usize / 4;
        Ok(Some((self.read_buffer(history, count)?, self.previous)))
    }
}

fn group(commands: &Commands, count: usize) -> Result<[usize; 3]> {
    commands.groups(count.max(1))
}
fn elements(commands: &Commands, count: usize) -> Result<[usize; 3]> {
    group(commands, count.div_ceil(128))
}
fn bind(
    w: &Arc<Buffer>,
    a: &Arc<Buffer>,
    b: &Arc<Buffer>,
    c: &Arc<Buffer>,
    d: &Arc<Buffer>,
    e: &Arc<Buffer>,
    status: &Arc<Buffer>,
) -> [Arc<Buffer>; 7] {
    [
        Arc::clone(w),
        Arc::clone(a),
        Arc::clone(b),
        Arc::clone(c),
        Arc::clone(d),
        Arc::clone(e),
        Arc::clone(status),
    ]
}
fn add(
    commands: &mut Commands,
    graph: &mut Graph,
    push: Push,
    buffers: [Arc<Buffer>; 7],
    groups: [usize; 3],
    dynamic: Dynamic,
) -> Result<()> {
    graph.push(Op {
        operation: commands.operation(push, buffers, groups)?,
        dynamic,
    });
    Ok(())
}
fn record(commands: &Commands, graph: &Graph, position: usize) {
    for op in graph {
        let mut push = op.operation.push.clone();
        let mut groups = op.operation.groups;
        match op.dynamic {
            Dynamic::Fixed => {}
            Dynamic::Position(i) => push.0[i] = position as u32,
            Dynamic::Tokens(i) => push.0[i] = (position + 1) as u32,
            Dynamic::RawIndex => push.0[12] = (position * push.0[1] as usize) as u32,
            Dynamic::Pool => {
                if (position + 1) % 4 != 0 {
                    continue;
                }
                push.0[2] = (position / 4) as u32;
            }
            Dynamic::Blocks => {
                let complete = (position + 1) / 4;
                if complete == 0 {
                    continue;
                }
                push.0[3] = complete as u32;
                groups[0] = complete as u32;
                push.0[14] = groups[0];
            }
            Dynamic::QsaSelect => {
                push.0[1] = ((position + 1) / 4) as u32;
                push.0[3] = (position + 1) as u32;
            }
            Dynamic::SparseAttention => push.0[6] = (position + 1) as u32,
            Dynamic::DenseScores => {
                push.0[1] = (position + 1) as u32;
                groups[0] = (position + 1) as u32;
            }
        }
        commands.record(&op.operation, &push, groups);
    }
}

impl QwenGpuModel {
    pub fn load(source: Arc<Gguf>, context: usize, options: DeviceOptions) -> Result<Self> {
        let plan = Qwen38::load(source, context, Arc::new(CpuMatvec))?;
        let cache = GpuMatvecExecutor::new(options)?;
        let device = cache.device();
        let max_x = device.properties.limits.max_compute_work_group_count[0] as usize;
        for layer in &plan.layers {
            if let Mixer::Attn(attention) = &layer.mixer {
                if attention.indexer.is_some() {
                    ensure!(
                        context.div_ceil(4) <= max_x,
                        "Qwen pooled QSA block count exceeds GPU dispatch range"
                    );
                    ensure!(
                        context.min(plan.architecture.indexer_token_budget + 3) <= max_x,
                        "Qwen selected QSA token count exceeds GPU dispatch range"
                    );
                } else {
                    ensure!(
                        context <= max_x,
                        "Qwen dense-attention context exceeds GPU dispatch range"
                    );
                }
            }
        }
        let mut names = BTreeSet::new();
        let hc_names = |h: &Hc, set: &mut BTreeSet<String>| {
            set.insert(h.down.clone());
            set.insert(h.up.clone());
            if let Some(i) = &h.inject {
                set.insert(i.clone());
            }
        };
        hc_names(&plan.output, &mut names);
        names.insert("output.weight".to_string());
        for layer in &plan.layers {
            hc_names(&layer.read_attn, &mut names);
            hc_names(&layer.read_ffn, &mut names);
            match &layer.mixer {
                Mixer::Attn(a) => {
                    for s in [&a.query, &a.key, &a.value, &a.output] {
                        names.insert(s.clone());
                    }
                    if let Some(i) = &a.indexer {
                        names.insert(i.query.clone());
                        names.insert(i.key.clone());
                    }
                }
                Mixer::Delta(d) => {
                    for s in [&d.qkv, &d.gate, &d.output] {
                        names.insert(s.clone());
                    }
                    match &d.gates {
                        Gates::Split { beta, alpha } => {
                            names.insert(beta.clone());
                            names.insert(alpha.clone());
                        }
                        Gates::Combined(n) => {
                            names.insert(n.clone());
                        }
                    }
                }
            }
            names.insert(layer.experts.router.clone());
            if let Some((g, u, d, scale)) = &layer.experts.shared {
                for s in [g, u, d] {
                    names.insert(s.clone());
                }
                if let Some(s) = scale {
                    names.insert(s.clone());
                }
            }
            if let Some(ple) = &layer.ple {
                names.insert(ple.key.clone());
                names.insert(ple.value.clone());
            }
        }
        let mut dense = BTreeMap::new();
        let mut fixed_bytes = 0usize;
        for name in names {
            let tensor = plan.source.tensor(&name)?;
            ensure!(
                (1..=2).contains(&tensor.dimensions().len()),
                "Qwen fixed projection must be a vector or matrix: {name}"
            );
            ensure!(
                super::supports_quant(tensor.quant_type().id()),
                "unsupported Qwen GPU quantization"
            );
            let stride = tensor.quant_type().row_bytes(tensor.width())?;
            let rows_per_part = (device.properties.limits.max_storage_buffer_range as usize
                / stride)
                .min(u32::MAX as usize / stride);
            ensure!(
                rows_per_part > 0,
                "one projection row exceeds GPU buffer range"
            );
            let mut parts = Vec::new();
            for start in (0..tensor.rows()).step_by(rows_per_part) {
                let rows = rows_per_part.min(tensor.rows() - start);
                let bytes = &tensor.bytes()[start * stride..(start + rows) * stride];
                let buffer = device.upload(bytes)?;
                fixed_bytes = fixed_bytes
                    .checked_add(buffer.size as usize)
                    .context("fixed GPU payload overflow")?;
                parts.push(Part {
                    buffer,
                    rows,
                    start,
                });
            }
            dense.insert(
                name,
                Matrix {
                    width: tensor.width(),
                    ty: tensor.quant_type().id(),
                    stride,
                    parts,
                },
            );
        }
        let mut extra = BTreeMap::new();
        let mut upload = |name: String, values: &[f32]| -> Result<()> {
            let buffer = device.upload(bytes_of_f32(values))?;
            fixed_bytes = fixed_bytes
                .checked_add(buffer.size as usize)
                .context("fixed norms overflow")?;
            extra.insert(name, buffer);
            Ok(())
        };
        upload("output_hc".into(), &plan.output.norm)?;
        for (i, layer) in plan.layers.iter().enumerate() {
            upload(format!("{i}/hca"), &layer.read_attn.norm)?;
            upload(format!("{i}/hcf"), &layer.read_ffn.norm)?;
            match &layer.mixer {
                Mixer::Attn(a) => {
                    upload(format!("{i}/qn"), &a.query_norm)?;
                    upload(format!("{i}/kn"), &a.key_norm)?;
                    if let Some(index) = &a.indexer {
                        upload(format!("{i}/iqn"), &index.query_norm)?;
                        upload(format!("{i}/ikn"), &index.key_norm)?;
                    }
                }
                Mixer::Delta(d) => {
                    let mut rates = d.dt_bias.clone();
                    rates.extend_from_slice(&d.decay);
                    upload(format!("{i}/rates"), &rates)?;
                    upload(format!("{i}/conv"), &d.convolution)?;
                    upload(format!("{i}/dn"), &d.norm)?;
                }
            }
            if let Some(ple) = &layer.ple {
                upload(format!("{i}/pkn"), &ple.key_norm)?;
                upload(format!("{i}/pqn"), &ple.query_norm)?;
                upload(format!("{i}/pcn"), &ple.convolution_norm)?;
                upload(format!("{i}/pc"), &ple.convolution)?;
            }
        }
        Ok(Self {
            plan,
            cache,
            device,
            dense,
            extra,
            owner: Arc::new(()),
            fixed_bytes,
            cpu_experts: AtomicU64::new(0),
            gpu_experts: AtomicU64::new(0),
            cpu_gpu_overlap_batches: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
        })
    }
    pub fn config(&self) -> &Config {
        &self.plan.common
    }
    pub fn context(&self) -> usize {
        self.plan.common.context_length
    }
    pub fn vocab_size(&self) -> usize {
        self.plan.common.vocab_size
    }
    pub fn adapter_info(&self) -> Value {
        self.device.info()
    }
    pub fn memory_stats(&self) -> Value {
        let mut value = self.cache.stats();
        value["execution"] = json!("qwen38_gpu_state_with_bounded_expert_cache");
        value["synchronous_projection_transfers"] = json!(false);
        value["layer_boundary_readbacks"] = json!(true);
        value["fixed_projection_payload_bytes"] = json!(self.fixed_bytes);
        value["pinned_dense_bytes"] = json!(self.fixed_bytes);
        value["cpu_state_operations"] = json!(false);
        value["cpu_expert_fallbacks"] = json!(self.cpu_experts.load(Ordering::Relaxed));
        value["cpu_gpu_transfer_overlap_batches"] =
            json!(self.cpu_gpu_overlap_batches.load(Ordering::Relaxed));
        value["gpu_expert_executions"] = json!(self.gpu_experts.load(Ordering::Relaxed));
        value["completed_tokens"] = json!(self.tokens.load(Ordering::Relaxed));
        value["kv_precision"] = json!("F32");
        value["recurrent_precision"] = json!("F32");
        value["selected_experts_omitted"] = json!(0);
        value["nominal_warm_submissions_per_token"] = json!(self.plan.common.block_count + 1);
        value["cache_miss_uploads_add_submissions"] = json!(true);
        value["lazy_embedding_rows"] = json!(true);
        value["prefill_misses_evict_decode_experts"] = json!(false);
        value
    }
    fn matrix(
        &self,
        commands: &mut Commands,
        graph: &mut Graph,
        name: &str,
        input: &Arc<Buffer>,
        output: &Arc<Buffer>,
        dummy: &Arc<Buffer>,
        status: &Arc<Buffer>,
        input_offset: usize,
        output_offset: usize,
    ) -> Result<()> {
        let matrix = self
            .dense
            .get(name)
            .with_context(|| format!("missing fixed projection {name}"))?;
        for part in &matrix.parts {
            let mut p = Push::new(0)
                .n(0, matrix.width)?
                .n(1, part.rows)?
                .n(2, matrix.ty as usize)?
                .n(3, matrix.stride)?
                .n(4, 1)?;
            p.0[11] = u32::try_from(input_offset)?;
            p.0[12] = u32::try_from(output_offset + part.start)?;
            let groups = commands.groups(part.rows)?;
            add(
                commands,
                graph,
                p,
                bind(&part.buffer, input, dummy, output, dummy, dummy, status),
                groups,
                Dynamic::Fixed,
            )?;
        }
        Ok(())
    }
    fn norm(
        &self,
        commands: &mut Commands,
        graph: &mut Graph,
        weight: &str,
        input: &Arc<Buffer>,
        output: &Arc<Buffer>,
        dummy: &Arc<Buffer>,
        status: &Arc<Buffer>,
        width: usize,
        heads: usize,
        distinct: bool,
    ) -> Result<()> {
        let w = self
            .extra
            .get(weight)
            .with_context(|| format!("missing GPU norm {weight}"))?;
        let groups = group(commands, heads)?;
        add(
            commands,
            graph,
            Push::new(1)
                .n(0, width)?
                .n(1, heads)?
                .f(0, self.plan.common.rms_epsilon)
                .flags(u32::from(distinct)),
            bind(w, input, dummy, output, dummy, dummy, status),
            groups,
            Dynamic::Fixed,
        )
    }
    fn hc_read(
        &self,
        commands: &mut Commands,
        graph: &mut Graph,
        hc: &Hc,
        norm: &str,
        s: &BTreeMap<&'static str, Arc<Buffer>>,
    ) -> Result<()> {
        let c = &self.plan.common;
        let a = &self.plan.architecture;
        let dummy = &s["dummy"];
        let status = &s["status"];
        self.norm(
            commands,
            graph,
            norm,
            &s["streams"],
            &s["stream_norm"],
            dummy,
            status,
            c.embedding_length,
            a.hyper_streams,
            true,
        )?;
        self.matrix(
            commands,
            graph,
            &hc.down,
            &s["stream_norm"],
            &s["rank"],
            dummy,
            status,
            0,
            0,
        )?;
        let groups = elements(commands, a.hyper_rank)?;
        add(
            commands,
            graph,
            Push::new(101).n(0, a.hyper_rank)?.n(1, a.hyper_streams)?,
            bind(dummy, &s["rank"], dummy, &s["rank"], dummy, dummy, status),
            groups,
            Dynamic::Fixed,
        )?;
        self.matrix(
            commands,
            graph,
            &hc.up,
            &s["rank"],
            &s["mixing"],
            dummy,
            status,
            0,
            0,
        )?;
        let groups = elements(commands, c.embedding_length)?;
        add(
            commands,
            graph,
            Push::new(102)
                .n(0, c.embedding_length)?
                .n(1, a.hyper_streams)?,
            bind(
                dummy,
                &s["stream_norm"],
                &s["mixing"],
                &s["input"],
                dummy,
                dummy,
                status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        if let Some(name) = &hc.inject {
            self.matrix(
                commands,
                graph,
                name,
                &s["stream_norm"],
                &s["inject"],
                dummy,
                status,
                0,
                0,
            )?;
        }
        Ok(())
    }
    fn hc_write(
        &self,
        commands: &mut Commands,
        graph: &mut Graph,
        s: &BTreeMap<&'static str, Arc<Buffer>>,
    ) -> Result<()> {
        let n = self.plan.common.embedding_length;
        let hc = self.plan.architecture.hyper_streams;
        let groups = elements(commands, n * hc)?;
        add(
            commands,
            graph,
            Push::new(103).n(0, n)?.n(1, hc)?,
            bind(
                &s["dummy"],
                &s["streams"],
                &s["update"],
                &s["streams"],
                &s["dummy"],
                &s["inject"],
                &s["status"],
            ),
            groups,
            Dynamic::Fixed,
        )
    }
    fn rope(
        &self,
        commands: &mut Commands,
        graph: &mut Graph,
        buffer: &Arc<Buffer>,
        dummy: &Arc<Buffer>,
        status: &Arc<Buffer>,
        width: usize,
        heads: usize,
    ) -> Result<()> {
        let c = &self.plan.common;
        let groups = elements(commands, heads * (c.rope_dimension_count / 2))?;
        add(
            commands,
            graph,
            Push::new(2)
                .n(0, width)?
                .n(1, heads)?
                .n(3, c.rope_dimension_count)?
                .f(0, c.rope_base),
            bind(dummy, buffer, dummy, buffer, dummy, dummy, status),
            groups,
            Dynamic::Position(3),
        )
    }
    pub fn new_sequence(&self) -> Result<QwenGpuSequence> {
        let c = &self.plan.common;
        let a = &self.plan.architecture;
        let n = c.embedding_length;
        let sw = n * a.hyper_streams;
        let qwidth = c.head_count * c.head_dim;
        let kwidth = c.head_count_kv * c.head_dim;
        let channels = a.recurrent_inner_width + 2 * a.recurrent_key_heads * a.recurrent_key_width;
        let cell_capacity = if a.indexer_heads > 0 {
            c.context_length.min(a.indexer_token_budget + 3)
        } else {
            c.context_length
        };
        let specs = [
            ("dummy", 1),
            ("status", 1),
            ("token", n),
            ("embedding", n),
            ("streams", sw),
            ("stream_norm", sw),
            ("rank", a.hyper_rank),
            ("mixing", sw),
            ("inject", a.hyper_streams),
            ("input", n),
            ("update", n),
            ("qg", qwidth * 2),
            ("q", qwidth),
            ("k", kwidth),
            ("v", kwidth),
            ("attn", qwidth),
            ("scores", c.head_count * cell_capacity),
            ("iq", (a.indexer_heads * a.indexer_head_width).max(1)),
            ("ik", a.indexer_head_width.max(1)),
            ("pooltmp", a.indexer_head_width.max(1)),
            ("blockscores", c.context_length.div_ceil(4).max(1)),
            ("mask", c.context_length.div_ceil(4).max(1)),
            ("cells", cell_capacity + 1),
            ("qkv", channels),
            ("activated", channels),
            ("z", a.recurrent_inner_width),
            ("beta", a.recurrent_value_heads),
            ("alpha", a.recurrent_value_heads),
            ("ba", a.recurrent_value_heads * 2),
            ("rates", a.recurrent_value_heads * 2),
            ("rawrec", a.recurrent_inner_width),
            ("recout", a.recurrent_inner_width),
            ("router", c.expert_count),
            ("probabilities", c.expert_count),
            ("selection", c.experts_per_token * 2),
            ("plekey", sw),
            ("plequery", sw),
            ("plevalue", n),
            ("plegated", sw),
            ("plenorm", sw),
            ("gate", c.experts_per_token * c.expert_feed_forward_length),
            ("up", c.experts_per_token * c.expert_feed_forward_length),
            ("expertout", c.experts_per_token * n),
            ("sharedgate", a.shared_expert_width.max(1)),
            ("sharedup", a.shared_expert_width.max(1)),
            ("sharedout", n),
            ("sharedscale", 1),
            ("logits", c.vocab_size),
        ];
        let layer_floats: usize = self
            .plan
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Attn(at) => {
                    2 * c.context_length * kwidth
                        + if at.indexer.is_some() {
                            c.context_length * a.indexer_head_width
                                + c.context_length.div_ceil(4) * a.indexer_head_width
                        } else {
                            0
                        }
                }
                Mixer::Delta(_) => {
                    (a.recurrent_convolution - 1) * channels
                        + a.recurrent_inner_width * a.recurrent_key_width
                }
            })
            .sum();
        let history_floats = if self.plan.hash.is_some() { 9 * sw } else { 0 };
        let readback_floats = (c.vocab_size + n + 1).max(c.experts_per_token * 2 + n + 1);
        let total_floats = specs
            .iter()
            .map(|(_, n)| *n)
            .sum::<usize>()
            .checked_add(layer_floats)
            .and_then(|v| {
                v.checked_add(history_floats + 2 * n + readback_floats + c.experts_per_token * n)
            })
            .context("Qwen state sizing overflow")?;
        self.cache.make_room(
            float_bytes(total_floats)?
                .checked_add(64 * 1024)
                .context("Qwen allocation allowance overflow")?,
        )?;
        let mut initial_commands = Commands::new(
            Arc::clone(&self.device),
            c.block_count
                .checked_mul(50)
                .and_then(|n| n.checked_add(32))
                .context("Qwen descriptor count overflow")?,
        )?;
        let expert_commands = Commands::new(
            Arc::clone(&self.device),
            c.experts_per_token
                .checked_mul(4)
                .and_then(|n| n.checked_add(16))
                .context("expert descriptor count overflow")?,
        )?;
        let mut scratch = BTreeMap::new();
        for (name, count) in specs {
            scratch.insert(name, self.device.storage(float_bytes(count.max(1))?)?);
        }
        let mut layers = Vec::with_capacity(c.block_count);
        for layer in &self.plan.layers {
            layers.push(match &layer.mixer {
                Mixer::Attn(attn) => LayerState::Attention {
                    key: self
                        .device
                        .storage(float_bytes(c.context_length * kwidth)?)?,
                    value: self
                        .device
                        .storage(float_bytes(c.context_length * kwidth)?)?,
                    indexer: if attn.indexer.is_some() {
                        Some(
                            self.device
                                .storage(float_bytes(c.context_length * a.indexer_head_width)?)?,
                        )
                    } else {
                        None
                    },
                    pooled: if attn.indexer.is_some() {
                        Some(self.device.storage(float_bytes(
                            c.context_length.div_ceil(4) * a.indexer_head_width,
                        )?)?)
                    } else {
                        None
                    },
                },
                Mixer::Delta(_) => LayerState::Delta {
                    convolution: self.device.storage(float_bytes(
                        ((a.recurrent_convolution - 1) * channels).max(1),
                    )?)?,
                    matrix: self.device.storage(float_bytes(
                        a.recurrent_inner_width * a.recurrent_key_width,
                    )?)?,
                },
            });
        }
        let history = if history_floats > 0 {
            Some(self.device.storage(float_bytes(history_floats)?)?)
        } else {
            None
        };
        let upload = self.device.host(float_bytes(2 * n)?)?;
        let readback = self.device.host(float_bytes(readback_floats)?)?;
        let expert_upload = self.device.host(float_bytes(c.experts_per_token * n)?)?;
        let dummy = &scratch["dummy"];
        let status = &scratch["status"];
        let mut initial = Vec::new();
        let mut stages = Vec::new();
        let mut shared = Vec::new();
        let mut finish = Vec::new();
        let mut output = Vec::new();
        let groups = elements(&initial_commands, sw)?;
        add(
            &mut initial_commands,
            &mut initial,
            Push::new(100).n(0, n)?.n(1, a.hyper_streams)?,
            bind(
                dummy,
                &scratch["token"],
                dummy,
                &scratch["streams"],
                dummy,
                dummy,
                status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        for (i, layer) in self.plan.layers.iter().enumerate() {
            let mut graph = Vec::new();
            if let Some(ple) = &layer.ple {
                self.matrix(
                    &mut initial_commands,
                    &mut graph,
                    &ple.key,
                    &scratch["embedding"],
                    &scratch["plekey"],
                    dummy,
                    status,
                    0,
                    0,
                )?;
                self.norm(
                    &mut initial_commands,
                    &mut graph,
                    &format!("{i}/pkn"),
                    &scratch["plekey"],
                    &scratch["plekey"],
                    dummy,
                    status,
                    n,
                    a.hyper_streams,
                    true,
                )?;
                self.norm(
                    &mut initial_commands,
                    &mut graph,
                    &format!("{i}/pqn"),
                    &scratch["streams"],
                    &scratch["plequery"],
                    dummy,
                    status,
                    n,
                    a.hyper_streams,
                    true,
                )?;
                self.matrix(
                    &mut initial_commands,
                    &mut graph,
                    &ple.value,
                    &scratch["embedding"],
                    &scratch["plevalue"],
                    dummy,
                    status,
                    0,
                    0,
                )?;
                let groups = group(&initial_commands, a.hyper_streams)?;
                add(
                    &mut initial_commands,
                    &mut graph,
                    Push::new(120).n(0, n)?.n(1, a.hyper_streams)?,
                    bind(
                        dummy,
                        &scratch["plekey"],
                        &scratch["plequery"],
                        &scratch["plegated"],
                        dummy,
                        &scratch["plevalue"],
                        status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
                self.norm(
                    &mut initial_commands,
                    &mut graph,
                    &format!("{i}/pcn"),
                    &scratch["plegated"],
                    &scratch["plenorm"],
                    dummy,
                    status,
                    n,
                    a.hyper_streams,
                    true,
                )?;
                let groups = elements(&initial_commands, sw)?;
                let h = history.as_ref().unwrap();
                add(
                    &mut initial_commands,
                    &mut graph,
                    Push::new(121).n(0, sw)?,
                    bind(
                        &self.extra[&format!("{i}/pc")],
                        &scratch["plenorm"],
                        h,
                        h,
                        &scratch["streams"],
                        &scratch["plegated"],
                        status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
            }
            self.hc_read(
                &mut initial_commands,
                &mut graph,
                &layer.read_attn,
                &format!("{i}/hca"),
                &scratch,
            )?;
            match (&layer.mixer, &layers[i]) {
                (
                    Mixer::Delta(weights),
                    LayerState::Delta {
                        convolution,
                        matrix,
                    },
                ) => {
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.qkv,
                        &scratch["input"],
                        &scratch["qkv"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.gate,
                        &scratch["input"],
                        &scratch["z"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                    match &weights.gates {
                        Gates::Split { beta, alpha } => {
                            self.matrix(
                                &mut initial_commands,
                                &mut graph,
                                beta,
                                &scratch["input"],
                                &scratch["beta"],
                                dummy,
                                status,
                                0,
                                0,
                            )?;
                            self.matrix(
                                &mut initial_commands,
                                &mut graph,
                                alpha,
                                &scratch["input"],
                                &scratch["alpha"],
                                dummy,
                                status,
                                0,
                                0,
                            )?;
                        }
                        Gates::Combined(name) => {
                            self.matrix(
                                &mut initial_commands,
                                &mut graph,
                                name,
                                &scratch["input"],
                                &scratch["ba"],
                                dummy,
                                status,
                                0,
                                0,
                            )?;
                            let groups = elements(&initial_commands, a.recurrent_value_heads)?;
                            add(
                                &mut initial_commands,
                                &mut graph,
                                Push::new(115)
                                    .n(0, a.recurrent_value_heads)?
                                    .n(1, a.recurrent_key_heads)?,
                                bind(
                                    dummy,
                                    &scratch["ba"],
                                    dummy,
                                    &scratch["beta"],
                                    &scratch["alpha"],
                                    dummy,
                                    status,
                                ),
                                groups,
                                Dynamic::Fixed,
                            )?;
                        }
                    }
                    let groups = elements(&initial_commands, channels)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(110)
                            .n(0, channels)?
                            .n(1, a.recurrent_convolution)?,
                        bind(
                            &self.extra[&format!("{i}/conv")],
                            &scratch["qkv"],
                            convolution,
                            &scratch["activated"],
                            convolution,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    let groups = group(&initial_commands, 2 * a.recurrent_key_heads)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(111)
                            .n(0, a.recurrent_key_width)?
                            .n(1, 2 * a.recurrent_key_heads)?
                            .f(0, c.rms_epsilon),
                        bind(
                            dummy,
                            &scratch["activated"],
                            dummy,
                            &scratch["activated"],
                            dummy,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    let groups = elements(&initial_commands, a.recurrent_value_heads)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(112).n(0, a.recurrent_value_heads)?,
                        bind(
                            &self.extra[&format!("{i}/rates")],
                            &scratch["beta"],
                            &scratch["alpha"],
                            &scratch["rates"],
                            dummy,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    let groups = group(
                        &initial_commands,
                        a.recurrent_value_heads * a.recurrent_value_width,
                    )?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(113)
                            .n(0, a.recurrent_key_width)?
                            .n(1, a.recurrent_key_heads)?
                            .n(2, a.recurrent_value_heads)?
                            .n(3, a.recurrent_value_width)?,
                        bind(
                            matrix,
                            &scratch["activated"],
                            &scratch["rates"],
                            matrix,
                            &scratch["rawrec"],
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    let groups = group(&initial_commands, a.recurrent_value_heads)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(114)
                            .n(0, a.recurrent_value_width)?
                            .n(1, a.recurrent_value_heads)?
                            .f(0, c.rms_epsilon),
                        bind(
                            &self.extra[&format!("{i}/dn")],
                            &scratch["rawrec"],
                            &scratch["z"],
                            &scratch["recout"],
                            dummy,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.output,
                        &scratch["recout"],
                        &scratch["update"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                }
                (
                    Mixer::Attn(weights),
                    LayerState::Attention {
                        key,
                        value,
                        indexer,
                        pooled,
                    },
                ) => {
                    if let Some(index) = &weights.indexer {
                        let raw = indexer.as_ref().unwrap();
                        let pool = pooled.as_ref().unwrap();
                        let d = a.indexer_head_width;
                        self.matrix(
                            &mut initial_commands,
                            &mut graph,
                            &index.key,
                            &scratch["input"],
                            &scratch["ik"],
                            dummy,
                            status,
                            0,
                            0,
                        )?;
                        let groups = elements(&initial_commands, d)?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            Push::new(12).n(0, d)?,
                            bind(dummy, &scratch["ik"], dummy, raw, dummy, dummy, status),
                            groups,
                            Dynamic::RawIndex,
                        )?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            Push::new(130)
                                .n(0, d)?
                                .n(2, c.rope_dimension_count)?
                                .n(3, 4)?
                                .f(0, c.rms_epsilon)
                                .f(1, c.rope_base),
                            bind(
                                &self.extra[&format!("{i}/ikn")],
                                raw,
                                &scratch["pooltmp"],
                                pool,
                                dummy,
                                dummy,
                                status,
                            ),
                            [1, 1, 1],
                            Dynamic::Pool,
                        )?;
                        self.matrix(
                            &mut initial_commands,
                            &mut graph,
                            &index.query,
                            &scratch["input"],
                            &scratch["iq"],
                            dummy,
                            status,
                            0,
                            0,
                        )?;
                        self.norm(
                            &mut initial_commands,
                            &mut graph,
                            &format!("{i}/iqn"),
                            &scratch["iq"],
                            &scratch["iq"],
                            dummy,
                            status,
                            d,
                            a.indexer_heads,
                            false,
                        )?;
                        self.rope(
                            &mut initial_commands,
                            &mut graph,
                            &scratch["iq"],
                            dummy,
                            status,
                            d,
                            a.indexer_heads,
                        )?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            Push::new(131).n(0, d)?.n(1, a.indexer_heads)?.n(2, 1)?,
                            bind(
                                dummy,
                                &scratch["iq"],
                                pool,
                                &scratch["blockscores"],
                                dummy,
                                dummy,
                                status,
                            ),
                            [1, 1, 1],
                            Dynamic::Blocks,
                        )?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            Push::new(132)
                                .n(0, 0)?
                                .n(1, a.indexer_token_budget / 4)?
                                .n(2, 1)?
                                .n(3, cell_capacity)?,
                            bind(
                                dummy,
                                &scratch["blockscores"],
                                dummy,
                                &scratch["mask"],
                                &scratch["cells"],
                                dummy,
                                status,
                            ),
                            [1, 1, 1],
                            Dynamic::QsaSelect,
                        )?;
                    }
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.query,
                        &scratch["input"],
                        &scratch["qg"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                    let groups = elements(&initial_commands, qwidth)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(104).n(0, c.head_dim)?.n(1, c.head_count)?,
                        bind(
                            dummy,
                            &scratch["qg"],
                            dummy,
                            &scratch["q"],
                            dummy,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.key,
                        &scratch["input"],
                        &scratch["k"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.value,
                        &scratch["input"],
                        &scratch["v"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                    self.norm(
                        &mut initial_commands,
                        &mut graph,
                        &format!("{i}/qn"),
                        &scratch["q"],
                        &scratch["q"],
                        dummy,
                        status,
                        c.head_dim,
                        c.head_count,
                        false,
                    )?;
                    self.norm(
                        &mut initial_commands,
                        &mut graph,
                        &format!("{i}/kn"),
                        &scratch["k"],
                        &scratch["k"],
                        dummy,
                        status,
                        c.head_dim,
                        c.head_count_kv,
                        false,
                    )?;
                    self.rope(
                        &mut initial_commands,
                        &mut graph,
                        &scratch["q"],
                        dummy,
                        status,
                        c.head_dim,
                        c.head_count,
                    )?;
                    self.rope(
                        &mut initial_commands,
                        &mut graph,
                        &scratch["k"],
                        dummy,
                        status,
                        c.head_dim,
                        c.head_count_kv,
                    )?;
                    let groups = elements(&initial_commands, kwidth)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(3).n(0, kwidth)?,
                        bind(
                            dummy,
                            &scratch["k"],
                            &scratch["v"],
                            key,
                            value,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Position(2),
                    )?;
                    if weights.indexer.is_some() {
                        let p = Push::new(133)
                            .n(0, cell_capacity)?
                            .n(1, c.head_count)?
                            .n(2, c.head_count_kv)?
                            .n(3, c.head_dim)?
                            .n(4, cell_capacity)?
                            .n(5, 1)?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            p,
                            bind(
                                dummy,
                                &scratch["q"],
                                key,
                                &scratch["scores"],
                                dummy,
                                &scratch["cells"],
                                status,
                            ),
                            [cell_capacity, c.head_count, 1],
                            Dynamic::SparseAttention,
                        )?;
                        let p = Push::new(134)
                            .n(0, cell_capacity)?
                            .n(1, c.head_count)?
                            .n(2, c.head_count_kv)?
                            .n(3, c.head_dim)?
                            .n(4, cell_capacity)?
                            .n(5, 1)?;
                        let groups = group(&initial_commands, c.head_count)?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            p,
                            bind(
                                dummy,
                                &scratch["scores"],
                                value,
                                &scratch["attn"],
                                dummy,
                                &scratch["cells"],
                                status,
                            ),
                            groups,
                            Dynamic::SparseAttention,
                        )?;
                    } else {
                        add(
                            &mut initial_commands,
                            &mut graph,
                            Push::new(4)
                                .n(0, 1)?
                                .n(1, c.head_count)?
                                .n(2, c.head_count_kv)?
                                .n(3, c.head_dim)?
                                .n(4, c.context_length)?,
                            bind(
                                dummy,
                                &scratch["q"],
                                key,
                                &scratch["scores"],
                                dummy,
                                dummy,
                                status,
                            ),
                            [1, c.head_count, 1],
                            Dynamic::DenseScores,
                        )?;
                        let groups = group(&initial_commands, c.head_count)?;
                        add(
                            &mut initial_commands,
                            &mut graph,
                            Push::new(5)
                                .n(0, 1)?
                                .n(1, c.head_count)?
                                .n(2, c.head_count_kv)?
                                .n(3, c.head_dim)?
                                .n(4, c.context_length)?,
                            bind(
                                dummy,
                                &scratch["scores"],
                                value,
                                &scratch["attn"],
                                dummy,
                                dummy,
                                status,
                            ),
                            groups,
                            Dynamic::Tokens(1),
                        )?;
                    }
                    let groups = elements(&initial_commands, qwidth)?;
                    add(
                        &mut initial_commands,
                        &mut graph,
                        Push::new(105).n(0, c.head_dim)?.n(1, c.head_count)?,
                        bind(
                            dummy,
                            &scratch["attn"],
                            &scratch["qg"],
                            &scratch["attn"],
                            dummy,
                            dummy,
                            status,
                        ),
                        groups,
                        Dynamic::Fixed,
                    )?;
                    self.matrix(
                        &mut initial_commands,
                        &mut graph,
                        &weights.output,
                        &scratch["attn"],
                        &scratch["update"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                }
                _ => unreachable!(),
            }
            self.hc_write(&mut initial_commands, &mut graph, &scratch)?;
            self.hc_read(
                &mut initial_commands,
                &mut graph,
                &layer.read_ffn,
                &format!("{i}/hcf"),
                &scratch,
            )?;
            self.matrix(
                &mut initial_commands,
                &mut graph,
                &layer.experts.router,
                &scratch["input"],
                &scratch["router"],
                dummy,
                status,
                0,
                0,
            )?;
            add(
                &mut initial_commands,
                &mut graph,
                Push::new(8)
                    .n(0, c.expert_count)?
                    .n(1, c.experts_per_token)?
                    .f(0, c.expert_weights_scale)
                    .flags(1),
                bind(
                    dummy,
                    &scratch["router"],
                    dummy,
                    &scratch["probabilities"],
                    dummy,
                    &scratch["selection"],
                    status,
                ),
                [1, 1, 1],
                Dynamic::Fixed,
            )?;
            stages.push(graph);
            let mut shared_graph = Vec::new();
            if let Some((gate, up, down, scale)) = &layer.experts.shared {
                if let Some(scale) = scale {
                    self.matrix(
                        &mut initial_commands,
                        &mut shared_graph,
                        scale,
                        &scratch["input"],
                        &scratch["sharedscale"],
                        dummy,
                        status,
                        0,
                        0,
                    )?;
                }
                self.matrix(
                    &mut initial_commands,
                    &mut shared_graph,
                    gate,
                    &scratch["input"],
                    &scratch["sharedgate"],
                    dummy,
                    status,
                    0,
                    0,
                )?;
                self.matrix(
                    &mut initial_commands,
                    &mut shared_graph,
                    up,
                    &scratch["input"],
                    &scratch["sharedup"],
                    dummy,
                    status,
                    0,
                    0,
                )?;
                let groups = elements(&initial_commands, a.shared_expert_width)?;
                add(
                    &mut initial_commands,
                    &mut shared_graph,
                    Push::new(6).n(0, a.shared_expert_width)?,
                    bind(
                        dummy,
                        &scratch["sharedgate"],
                        &scratch["sharedup"],
                        &scratch["sharedgate"],
                        dummy,
                        dummy,
                        status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
                self.matrix(
                    &mut initial_commands,
                    &mut shared_graph,
                    down,
                    &scratch["sharedgate"],
                    &scratch["sharedout"],
                    dummy,
                    status,
                    0,
                    0,
                )?;
                let groups = elements(&initial_commands, n)?;
                add(
                    &mut initial_commands,
                    &mut shared_graph,
                    Push::new(106).n(0, n)?.flags(u32::from(scale.is_none())),
                    bind(
                        dummy,
                        &scratch["update"],
                        &scratch["sharedout"],
                        &scratch["update"],
                        dummy,
                        &scratch["sharedscale"],
                        status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
            }
            shared.push(shared_graph);
            let mut end = Vec::new();
            self.hc_write(&mut initial_commands, &mut end, &scratch)?;
            finish.push(end);
        }
        self.hc_read(
            &mut initial_commands,
            &mut output,
            &self.plan.output,
            "output_hc",
            &scratch,
        )?;
        self.matrix(
            &mut initial_commands,
            &mut output,
            "output.weight",
            &scratch["input"],
            &scratch["logits"],
            dummy,
            status,
            0,
            0,
        )?;
        let mut sequence = QwenGpuSequence {
            initial_commands,
            expert_commands,
            owner: Arc::clone(&self.owner),
            position: 0,
            previous: [-1, -1],
            poisoned: false,
            scratch,
            layers,
            history,
            initial,
            stages,
            shared,
            finish,
            output,
            upload,
            readback,
            expert_upload,
            cpu_input: vec![0.; n],
            cpu_embeddings: vec![0.; 2 * n],
            cpu_outputs: vec![0.; c.experts_per_token * n],
            cpu_gate: vec![0.; c.expert_feed_forward_length],
            cpu_up: vec![0.; c.expert_feed_forward_length],
            cpu_expert: vec![0.; n],
            cpu_routes: vec![0.; c.experts_per_token * 2],
            selected: Vec::with_capacity(c.experts_per_token),
            config: c.clone(),
            architecture: a.clone(),
        };
        sequence.reset()?;
        Ok(sequence)
    }
    fn cpu_expert(
        &self,
        layer: usize,
        expert: usize,
        admit: bool,
        s: &mut QwenGpuSequence,
        slot: usize,
        staged: &BTreeMap<TensorKey, Tensor<'_>>,
    ) -> Result<()> {
        let e = &self.plan.layers[layer].experts;
        for (name, output) in [(&e.gate, &mut s.cpu_gate), (&e.up, &mut s.cpu_up)] {
            let key = self.plan.source.tensor(name)?.expert(expert)?.cache_key();
            if let Some(view) = staged.get(&key) {
                view.matvec_into(&s.cpu_input, output)?;
            } else {
                let lease = if admit {
                    self.plan.source.expert_cached(name, expert)?
                } else {
                    self.plan.source.expert_cached_for_prefill(name, expert)?
                };
                lease.tensor().matvec_into(&s.cpu_input, output)?;
            }
        }
        math::swiglu(&mut s.cpu_gate, &s.cpu_up)?;
        let key = self
            .plan
            .source
            .tensor(&e.down)?
            .expert(expert)?
            .cache_key();
        if let Some(view) = staged.get(&key) {
            view.matvec_into(&s.cpu_gate, &mut s.cpu_expert)?;
        } else {
            let lease = if admit {
                self.plan.source.expert_cached(&e.down, expert)?
            } else {
                self.plan
                    .source
                    .expert_cached_for_prefill(&e.down, expert)?
            };
            lease.tensor().matvec_into(&s.cpu_gate, &mut s.cpu_expert)?;
        }
        let n = self.plan.common.embedding_length;
        s.cpu_outputs[slot * n..(slot + 1) * n].copy_from_slice(&s.cpu_expert);
        self.cpu_experts.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn expert_matrix(
        &self,
        commands: &mut Commands,
        graph: &mut Graph,
        tensor: &Tensor<'_>,
        weight: Arc<Buffer>,
        input: &Arc<Buffer>,
        output: &Arc<Buffer>,
        dummy: &Arc<Buffer>,
        status: &Arc<Buffer>,
        input_offset: usize,
        output_offset: usize,
    ) -> Result<()> {
        let mut push = Push::new(0)
            .n(0, tensor.width())?
            .n(1, tensor.rows())?
            .n(2, tensor.quant_type().id() as usize)?
            .n(3, tensor.quant_type().row_bytes(tensor.width())?)?
            .n(4, 1)?;
        push.0[11] = u32::try_from(input_offset)?;
        push.0[12] = u32::try_from(output_offset)?;
        let groups = commands.groups(tensor.rows())?;
        add(
            commands,
            graph,
            push,
            bind(&weight, input, dummy, output, dummy, dummy, status),
            groups,
            Dynamic::Fixed,
        )
    }
    fn routing_copy(&self, commands: &Commands, s: &QwenGpuSequence) {
        let n = self.plan.common.embedding_length;
        let selected = self.plan.common.experts_per_token * 2;
        commands.compute_to_transfer();
        unsafe {
            self.device.raw.cmd_copy_buffer(
                commands.raw,
                s.scratch["selection"].raw,
                s.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (selected * 4) as u64,
                }],
            );
            self.device.raw.cmd_copy_buffer(
                commands.raw,
                s.scratch["input"].raw,
                s.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: (selected * 4) as u64,
                    size: (n * 4) as u64,
                }],
            );
            self.device.raw.cmd_copy_buffer(
                commands.raw,
                s.scratch["status"].raw,
                s.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: ((selected + n) * 4) as u64,
                    size: 4,
                }],
            );
        }
        commands.transfer_to_host();
    }
    fn read_routing(&self, s: &mut QwenGpuSequence) -> Result<()> {
        let n = self.plan.common.embedding_length;
        let selected = self.plan.common.experts_per_token * 2;
        let mut status = [0u8; 4];
        s.readback.read((selected + n) * 4, &mut status)?;
        ensure!(
            u32::from_ne_bytes(status) == 0,
            "Qwen GPU state computation produced a non-finite intermediate"
        );
        s.readback.read(0, bytes_of_f32_mut(&mut s.cpu_routes))?;
        s.readback
            .read(selected * 4, bytes_of_f32_mut(&mut s.cpu_input))?;
        s.selected.clear();
        for pair in s.cpu_routes.chunks_exact(2) {
            let id = pair[0];
            let weight = pair[1];
            ensure!(
                id.is_finite()
                    && id >= 0.0
                    && id.fract() == 0.0
                    && (id as usize) < self.plan.common.expert_count
                    && weight.is_finite()
                    && weight >= 0.0,
                "invalid GPU expert routing result"
            );
            let id = id as usize;
            ensure!(
                !s.selected.iter().any(|(previous, _)| *previous == id),
                "GPU router selected an expert twice"
            );
            s.selected.push((id, weight));
        }
        ensure!(
            s.cpu_input.iter().all(|v| v.is_finite()),
            "non-finite GPU FFN input"
        );
        self.device
            .readbacks
            .fetch_add(((selected + n) * 4 + 4) as u64, Ordering::Relaxed);
        Ok(())
    }
    fn execute_token(
        &self,
        s: &mut QwenGpuSequence,
        token: u32,
        project: bool,
        admit: bool,
        out: &mut Step,
    ) -> Result<()> {
        let c = &self.plan.common;
        let n = c.embedding_length;
        let ff = c.expert_feed_forward_length;
        self.plan
            .source
            .tensor("token_embd.weight")?
            .read_row(token as usize, &mut s.cpu_embeddings[..n])?;
        self.plan
            .hash_embedding(s.previous, token, &mut s.cpu_embeddings[n..])?;
        s.upload.write(0, bytes_of_f32(&s.cpu_embeddings))?;
        s.initial_commands.begin()?;
        unsafe {
            self.device.raw.cmd_fill_buffer(
                s.initial_commands.raw,
                s.scratch["status"].raw,
                0,
                4,
                0,
            );
            self.device.raw.cmd_copy_buffer(
                s.initial_commands.raw,
                s.upload.raw,
                s.scratch["token"].raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (n * 4) as u64,
                }],
            );
            if self.plan.hash.is_some() {
                self.device.raw.cmd_copy_buffer(
                    s.initial_commands.raw,
                    s.upload.raw,
                    s.scratch["embedding"].raw,
                    &[vk::BufferCopy {
                        src_offset: (n * 4) as u64,
                        dst_offset: 0,
                        size: (n * 4) as u64,
                    }],
                );
            }
        }
        s.initial_commands.transfer_to_compute();
        record(&s.initial_commands, &s.initial, s.position);
        record(&s.initial_commands, &s.stages[0], s.position);
        self.routing_copy(&s.initial_commands, s);
        s.initial_commands.submit()?;
        self.read_routing(s)?;
        self.device.uploads.fetch_add(
            (n * 4 * if self.plan.hash.is_some() { 2 } else { 1 }) as u64,
            Ordering::Relaxed,
        );
        for layer in 0..c.block_count {
            let e = &self.plan.layers[layer].experts;
            let mut gpu_jobs = Vec::with_capacity(c.experts_per_token);
            let mut cpu_slots = Vec::new();
            let mut weights = vec![None; c.experts_per_token * 3];
            let mut leases = Vec::new();
            let mut misses = Vec::new();
            for slot in 0..c.experts_per_token {
                let expert = s.selected[slot].0;
                for (matrix, name) in [&e.gate, &e.up, &e.down].into_iter().enumerate() {
                    let index = slot * 3 + matrix;
                    let key = self.plan.source.tensor(name)?.expert(expert)?.cache_key();
                    if let Some(buffer) = self.cache.cached_weight(key, admit)? {
                        weights[index] = Some(buffer);
                    } else {
                        leases.push(if admit {
                            self.plan.source.expert_cached(name, expert)?
                        } else {
                            self.plan.source.expert_cached_for_prefill(name, expert)?
                        });
                        misses.push(index);
                    }
                }
            }
            let views: Vec<_> = leases.iter().map(|lease| lease.tensor()).collect();
            let staged: BTreeMap<_, _> =
                views.iter().map(|view| (view.cache_key(), *view)).collect();
            let mut overlap_used = false;
            // Classify all routed expert triplets before attempting GPU compute.
            // For incomplete triplets, compute the *same exact CPU fallback*
            // while the final packed-weight upload batch is in flight.
            // Buffers returned in "ready" cannot be used until the fence joins.
            let uploaded = self.cache.prepare_batch_with_host_work(
                &views, admit, |ready, transfer_in_flight| {
                    for (&index, buffer) in misses.iter().zip(ready.iter()) {
                        weights[index] = buffer.clone();
                    }
                    for slot in 0..c.experts_per_token {
                        if (0..3).any(|matrix| weights[slot * 3 + matrix].is_none()) {
                            let expert = s.selected[slot].0;
                            self.cpu_expert(layer, expert, admit, s, slot, &staged)?;
                            cpu_slots.push(slot);
                            if transfer_in_flight {
                                overlap_used = true;
                            }
                        }
                    }
                    Ok(())
                }
            )?;
            drop(uploaded);
            if overlap_used {
                self.cpu_gpu_overlap_batches.fetch_add(1, Ordering::Relaxed);
            }
            for slot in 0..c.experts_per_token {
                let expert = s.selected[slot].0;
                let gate = weights[slot * 3].take();
                let up = weights[slot * 3 + 1].take();
                let down = weights[slot * 3 + 2].take();
                if let (Some(g), Some(u), Some(d)) = (gate, up, down) {
                    gpu_jobs.push((slot, expert, g, u, d));
                }
            }
            debug_assert_eq!(gpu_jobs.len() + cpu_slots.len(), c.experts_per_token);
            drop(staged);
            drop(views);
            drop(leases);
            s.expert_commands.reset_descriptors()?;
            let mut graph = Vec::with_capacity(gpu_jobs.len() * 4 + 1);
            let dummy = &s.scratch["dummy"];
            let status = &s.scratch["status"];
            for (slot, expert, gate, up, down) in gpu_jobs {
                let g = self.plan.source.tensor(&e.gate)?.expert(expert)?;
                let u = self.plan.source.tensor(&e.up)?.expert(expert)?;
                let d = self.plan.source.tensor(&e.down)?.expert(expert)?;
                self.expert_matrix(
                    &mut s.expert_commands,
                    &mut graph,
                    &g,
                    gate,
                    &s.scratch["input"],
                    &s.scratch["gate"],
                    dummy,
                    status,
                    0,
                    slot * ff,
                )?;
                self.expert_matrix(
                    &mut s.expert_commands,
                    &mut graph,
                    &u,
                    up,
                    &s.scratch["input"],
                    &s.scratch["up"],
                    dummy,
                    status,
                    0,
                    slot * ff,
                )?;
                let mut push = Push::new(6).n(0, ff)?;
                push.0[10] = u32::try_from(slot * ff)?;
                push.0[11] = u32::try_from(slot * ff)?;
                push.0[12] = u32::try_from(slot * ff)?;
                let groups = elements(&s.expert_commands, ff)?;
                add(
                    &mut s.expert_commands,
                    &mut graph,
                    push,
                    bind(
                        dummy,
                        &s.scratch["gate"],
                        &s.scratch["up"],
                        &s.scratch["gate"],
                        dummy,
                        dummy,
                        status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
                self.expert_matrix(
                    &mut s.expert_commands,
                    &mut graph,
                    &d,
                    down,
                    &s.scratch["gate"],
                    &s.scratch["expertout"],
                    dummy,
                    status,
                    slot * ff,
                    slot * n,
                )?;
                self.gpu_experts.fetch_add(1, Ordering::Relaxed);
            }
            let groups = elements(&s.expert_commands, n)?;
            add(
                &mut s.expert_commands,
                &mut graph,
                Push::new(10).n(0, n)?.n(1, c.experts_per_token)?,
                bind(
                    dummy,
                    &s.scratch["expertout"],
                    dummy,
                    &s.scratch["update"],
                    dummy,
                    &s.scratch["selection"],
                    status,
                ),
                groups,
                Dynamic::Fixed,
            )?;
            if !cpu_slots.is_empty() {
                s.expert_upload.write(0, bytes_of_f32(&s.cpu_outputs))?;
            }
            s.expert_commands.begin()?;
            for slot in &cpu_slots {
                unsafe {
                    self.device.raw.cmd_copy_buffer(
                        s.expert_commands.raw,
                        s.expert_upload.raw,
                        s.scratch["expertout"].raw,
                        &[vk::BufferCopy {
                            src_offset: (slot * n * 4) as u64,
                            dst_offset: (slot * n * 4) as u64,
                            size: (n * 4) as u64,
                        }],
                    )
                };
            }
            if !cpu_slots.is_empty() {
                s.expert_commands.transfer_to_compute();
                self.device
                    .uploads
                    .fetch_add((cpu_slots.len() * n * 4) as u64, Ordering::Relaxed);
            }
            record(&s.expert_commands, &graph, s.position);
            record(&s.expert_commands, &s.shared[layer], s.position);
            record(&s.expert_commands, &s.finish[layer], s.position);
            let final_layer = layer + 1 == c.block_count;
            if !final_layer {
                record(&s.expert_commands, &s.stages[layer + 1], s.position);
                self.routing_copy(&s.expert_commands, s);
            } else {
                if project {
                    record(&s.expert_commands, &s.output, s.position);
                }
                s.expert_commands.compute_to_transfer();
                unsafe {
                    if project {
                        self.device.raw.cmd_copy_buffer(
                            s.expert_commands.raw,
                            s.scratch["logits"].raw,
                            s.readback.raw,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: (c.vocab_size * 4) as u64,
                            }],
                        );
                        self.device.raw.cmd_copy_buffer(
                            s.expert_commands.raw,
                            s.scratch["input"].raw,
                            s.readback.raw,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: (c.vocab_size * 4) as u64,
                                size: (n * 4) as u64,
                            }],
                        );
                    }
                    self.device.raw.cmd_copy_buffer(
                        s.expert_commands.raw,
                        s.scratch["status"].raw,
                        s.readback.raw,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: ((c.vocab_size + n) * 4) as u64,
                            size: 4,
                        }],
                    );
                }
                s.expert_commands.transfer_to_host();
            }
            s.expert_commands.submit()?;
            // These immutable weight leases lived through submission and fence.
            // Dropping them now permits the bounded cache to evict safely.
            drop(graph);
            if !final_layer {
                self.read_routing(s)?;
            } else {
                let mut status = [0u8; 4];
                s.readback.read((c.vocab_size + n) * 4, &mut status)?;
                ensure!(
                    u32::from_ne_bytes(status) == 0,
                    "Qwen GPU forward produced a non-finite intermediate"
                );
                if project {
                    out.logits.resize(c.vocab_size, 0.0);
                    out.hidden.resize(n, 0.0);
                    s.readback.read(0, bytes_of_f32_mut(&mut out.logits))?;
                    s.readback
                        .read(c.vocab_size * 4, bytes_of_f32_mut(&mut out.hidden))?;
                    ensure!(
                        out.logits.iter().chain(&out.hidden).all(|v| v.is_finite()),
                        "Qwen GPU logits/hidden contain non-finite values"
                    );
                    self.device
                        .readbacks
                        .fetch_add(((c.vocab_size + n) * 4 + 4) as u64, Ordering::Relaxed);
                } else {
                    self.device.readbacks.fetch_add(4, Ordering::Relaxed);
                }
            }
        }
        if self.plan.hash.is_some() {
            s.previous = [token as i64, s.previous[0]];
        }
        s.position += 1;
        self.tokens.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn validate(&self, s: &QwenGpuSequence, tokens: &[u32]) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.owner, &s.owner),
            "Qwen GPU sequence belongs to another model"
        );
        ensure!(
            !s.poisoned,
            "Qwen GPU sequence requires reset after an execution error"
        );
        ensure!(
            !tokens.is_empty()
                && s.position
                    .checked_add(tokens.len())
                    .is_some_and(|end| end <= self.context()),
            "Qwen GPU token/context bounds exceeded"
        );
        ensure!(
            tokens.iter().all(|t| (*t as usize) < self.vocab_size()),
            "Qwen GPU token ID outside vocabulary"
        );
        Ok(())
    }
    pub fn step(&self, s: &mut QwenGpuSequence, token: u32) -> Result<Step> {
        let mut out = Step::default();
        self.step_into(s, token, &mut out)?;
        Ok(out)
    }
    pub fn step_into(&self, s: &mut QwenGpuSequence, token: u32, out: &mut Step) -> Result<()> {
        self.validate(s, &[token])?;
        let result = self.execute_token(s, token, true, true, out);
        if result.is_err() {
            s.poisoned = true;
        }
        result
    }
    pub fn prefill(&self, s: &mut QwenGpuSequence, tokens: &[u32]) -> Result<Step> {
        let mut out = Step::default();
        self.prefill_into(s, tokens, &mut out)?;
        Ok(out)
    }
    pub fn prefill_into(
        &self,
        s: &mut QwenGpuSequence,
        tokens: &[u32],
        out: &mut Step,
    ) -> Result<()> {
        self.validate(s, tokens)?;
        for (i, &token) in tokens.iter().enumerate() {
            if let Err(e) = self.execute_token(s, token, i + 1 == tokens.len(), false, out) {
                s.poisoned = true;
                return Err(e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 3e-5 + 5e-5 * b.abs(),
                "element{i}: {a} != {b}"
            );
        }
    }
    #[test]
    #[ignore = "requires explicit Vulkan shader execution environment"]
    fn software_vulkan_qwen_budget_pressure_retains_every_expert() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen38.gguf");
        let source = Arc::new(Gguf::open(path).unwrap());
        let cpu = Model::load(Arc::clone(&source), 32).unwrap();
        let model = QwenGpuModel::load(
            source,
            32,
            DeviceOptions {
                device_index: None,
                allow_software: true,
                memory_budget_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();
        let mut gpu_sequence = model.new_sequence().unwrap();
        let mut cpu_sequence = cpu.new_sequence();
        // Simulate other correctly-accounted live GPU allocations. The model's
        // fixed weights and sequence are already resident; expert admission has
        // too little room for its bounded staging block and must compute every
        // selected contribution on CPU without dropping one or growing the cap.
        let room = model.device.budget - model.device.reserved();
        assert!(room > 128 * 1024);
        let pressure = model.device.storage((room - 64 * 1024) as usize).unwrap();
        let tokens = [2, 19, 77, 5];
        for token in tokens {
            let expected = cpu.step(&mut cpu_sequence, token).unwrap();
            let actual = model.step(&mut gpu_sequence, token).unwrap();
            close(&actual.logits, &expected.logits);
            close(&actual.hidden, &expected.hidden);
        }
        let stats = model.memory_stats();
        assert_eq!(
            stats["cpu_expert_fallbacks"].as_u64(),
            Some(
                (tokens.len() * model.config().block_count * model.config().experts_per_token)
                    as u64
            )
        );
        assert_eq!(stats["gpu_expert_executions"].as_u64(), Some(0));
        assert_eq!(stats["selected_experts_omitted"].as_u64(), Some(0));
        assert!(stats["live_allocation_bytes"].as_u64().unwrap() <= model.device.budget);
        drop(pressure);
        gpu_sequence.reset().unwrap();
        cpu_sequence.reset();
        let before = model.device.submissions.load(Ordering::Relaxed);
        let expected = cpu.step(&mut cpu_sequence, 2).unwrap();
        let actual = model.step(&mut gpu_sequence, 2).unwrap();
        close(&actual.logits, &expected.logits);
        close(&actual.hidden, &expected.hidden);
        let cold_submissions = model.device.submissions.load(Ordering::Relaxed) - before;
        // Tiny cold expert matrices fit one staging window per layer, so their
        // upload fence count is bounded by layers, not gate/up/down matrix count.
        assert!(
            cold_submissions <= (2 * model.config().block_count + 1) as u64,
            "cold submissions {cold_submissions}"
        );
        let stats = model.memory_stats();
        assert!(stats["gpu_expert_executions"].as_u64().unwrap() > 0);
        assert_eq!(
            stats["batched_expert_upload_calls"].as_u64(),
            Some(model.config().block_count as u64)
        );
        let cached = stats["cached_packed_weight_bytes"].as_u64().unwrap();
        let evictions = stats["cache_evictions"].as_u64().unwrap();
        let recencies = model.cache.cache_recencies();
        gpu_sequence.reset().unwrap();
        model.prefill(&mut gpu_sequence, &[11, 12, 13]).unwrap();
        let after = model.memory_stats();
        assert_eq!(after["cached_packed_weight_bytes"].as_u64(), Some(cached));
        assert_eq!(after["cache_evictions"].as_u64(), Some(evictions));
        assert_eq!(
            model.cache.cache_recencies(),
            recencies,
            "prefill must preserve decode LRU ordering"
        );
        println!(
            "budget-retention: {} exact CPU expert fallbacks; cold GPU token {} submissions; owned allocation limit {} bytes",
            stats["cpu_expert_fallbacks"], cold_submissions, model.device.budget
        );
    }

    #[test]
    #[ignore = "requires explicit Vulkan shader execution environment"]
    fn software_vulkan_qwen_128k_context_short_prefix() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen38.gguf");
        let mut bytes = std::fs::read(fixture).unwrap();
        let key = b"qwen4exp.context_length";
        let offsets: Vec<_> = bytes
            .windows(key.len())
            .enumerate()
            .filter_map(|(i, window)| (window == key).then_some(i))
            .collect();
        assert_eq!(offsets.len(), 1);
        let offset = offsets[0];
        assert_eq!(
            &bytes[offset - 8..offset],
            &(key.len() as u64).to_le_bytes()
        );
        let value = offset + key.len();
        assert_eq!(&bytes[value..value + 4], &4u32.to_le_bytes());
        // Only the synthetic training-context metadata changes. This tests
        // allocation/dispatch capacity, not pretrained long-context quality.
        let context = 128 * 1024usize;
        bytes[value + 4..value + 8].copy_from_slice(&(context as u32).to_le_bytes());
        struct Temporary(std::path::PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let temporary = Temporary(
            std::env::temp_dir().join(format!("forge-qwen38-128k-{}.gguf", std::process::id())),
        );
        std::fs::write(&temporary.0, bytes).unwrap();
        let source = Arc::new(Gguf::open(&temporary.0).unwrap());
        let cpu = Model::load(Arc::clone(&source), context).unwrap();
        let model = QwenGpuModel::load(
            source,
            context,
            DeviceOptions {
                device_index: None,
                allow_software: true,
                memory_budget_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();
        let mut actual = model.new_sequence().unwrap();
        let mut expected = cpu.new_sequence();
        let capacity = context.min(model.plan.architecture.indexer_token_budget + 3);
        assert_eq!(
            actual.scratch["scores"].size,
            (4 * model.config().head_count * capacity) as u64
        );
        assert!(capacity < context);
        for token in [
            2, 19, 77, 5, 3, 11, 71, 8, 99, 101, 12, 13, 7, 17, 23, 1, 9, 22, 15, 6,
        ] {
            let a = model.step(&mut actual, token).unwrap();
            let b = cpu.step(&mut expected, token).unwrap();
            close(&a.logits, &b.logits);
            close(&a.hidden, &b.hidden);
        }
        assert_eq!(actual.position(), 20);
        assert_eq!(model.context(), context);
        let stats = model.memory_stats();
        assert_eq!(stats["cpu_expert_fallbacks"].as_u64(), Some(0));
        assert!(stats["live_allocation_bytes"].as_u64().unwrap() <= model.device.budget);
        println!(
            "synthetic 128K context: 20-token full-logit/hidden prefix parity, {} selected-cell score capacity, {} live allocation bytes; no long-context quality claim",
            capacity, stats["live_allocation_bytes"]
        );
    }
}
