//! Fully resident ordinary Qwen3/Qwen3MoE execution. Each request owns its GPU
//! state and preallocated descriptors; token operations stay on the device.
use super::context::{
    Buffer, Commands, Device, DeviceOptions, Operation, Push, bytes_of_f32, bytes_of_f32_mut,
    float_bytes,
};
use crate::{
    gguf::Gguf,
    model::{Config, Model, Step},
};
use anyhow::{Context, Result, ensure};
use ash::vk;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::Ordering},
};

#[derive(Clone, Debug)]
struct Shape {
    width: usize,
    rows: usize,
    experts: usize,
    ty: u32,
    row_bytes: usize,
}
#[derive(Debug)]
pub struct GpuModel {
    device: Arc<Device>,
    config: Config,
    weights: BTreeMap<String, Arc<Buffer>>,
    shapes: BTreeMap<String, Shape>,
    owner: Arc<()>,
    weight_bytes: usize,
    output: String,
}
#[derive(Clone, Copy, Debug)]
enum Dynamic {
    Fixed,
    Embedding,
    Rope,
    Store,
    Scores,
    Attention,
}
#[derive(Debug)]
struct GraphOp {
    operation: Operation,
    dynamic: Dynamic,
}

#[derive(Debug)]
pub struct GpuSequence {
    // Commands are destroyed before resource ownership, including on errors.
    commands: Commands,
    owner: Arc<()>,
    position: usize,
    poisoned: bool,
    forward: Vec<GraphOp>,
    project: Vec<GraphOp>,
    logits: Arc<Buffer>,
    hidden: Arc<Buffer>,
    status: Arc<Buffer>,
    readback: Arc<Buffer>,
    keys: Vec<Arc<Buffer>>,
    values: Vec<Arc<Buffer>>,
    kv_width: usize,
    vocabulary: usize,
    embedding: usize,
}
impl GpuSequence {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) -> Result<()> {
        ensure!(
            !self.poisoned,
            "GPU sequence failed during submission and must be recreated"
        );
        self.position = 0;
        Ok(())
    }
    /// Diagnostic snapshot of the initialized F32 attention-cache prefix. Normal
    /// generation does not transfer KV state back to the host.
    pub fn read_kv(&mut self, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(!self.poisoned, "GPU sequence is poisoned");
        let key = self.keys.get(layer).context("KV layer outside model")?;
        let value = &self.values[layer];
        let elements = self
            .position
            .checked_mul(self.kv_width)
            .context("KV snapshot size overflow")?;
        if elements == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let bytes = float_bytes(elements)?;
        let host = self
            .commands
            .device
            .host(bytes.checked_mul(2).context("KV snapshot byte overflow")?)?;
        self.commands.begin()?;
        self.commands.compute_to_transfer();
        unsafe {
            self.commands.device.raw.cmd_copy_buffer(
                self.commands.raw,
                key.raw,
                host.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                }],
            );
            self.commands.device.raw.cmd_copy_buffer(
                self.commands.raw,
                value.raw,
                host.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: bytes as u64,
                    size: bytes as u64,
                }],
            );
        }
        self.commands.transfer_to_host();
        self.commands.submit()?;
        let mut k = vec![0.0; elements];
        let mut v = vec![0.0; elements];
        host.read(0, bytes_of_f32_mut(&mut k))?;
        host.read(bytes, bytes_of_f32_mut(&mut v))?;
        self.commands
            .device
            .readbacks
            .fetch_add((bytes * 2) as u64, Ordering::Relaxed);
        Ok((k, v))
    }
}

fn bindings(
    dummy: &Arc<Buffer>,
    weight: &Arc<Buffer>,
    a: &Arc<Buffer>,
    b: &Arc<Buffer>,
    c: &Arc<Buffer>,
    d: &Arc<Buffer>,
    e: &Arc<Buffer>,
    status: &Arc<Buffer>,
) -> [Arc<Buffer>; 7] {
    let _ = dummy;
    [
        Arc::clone(weight),
        Arc::clone(a),
        Arc::clone(b),
        Arc::clone(c),
        Arc::clone(d),
        Arc::clone(e),
        Arc::clone(status),
    ]
}
fn append(
    commands: &mut Commands,
    graph: &mut Vec<GraphOp>,
    push: Push,
    buffers: [Arc<Buffer>; 7],
    groups: [usize; 3],
    dynamic: Dynamic,
) -> Result<()> {
    graph.push(GraphOp {
        operation: commands.operation(push, buffers, groups)?,
        dynamic,
    });
    Ok(())
}
fn elements_groups(commands: &Commands, count: usize) -> Result<[usize; 3]> {
    commands.groups(count.div_ceil(128))
}

impl GpuModel {
    pub fn load(source: Arc<Gguf>, context: usize, options: DeviceOptions) -> Result<Self> {
        // Reuse the independent loader's geometry/semantic validation, never
        // inference or weights from the compatibility engine.
        let checked = Model::load(Arc::clone(&source), context)?;
        let config = checked.config().clone();
        ensure!(
            matches!(config.architecture.as_str(), "qwen3" | "qwen3moe"),
            "resident Vulkan graph supports ordinary qwen3/qwen3moe; use the explicit hybrid engine for extended architectures"
        );
        drop(checked);
        let device = Device::new(&options)?;
        ensure!(
            context <= device.properties.limits.max_compute_work_group_count[0] as usize,
            "context exceeds this graph's Vulkan attention dispatch limit"
        );
        ensure!(
            config.head_count <= device.properties.limits.max_compute_work_group_count[1] as usize,
            "attention head count exceeds dispatch limit"
        );
        ensure!(
            config.experts_per_token.max(1)
                <= device.properties.limits.max_compute_work_group_count[2] as usize,
            "selected expert count exceeds dispatch limit"
        );
        let mut weights = BTreeMap::new();
        let mut shapes = BTreeMap::new();
        let mut weight_bytes = 0usize;
        for (name, info) in &source.tensors {
            let tensor = source.tensor(name)?;
            ensure!(
                super::supports_quant(tensor.quant_type().id()),
                "resident Vulkan tensor {name} uses unsupported GGML type {}",
                tensor.quant_type().id()
            );
            ensure!(
                tensor.bytes().len() <= u32::MAX as usize,
                "tensor {name} exceeds the 32-bit packed-address bound"
            );
            let shape = Shape {
                width: tensor.width(),
                rows: if info.dims.len() == 3 {
                    info.dims[1] as usize
                } else {
                    tensor.rows()
                },
                experts: if info.dims.len() == 3 {
                    info.dims[2] as usize
                } else {
                    1
                },
                ty: tensor.quant_type().id(),
                row_bytes: tensor.quant_type().row_bytes(tensor.width())?,
            };
            let uploaded = if info.dims.len() == 1 {
                let values = tensor.row(0)?;
                device.upload(bytes_of_f32(&values))?
            } else {
                device.upload(tensor.bytes())?
            };
            weight_bytes = weight_bytes
                .checked_add(uploaded.size as usize)
                .context("resident weight size overflow")?;
            shapes.insert(name.clone(), shape);
            weights.insert(name.clone(), uploaded);
        }
        let output = if source.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
        .to_string();
        Ok(Self {
            device,
            config,
            weights,
            shapes,
            owner: Arc::new(()),
            weight_bytes,
            output,
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn context(&self) -> usize {
        self.config.context_length
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    pub fn adapter_info(&self) -> Value {
        self.device.info()
    }
    pub fn memory_stats(&self) -> Value {
        let mut stats = self.device.stats();
        stats["resident_weight_payload_bytes"] = json!(self.weight_bytes);
        stats["kv_precision"] = json!("F32");
        stats["execution"] = json!("resident_gpu_graph");
        stats["cpu_projection_fallbacks"] = json!(0);
        stats
    }
    fn weight(&self, name: &str) -> Result<&Arc<Buffer>> {
        self.weights
            .get(name)
            .with_context(|| format!("missing resident weight {name}"))
    }
    fn shape(&self, name: &str) -> Result<&Shape> {
        self.shapes
            .get(name)
            .with_context(|| format!("missing resident shape {name}"))
    }
    fn matrix(
        &self,
        commands: &mut Commands,
        graph: &mut Vec<GraphOp>,
        name: &str,
        input: &Arc<Buffer>,
        output: &Arc<Buffer>,
        selection: &Arc<Buffer>,
        status: &Arc<Buffer>,
        dummy: &Arc<Buffer>,
        selected: bool,
        individual_inputs: bool,
    ) -> Result<()> {
        let s = self.shape(name)?;
        let mut groups = commands.groups(s.rows)?;
        groups[2] = if selected {
            self.config.experts_per_token
        } else {
            1
        };
        let p = Push::new(0)
            .n(0, s.width)?
            .n(1, s.rows)?
            .n(2, s.ty as usize)?
            .n(3, s.row_bytes)?
            .n(4, s.experts)?
            .flags(u32::from(selected) | u32::from(individual_inputs) * 2);
        append(
            commands,
            graph,
            p,
            bindings(
                dummy,
                self.weight(name)?,
                input,
                dummy,
                output,
                dummy,
                selection,
                status,
            ),
            groups,
            Dynamic::Fixed,
        )
    }
    pub fn new_sequence(&self) -> Result<GpuSequence> {
        let c = &self.config;
        let mut commands = Commands::new(
            Arc::clone(&self.device),
            c.block_count
                .checked_mul(24)
                .and_then(|n| n.checked_add(8))
                .context("graph descriptor bound overflow")?,
        )?;
        let storage = |n: usize| self.device.storage(float_bytes(n)?);
        let dummy = storage(1)?;
        let status = storage(1)?;
        let residual = storage(c.embedding_length)?;
        let normalized = storage(c.embedding_length)?;
        let query = storage(c.head_count * c.head_dim)?;
        let key = storage(c.head_count_kv * c.head_dim)?;
        let value = storage(c.head_count_kv * c.head_dim)?;
        let attention = storage(c.head_count * c.head_dim)?;
        let projected = storage(c.embedding_length)?;
        let ffn = if c.expert_count == 0 {
            c.feed_forward_length
        } else {
            c.expert_feed_forward_length
        };
        let slots = c.experts_per_token.max(1);
        let gate = storage(
            ffn.checked_mul(slots)
                .context("selected expert scratch overflow")?,
        )?;
        let up = storage(ffn * slots)?;
        let expert_output = storage(
            c.embedding_length
                .checked_mul(slots)
                .context("expert output scratch overflow")?,
        )?;
        let router = storage(c.expert_count.max(1))?;
        let probabilities = storage(c.expert_count.max(1))?;
        let selection = storage(slots * 2)?;
        let scores = storage(
            c.head_count
                .checked_mul(c.context_length)
                .context("attention scratch overflow")?,
        )?;
        let logits = storage(c.vocab_size)?;
        let hidden = storage(c.embedding_length)?;
        let readback = self.device.host(float_bytes(
            c.vocab_size
                .checked_add(c.embedding_length)
                .and_then(|n| n.checked_add(1))
                .context("readback overflow")?,
        )?)?;
        let kv_width = c.head_count_kv * c.head_dim;
        let kv_count = kv_width
            .checked_mul(c.context_length)
            .context("KV capacity overflow")?;
        let mut keys = Vec::with_capacity(c.block_count);
        let mut values = Vec::with_capacity(c.block_count);
        for _ in 0..c.block_count {
            keys.push(storage(kv_count)?);
            values.push(storage(kv_count)?);
        }
        let mut forward = Vec::new();
        let mut project = Vec::new();
        let emb = self.shape("token_embd.weight")?;
        let groups = elements_groups(&commands, c.embedding_length)?;
        append(
            &mut commands,
            &mut forward,
            Push::new(9)
                .n(0, c.embedding_length)?
                .n(2, emb.ty as usize)?
                .n(3, emb.row_bytes)?,
            bindings(
                &dummy,
                self.weight("token_embd.weight")?,
                &dummy,
                &dummy,
                &residual,
                &dummy,
                &dummy,
                &status,
            ),
            groups,
            Dynamic::Embedding,
        )?;
        for layer in 0..c.block_count {
            let name = |s: &str| format!("blk.{layer}.{s}.weight");
            let norm_groups = commands.groups(1)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(1)
                    .n(0, c.embedding_length)?
                    .n(1, 1)?
                    .f(0, c.rms_epsilon),
                bindings(
                    &dummy,
                    self.weight(&name("attn_norm"))?,
                    &residual,
                    &dummy,
                    &normalized,
                    &dummy,
                    &dummy,
                    &status,
                ),
                norm_groups,
                Dynamic::Fixed,
            )?;
            self.matrix(
                &mut commands,
                &mut forward,
                &name("attn_q"),
                &normalized,
                &query,
                &selection,
                &status,
                &dummy,
                false,
                false,
            )?;
            self.matrix(
                &mut commands,
                &mut forward,
                &name("attn_k"),
                &normalized,
                &key,
                &selection,
                &status,
                &dummy,
                false,
                false,
            )?;
            self.matrix(
                &mut commands,
                &mut forward,
                &name("attn_v"),
                &normalized,
                &value,
                &selection,
                &status,
                &dummy,
                false,
                false,
            )?;
            for (buffer, heads, norm_name) in [
                (&query, c.head_count, "attn_q_norm"),
                (&key, c.head_count_kv, "attn_k_norm"),
            ] {
                let groups = commands.groups(heads)?;
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(1)
                        .n(0, c.head_dim)?
                        .n(1, heads)?
                        .f(0, c.rms_epsilon),
                    bindings(
                        &dummy,
                        self.weight(&name(norm_name))?,
                        buffer,
                        &dummy,
                        buffer,
                        &dummy,
                        &dummy,
                        &status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
                let groups = elements_groups(&commands, heads * (c.head_dim / 2))?;
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(2)
                        .n(0, c.head_dim)?
                        .n(1, heads)?
                        .f(0, c.rope_base),
                    bindings(
                        &dummy, &dummy, buffer, &dummy, buffer, &dummy, &dummy, &status,
                    ),
                    groups,
                    Dynamic::Rope,
                )?;
            }
            let groups = elements_groups(&commands, kv_width)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(3).n(0, kv_width)?,
                bindings(
                    &dummy,
                    &dummy,
                    &key,
                    &value,
                    &keys[layer],
                    &values[layer],
                    &dummy,
                    &status,
                ),
                groups,
                Dynamic::Store,
            )?;
            append(
                &mut commands,
                &mut forward,
                Push::new(4)
                    .n(0, 1)?
                    .n(1, c.head_count)?
                    .n(2, c.head_count_kv)?
                    .n(3, c.head_dim)?
                    .n(4, c.context_length)?,
                bindings(
                    &dummy,
                    &dummy,
                    &query,
                    &keys[layer],
                    &scores,
                    &dummy,
                    &dummy,
                    &status,
                ),
                [1, c.head_count, 1],
                Dynamic::Scores,
            )?;
            let groups = commands.groups(c.head_count)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(5)
                    .n(0, 1)?
                    .n(1, c.head_count)?
                    .n(2, c.head_count_kv)?
                    .n(3, c.head_dim)?
                    .n(4, c.context_length)?,
                bindings(
                    &dummy,
                    &dummy,
                    &scores,
                    &values[layer],
                    &attention,
                    &dummy,
                    &dummy,
                    &status,
                ),
                groups,
                Dynamic::Attention,
            )?;
            self.matrix(
                &mut commands,
                &mut forward,
                &name("attn_output"),
                &attention,
                &projected,
                &selection,
                &status,
                &dummy,
                false,
                false,
            )?;
            let groups = elements_groups(&commands, c.embedding_length)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(7).n(0, c.embedding_length)?,
                bindings(
                    &dummy, &dummy, &residual, &projected, &residual, &dummy, &dummy, &status,
                ),
                groups,
                Dynamic::Fixed,
            )?;
            let groups = commands.groups(1)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(1)
                    .n(0, c.embedding_length)?
                    .n(1, 1)?
                    .f(0, c.rms_epsilon),
                bindings(
                    &dummy,
                    self.weight(&name("ffn_norm"))?,
                    &residual,
                    &dummy,
                    &normalized,
                    &dummy,
                    &dummy,
                    &status,
                ),
                groups,
                Dynamic::Fixed,
            )?;
            if c.expert_count > 0 {
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_gate_inp"),
                    &normalized,
                    &router,
                    &selection,
                    &status,
                    &dummy,
                    false,
                    false,
                )?;
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(8)
                        .n(0, c.expert_count)?
                        .n(1, c.experts_per_token)?
                        .f(0, c.expert_weights_scale)
                        .flags(u32::from(c.normalize_expert_weights)),
                    bindings(
                        &dummy,
                        &dummy,
                        &router,
                        &dummy,
                        &probabilities,
                        &dummy,
                        &selection,
                        &status,
                    ),
                    [1, 1, 1],
                    Dynamic::Fixed,
                )?;
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_gate_exps"),
                    &normalized,
                    &gate,
                    &selection,
                    &status,
                    &dummy,
                    true,
                    false,
                )?;
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_up_exps"),
                    &normalized,
                    &up,
                    &selection,
                    &status,
                    &dummy,
                    true,
                    false,
                )?;
            } else {
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_gate"),
                    &normalized,
                    &gate,
                    &selection,
                    &status,
                    &dummy,
                    false,
                    false,
                )?;
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_up"),
                    &normalized,
                    &up,
                    &selection,
                    &status,
                    &dummy,
                    false,
                    false,
                )?;
            }
            let groups = elements_groups(&commands, ffn * slots)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(6).n(0, ffn * slots)?,
                bindings(&dummy, &dummy, &gate, &up, &gate, &dummy, &dummy, &status),
                groups,
                Dynamic::Fixed,
            )?;
            if c.expert_count > 0 {
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_down_exps"),
                    &gate,
                    &expert_output,
                    &selection,
                    &status,
                    &dummy,
                    true,
                    true,
                )?;
                let groups = elements_groups(&commands, c.embedding_length)?;
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(10).n(0, c.embedding_length)?.n(1, slots)?,
                    bindings(
                        &dummy,
                        &dummy,
                        &expert_output,
                        &dummy,
                        &projected,
                        &dummy,
                        &selection,
                        &status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
            } else {
                self.matrix(
                    &mut commands,
                    &mut forward,
                    &name("ffn_down"),
                    &gate,
                    &projected,
                    &selection,
                    &status,
                    &dummy,
                    false,
                    false,
                )?;
            }
            let groups = elements_groups(&commands, c.embedding_length)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(7).n(0, c.embedding_length)?,
                bindings(
                    &dummy, &dummy, &residual, &projected, &residual, &dummy, &dummy, &status,
                ),
                groups,
                Dynamic::Fixed,
            )?;
        }
        let groups = commands.groups(1)?;
        append(
            &mut commands,
            &mut project,
            Push::new(1)
                .n(0, c.embedding_length)?
                .n(1, 1)?
                .f(0, c.rms_epsilon),
            bindings(
                &dummy,
                self.weight("output_norm.weight")?,
                &residual,
                &dummy,
                &hidden,
                &dummy,
                &dummy,
                &status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        self.matrix(
            &mut commands,
            &mut project,
            &self.output,
            &hidden,
            &logits,
            &selection,
            &status,
            &dummy,
            false,
            false,
        )?;
        Ok(GpuSequence {
            commands,
            owner: Arc::clone(&self.owner),
            position: 0,
            poisoned: false,
            forward,
            project,
            logits,
            hidden,
            status,
            readback,
            keys,
            values,
            kv_width,
            vocabulary: c.vocab_size,
            embedding: c.embedding_length,
        })
    }
    pub fn step(&self, sequence: &mut GpuSequence, token: u32) -> Result<Step> {
        let mut step = Step::default();
        self.step_into(sequence, token, &mut step)?;
        Ok(step)
    }
    pub fn step_into(&self, sequence: &mut GpuSequence, token: u32, out: &mut Step) -> Result<()> {
        self.prefill_into(sequence, &[token], out)
    }
    pub fn prefill(&self, sequence: &mut GpuSequence, tokens: &[u32]) -> Result<Step> {
        let mut step = Step::default();
        self.prefill_into(sequence, tokens, &mut step)?;
        Ok(step)
    }
    pub fn prefill_into(
        &self,
        sequence: &mut GpuSequence,
        tokens: &[u32],
        out: &mut Step,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.owner, &sequence.owner),
            "GPU sequence belongs to a different model"
        );
        ensure!(
            !sequence.poisoned,
            "GPU sequence failed during submission and must be recreated"
        );
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        ensure!(
            sequence
                .position
                .checked_add(tokens.len())
                .is_some_and(|end| end <= self.context()),
            "GPU context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|t| (*t as usize) < self.vocab_size()),
            "GPU input token is outside vocabulary"
        );
        // Bound each recorded command buffer while keeping intermediate token
        // activations and KV resident. A failed chunk poisons this sequence;
        // it cannot be mistaken for an unchanged reusable request state.
        let start = sequence.position;
        for (chunk_index, chunk) in tokens.chunks(32).enumerate() {
            let last = (chunk_index + 1) * 32 >= tokens.len();
            let result = self.execute_chunk(sequence, chunk, last, out);
            if let Err(error) = result {
                sequence.position = start;
                sequence.poisoned = true;
                return Err(error);
            }
            sequence.position += chunk.len();
        }
        Ok(())
    }
    fn execute_chunk(
        &self,
        s: &mut GpuSequence,
        tokens: &[u32],
        last: bool,
        out: &mut Step,
    ) -> Result<()> {
        s.commands.begin()?;
        unsafe {
            self.device
                .raw
                .cmd_fill_buffer(s.commands.raw, s.status.raw, 0, 4, 0)
        };
        s.commands.transfer_to_compute();
        for (offset, &token) in tokens.iter().enumerate() {
            let position = s.position + offset;
            for op in &s.forward {
                let mut push = op.operation.push.clone();
                let mut groups = op.operation.groups;
                match op.dynamic {
                    Dynamic::Fixed => {}
                    Dynamic::Embedding => push.0[2] = token,
                    Dynamic::Rope => push.0[3] = position as u32,
                    Dynamic::Store => push.0[2] = position as u32,
                    Dynamic::Scores => {
                        push.0[1] = (position + 1) as u32;
                        groups[0] = (position + 1) as u32;
                    }
                    Dynamic::Attention => push.0[1] = (position + 1) as u32,
                }
                s.commands.record(&op.operation, &push, groups);
            }
        }
        if last {
            for op in &s.project {
                s.commands
                    .record(&op.operation, &op.operation.push, op.operation.groups);
            }
        }
        s.commands.compute_to_transfer();
        let logits_bytes = float_bytes(s.vocabulary)?;
        let hidden_bytes = float_bytes(s.embedding)?;
        if last {
            unsafe {
                self.device.raw.cmd_copy_buffer(
                    s.commands.raw,
                    s.logits.raw,
                    s.readback.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: logits_bytes as u64,
                    }],
                );
                self.device.raw.cmd_copy_buffer(
                    s.commands.raw,
                    s.hidden.raw,
                    s.readback.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: logits_bytes as u64,
                        size: hidden_bytes as u64,
                    }],
                );
            }
        }
        unsafe {
            self.device.raw.cmd_copy_buffer(
                s.commands.raw,
                s.status.raw,
                s.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: (logits_bytes + hidden_bytes) as u64,
                    size: 4,
                }],
            );
        }
        s.commands.transfer_to_host();
        s.commands.submit()?;
        let mut status = [0u8; 4];
        s.readback.read(logits_bytes + hidden_bytes, &mut status)?;
        ensure!(
            u32::from_ne_bytes(status) == 0,
            "independent Vulkan computation produced a non-finite intermediate"
        );
        self.device.readbacks.fetch_add(
            if last {
                (logits_bytes + hidden_bytes + 4) as u64
            } else {
                4
            },
            Ordering::Relaxed,
        );
        if last {
            out.logits.resize(s.vocabulary, 0.0);
            out.hidden.resize(s.embedding, 0.0);
            s.readback.read(0, bytes_of_f32_mut(&mut out.logits))?;
            s.readback
                .read(logits_bytes, bytes_of_f32_mut(&mut out.hidden))?;
            ensure!(
                out.logits.iter().chain(&out.hidden).all(|x| x.is_finite()),
                "independent Vulkan output is non-finite"
            );
        }
        Ok(())
    }
}
