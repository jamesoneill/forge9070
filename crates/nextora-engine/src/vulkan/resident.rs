//! Ordinary Qwen3 execution with GPU-resident state and bounded MoE weights.
//! Dense weights stay resident; selected experts use the shared packed cache.
use super::context::{
    Buffer, Commands, Device, DeviceOptions, KvPlacement, Operation, Push, bytes_of_f32,
    bytes_of_f32_mut, float_bytes,
};
use crate::{
    gguf::{Gguf, Tensor, TensorInfo, TensorKey},
    math,
    model::{Config, Model, Step},
};
use anyhow::{Context, Result, ensure};
use ash::vk;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
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
    cache: Option<super::hybrid::GpuMatvecExecutor>,
    source: Arc<Gguf>,
    cpu_experts: AtomicU64,
    gpu_experts: AtomicU64,
    packed_layers: AtomicU64,
    packed_bytes: AtomicU64,
    direct_layers: AtomicU64,
    direct_source_bytes: AtomicU64,
    sparse_profile: SparseProfile,
    routing_poll_budget: std::time::Duration,
    routing_poll_hits: AtomicU64,
    routing_poll_misses: AtomicU64,
    routing_poll_ns: AtomicU64,
    reuse_packed_graphs: bool,
    packed_graph_builds: AtomicU64,
    packed_graph_reuses: AtomicU64,
    config: Config,
    weights: BTreeMap<String, Arc<Buffer>>,
    shapes: BTreeMap<String, Shape>,
    owner: Arc<()>,
    weight_bytes: usize,
    output: String,
    rope_frequencies: Option<Vec<f32>>,
    kv_plan: KvMemoryPlan,
    full_context_minimum_peak: AtomicU64,
    full_context_ram_peak: AtomicU64,
}
/// Optional host-wall timings. GPU submit includes the fence wait, so graph
/// recording/allocation costs remain separate from device execution/latency.
#[derive(Debug, Default)]
struct SparsePhaseProfile {
    tokens: AtomicU64,
    layers: AtomicU64,
    cpu_experts: AtomicU64,
    total_ns: AtomicU64,
    lookup_ns: AtomicU64,
    prepare_ns: AtomicU64,
    graph_ns: AtomicU64,
    cpu_expert_ns: AtomicU64,
    record_ns: AtomicU64,
    submit_ns: AtomicU64,
    commit_ns: AtomicU64,
    routing_ns: AtomicU64,
}
impl SparsePhaseProfile {
    fn stats(&self) -> Value {
        let ms = |value: &AtomicU64| value.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        json!({
            "tokens": self.tokens.load(Ordering::Relaxed),
            "layers": self.layers.load(Ordering::Relaxed),
            "cpu_experts": self.cpu_experts.load(Ordering::Relaxed),
            "total_ms": ms(&self.total_ns),
            "cache_lookup_and_host_weights_ms": ms(&self.lookup_ns),
            "upload_prepare_and_descriptor_reset_ms": ms(&self.prepare_ns),
            "expert_graph_build_ms": ms(&self.graph_ns),
            "cpu_expert_compute_ms": ms(&self.cpu_expert_ns),
            "command_record_ms": ms(&self.record_ns),
            "submit_and_fence_ms": ms(&self.submit_ns),
            "cache_commit_and_lease_release_ms": ms(&self.commit_ns),
            "routing_readback_ms": ms(&self.routing_ns),
        })
    }
}
#[derive(Debug)]
struct SparseProfile {
    enabled: bool,
    decode: SparsePhaseProfile,
    prefill: SparsePhaseProfile,
}
impl SparseProfile {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("NEXTORA_PROFILE_SPARSE").is_some_and(|value| value != "0"),
            decode: SparsePhaseProfile::default(),
            prefill: SparsePhaseProfile::default(),
        }
    }
    fn phase(&self, decode: bool) -> &SparsePhaseProfile {
        if decode { &self.decode } else { &self.prefill }
    }
    fn start(&self) -> Option<std::time::Instant> {
        self.enabled.then(std::time::Instant::now)
    }
    fn finish(start: Option<std::time::Instant>, counter: &AtomicU64) {
        if let Some(start) = start {
            counter.fetch_add(
                start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Dynamic {
    Fixed,
    Embedding,
    Rope,
    Store,
    Scores,
    Attention,
    CompactScores,
    CompactAttention,
    Experts,
}
#[derive(Debug)]
struct GraphOp {
    operation: Operation,
    dynamic: Dynamic,
    continues_barrier_group: bool,
}

#[derive(Debug)]
pub struct GpuSequence {
    // Commands are destroyed before resource ownership, including on errors.
    commands: Commands,
    owner: Arc<()>,
    position: usize,
    poisoned: bool,
    forward: Vec<GraphOp>,
    sparse: Option<SparseSequence>,
    kv_capacity: usize,
    rope: Option<RopeTables>,
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
/// Each sequence owns its current token/chunk table; concurrent requests cannot
/// overwrite one another's angles. All layers share these already-scaled values.
#[derive(Debug)]
struct RopeTables {
    device: Arc<Buffer>,
    upload: Arc<Buffer>,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct QuantScratch {
    packed: Arc<Buffer>,
    scales: Arc<Buffer>,
}

#[derive(Debug)]
struct SparseSequence {
    commands: Commands,
    // This pool only owns immutable packed-graph descriptors. The submitting
    // commands above are destroyed first, before any recorded set can disappear.
    packed_commands: Commands,
    packed_graphs: Vec<Option<Arc<[GraphOp]>>>,
    dynamic_descriptors_used: bool,
    routes: Vec<usize>,
    dummy: Arc<Buffer>,
    input: Arc<Buffer>,
    gate: Arc<Buffer>,
    up: Arc<Buffer>,
    output: Arc<Buffer>,
    projected: Arc<Buffer>,
    selection: Arc<Buffer>,
    upload: Arc<Buffer>,
    routing_readback: Arc<Buffer>,
    quant: Option<QuantScratch>,
    packed_weights: [Arc<Buffer>; 3],
    packed_selection: Arc<Buffer>,
    packed_leases: Vec<Arc<Buffer>>,
    weight_staging: Arc<Buffer>,
    pending_upload: Option<super::hybrid::PreparedBatch>,
    cpu_routes: Vec<f32>,
    cpu_input: Vec<f32>,
    cpu_gate: Vec<f32>,
    cpu_up: Vec<f32>,
    cpu_output: Vec<f32>,
    selected: Vec<usize>,
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
        let bytes = self
            .commands
            .device
            .kv_precision
            .row_bytes(self.kv_width)?
            .checked_mul(self.position)
            .context("KV snapshot byte overflow")?;
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
        let (k, v) = if self.commands.device.kv_precision.is_compact() {
            let mut key_bytes = vec![0; bytes];
            let mut value_bytes = vec![0; bytes];
            host.read(0, &mut key_bytes)?;
            host.read(bytes, &mut value_bytes)?;
            (
                decode_q4_64_prefix(&key_bytes, self.position, self.kv_width)?,
                decode_q4_64_prefix(&value_bytes, self.position, self.kv_width)?,
            )
        } else {
            let mut k = vec![0.0; elements];
            let mut v = vec![0.0; elements];
            host.read(0, bytes_of_f32_mut(&mut k))?;
            host.read(bytes, bytes_of_f32_mut(&mut v))?;
            (k, v)
        };
        self.commands
            .device
            .readbacks
            .fetch_add((bytes * 2) as u64, Ordering::Relaxed);
        Ok((k, v))
    }
}

/// Diagnostic/reference decode; execution dequantizes directly in attention.
fn decode_q4_64_prefix(bytes: &[u8], rows: usize, width: usize) -> Result<Vec<f32>> {
    let row_bytes = width
        .div_ceil(128)
        .checked_mul(68)
        .context("compact KV row overflow")?;
    ensure!(
        bytes.len()
            == row_bytes
                .checked_mul(rows)
                .context("compact KV snapshot overflow")?,
        "compact KV snapshot byte count mismatch"
    );
    let mut decoded = vec![
        0.0;
        rows.checked_mul(width)
            .context("compact KV element overflow")?
    ];
    for (row, output) in decoded.chunks_mut(width).enumerate() {
        for (col, value) in output.iter_mut().enumerate() {
            let block = row * row_bytes + (col / 128) * 68;
            let scale_offset = block + ((col % 128) / 64) * 2;
            let scale = half::f16::from_bits(u16::from_le_bytes(
                bytes[scale_offset..scale_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let word_offset = block + 4 + ((col % 128) / 8) * 4;
            let word = u32::from_le_bytes(bytes[word_offset..word_offset + 4].try_into().unwrap());
            let code = ((word >> ((col % 8) * 4)) & 15) as i32;
            *value = (if code >= 8 { code - 16 } else { code }) as f32 * scale;
        }
    }
    Ok(decoded)
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
    let operation = commands.operation(push, buffers, groups)?;
    let mut continues_barrier_group = false;
    if commands.device.group_compute_barriers && operation.push.0[0] <= 2 {
        // GEMV, RMS and RoPE each write just binding 3. Binding 6 is a
        // separate status word updated only with atomic OR. Direct weights
        // rebound at binding 0 are immutable cache allocations distinct from
        // all graph scratch buffers. Treat every other binding as accessed.
        for previous in graph.iter().rev() {
            let earlier = &previous.operation;
            if earlier.push.0[0] > 2
                || earlier.buffers.iter().any(|buffer| Arc::ptr_eq(buffer, &operation.buffers[3]))
                || operation.buffers.iter().any(|buffer| Arc::ptr_eq(buffer, &earlier.buffers[3]))
            {
                break;
            }
            if !previous.continues_barrier_group {
                continues_barrier_group = true;
                break;
            }
        }
    }
    graph.push(GraphOp { operation, dynamic, continues_barrier_group });
    Ok(())
}
fn elements_groups(commands: &Commands, count: usize) -> Result<[usize; 3]> {
    commands.groups(count.div_ceil(128))
}

fn routed_expert(name: &str) -> bool {
    name.ends_with(".ffn_gate_exps.weight")
        || name.ends_with(".ffn_up_exps.weight")
        || name.ends_with(".ffn_down_exps.weight")
}

// This is a lower bound: Vulkan alignment, upload staging and request state
// need additional memory. Check it before uploading any checkpoint tensors.
fn resident_weight_bytes<'a>(tensors: impl IntoIterator<Item = &'a TensorInfo>) -> Result<u64> {
    tensors.into_iter().try_fold(0u64, |total, info| {
        let bytes = if info.dims.len() == 1 {
            // The resident loader expands vectors (including F16/BF16) to F32.
            info.dims[0]
                .checked_mul(4)
                .context("resident vector size overflow")?
        } else {
            u64::try_from(info.byte_len).context("resident tensor size overflow")?
        };
        total
            .checked_add(bytes)
            .context("resident weight size overflow")
    })
}

fn check_resident_weight_budget(weight_bytes: u64, budget: u64) -> Result<()> {
    const GIB: f64 = (1024 * 1024 * 1024) as f64;
    ensure!(
        weight_bytes <= budget,
        "resident Vulkan fixed GPU weights need at least {:.2} GiB \
({weight_bytes} bytes), exceeding the {:.2} GiB owned-allocation budget ({budget} bytes). \
Choose the Hybrid backend (--backend hybrid) to use bounded GPU weight caching, or a \
smaller checkpoint. Upload staging and request state need additional memory; increasing \
the cap cannot make a model larger than the GPU fit",
        weight_bytes as f64 / GIB,
        budget as f64 / GIB,
    );
    Ok(())
}

/// Payload accounting excludes evictable expert weights. A growing contiguous
/// cache temporarily retains one old layer while its replacement is installed.
#[derive(Clone, Copy, Debug)]
struct KvMemoryPlan {
    row_bytes: usize,
    full_payload: u64,
    attention_scratch: u64,
    growth_overlap: u64,
}
impl KvMemoryPlan {
    fn new(config: &Config, row_bytes: usize, compact: bool) -> Result<Self> {
        let context = config.context_length;
        let full_payload = (row_bytes as u64)
            .checked_mul(context as u64)
            .and_then(|n| n.checked_mul(config.block_count as u64))
            .and_then(|n| n.checked_mul(2))
            .context("full context KV byte overflow")?;
        let attention_elements = if compact {
            config
                .head_count
                .checked_mul(context.div_ceil(1024))
                .and_then(|n| n.checked_mul(config.head_dim.checked_add(2)?))
        } else {
            config.head_count.checked_mul(context)
        }
        .context("full context attention scratch overflow")?;
        let mut capacity = context.min(256);
        let mut old_capacity = 0;
        while capacity < context {
            old_capacity = capacity;
            capacity = capacity.saturating_mul(2).min(context);
        }
        let growth_overlap = (row_bytes as u64)
            .checked_mul(old_capacity as u64)
            .and_then(|n| n.checked_mul(2))
            .context("KV growth overlap overflow")?;
        Ok(Self {
            row_bytes,
            full_payload,
            attention_scratch: float_bytes(attention_elements)? as u64,
            growth_overlap,
        })
    }
    fn minimum_peak(&self, fixed_bytes: u64) -> Result<u64> {
        fixed_bytes
            .checked_add(self.full_payload)
            .and_then(|n| n.checked_add(self.growth_overlap))
            .context("full context memory plan overflow")
    }
    fn check(&self, fixed_bytes: u64, budget: u64) -> Result<u64> {
        self.check_location(fixed_bytes, budget, "GPU")
    }
    fn check_location(&self, fixed_bytes: u64, budget: u64, location: &str) -> Result<u64> {
        let peak = self.minimum_peak(fixed_bytes)?;
        ensure!(
            peak <= budget,
            "full context requires at least {peak} {location} bytes ({:.3} GiB), including all KV history, fixed state, and one-layer growth overlap; allocation budget is {budget} bytes ({:.3} GiB)",
            peak as f64 / (1024.0 * 1024.0 * 1024.0),
            budget as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        Ok(peak)
    }
}

impl GpuModel {
    pub fn load(
        source: Arc<Gguf>,
        context: usize,
        options: DeviceOptions,
        progress: super::UploadProgress<'_>,
    ) -> Result<Self> {
        // Reuse the independent loader's geometry/semantic validation, never
        // inference or weights from the compatibility engine.
        let checked = Model::load(Arc::clone(&source), context)?;
        let config = checked.config().clone();
        ensure!(
            matches!(config.architecture.as_str(), "qwen3" | "qwen3moe"),
            "resident Vulkan graph supports ordinary qwen3/qwen3moe; use the explicit hybrid engine for extended architectures"
        );
        drop(checked);
        let cache = if config.expert_count > 0 {
            Some(super::hybrid::GpuMatvecExecutor::new(options.clone())?)
        } else {
            None
        };
        let device = if let Some(cache) = &cache {
            cache.device()
        } else {
            Device::new(&options)?
        };
        check_resident_weight_budget(
            resident_weight_bytes(
                source
                    .tensors
                    .iter()
                    .filter(|(name, _)| !routed_expert(name))
                    .map(|(_, info)| info),
            )?,
            device.budget,
        )?;
        let compact = device.kv_precision.is_compact();
        let ram = device.kv_placement == KvPlacement::Ram;
        let kv_width = config
            .head_count_kv
            .checked_mul(config.head_dim)
            .context("KV width overflow")?;
        let kv_plan =
            KvMemoryPlan::new(&config, device.kv_precision.row_bytes(kv_width)?, compact)?;
        let fixed_minimum = resident_weight_bytes(
            source
                .tensors
                .iter()
                .filter(|(name, _)| !routed_expert(name))
                .map(|(_, info)| info),
        )?
        .checked_add(kv_plan.attention_scratch)
        .context("context fixed-state overflow")?;
        let (minimum_peak, ram_peak) = if ram {
            ensure!(
                fixed_minimum <= device.budget,
                "fixed GPU model and attention state exceed GPU allocation budget"
            );
            (
                fixed_minimum,
                kv_plan.check_location(0, device.host_budget, "RAM")?,
            )
        } else if compact {
            (kv_plan.check(fixed_minimum, device.budget)?, 0)
        } else {
            (0, 0)
        };
        ensure!(
            (if compact {
                context.div_ceil(1024)
            } else {
                context
            }) <= device.properties.limits.max_compute_work_group_count[0] as usize,
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
        let weight_count = source
            .tensors
            .keys()
            .filter(|name| !routed_expert(name))
            .count();
        progress(
            "Uploading resident GPU weights",
            "tensors",
            0,
            weight_count,
            0,
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
            if config.expert_count > 0 && routed_expert(name) {
                shapes.insert(name.clone(), shape);
                continue;
            }
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
            progress(
                "Uploading resident GPU weights",
                "tensors",
                weights.len(),
                weight_count,
                device.uploads.load(Ordering::Relaxed),
            );
        }
        let output = if source.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
        .to_string();
        let rope_frequencies = if compact || config.rope_scaling.is_some() {
            Some(config.rope_frequencies()?)
        } else {
            None
        };
        // Short sparse routing passes can finish before the Windows AMD
        // driver wakes a sleeping worker. Bound the polling cost per fence.
        let routing_poll_budget = if cfg!(windows) && device.group_compute_barriers
            && device.properties.vendor_id == 0x1002
        {
            std::time::Duration::from_micros(750)
        } else {
            std::time::Duration::ZERO
        };
        let model = Self {
            rope_frequencies,
            kv_plan,
            full_context_minimum_peak: AtomicU64::new(minimum_peak),
            full_context_ram_peak: AtomicU64::new(ram_peak),
            device,
            cache,
            source,
            cpu_experts: AtomicU64::new(0),
            gpu_experts: AtomicU64::new(0),
            packed_layers: AtomicU64::new(0),
            packed_bytes: AtomicU64::new(0),
            direct_layers: AtomicU64::new(0),
            direct_source_bytes: AtomicU64::new(0),
            sparse_profile: SparseProfile::new(),
            routing_poll_budget,
            routing_poll_hits: AtomicU64::new(0),
            routing_poll_misses: AtomicU64::new(0),
            routing_poll_ns: AtomicU64::new(0),
            reuse_packed_graphs: std::env::var_os("NEXTORA_REUSE_EXPERT_GRAPH")
                .is_none_or(|value| value != "0"),
            packed_graph_builds: AtomicU64::new(0),
            packed_graph_reuses: AtomicU64::new(0),
            config,
            weights,
            shapes,
            owner: Arc::new(()),
            weight_bytes,
            output,
        };
        if compact || ram {
            // A service must not advertise Ready before validating the actual
            // fixed graph/scratch cost as well as the complete future KV cache.
            progress(
                "Validating full context memory",
                "tokens",
                0,
                context,
                model.device.uploads.load(Ordering::Relaxed),
            );
            let probe = model.new_sequence().map_err(|error| {
                anyhow::anyhow!("Vulkan context {context} memory validation failed: {error:#}")
            })?;
            drop(probe);
            progress(
                "Validating full context memory",
                "tokens",
                context,
                context,
                model.device.uploads.load(Ordering::Relaxed),
            );
        }
        Ok(model)
    }
    /// Capacity proof only: writes every allocated KV byte, without claiming
    /// that a synthetic zero history tests model quality at that length.
    pub fn probe_capacity(&self) -> Result<Value> {
        let mut sequence = self.new_sequence()?;
        self.ensure_capacity(&mut sequence, self.context())?;
        sequence.commands.begin()?;
        for buffer in sequence.keys.iter().chain(&sequence.values) {
            unsafe {
                self.device.raw.cmd_fill_buffer(
                    sequence.commands.raw,
                    buffer.raw,
                    0,
                    buffer.size,
                    0,
                );
            }
        }
        sequence.commands.transfer_to_compute();
        sequence.commands.submit()?;
        let report = json!({
            "capacity_proof":"full_kv_allocated_and_touched",
            "context_tokens":self.context(),"kv_precision":self.device.kv_precision.as_str(),
            "kv_placement":self.device.kv_placement.as_str(),
            "kv_payload_bytes":self.kv_plan.full_payload,
            "kv_buffers_on_system_ram":sequence.keys.iter().chain(&sequence.values).all(|buffer| buffer.is_system_ram()),
            "software_ram_simulation":self.device.info()["software"],
            "history_tokens_processed":0,"long_context_quality_verified":false,
            "memory":self.memory_stats(),
        });
        drop(sequence);
        Ok(report)
    }
    pub(crate) fn fit_memory_report(
        config: &Config,
        source: &Gguf,
        options: &DeviceOptions,
    ) -> Result<Value> {
        let row = options
            .kv_precision
            .row_bytes(config.head_count_kv * config.head_dim)?;
        let plan = KvMemoryPlan::new(config, row, options.kv_precision.is_compact())?;
        let dense = resident_weight_bytes(
            source
                .tensors
                .iter()
                .filter(|(name, _)| !routed_expert(name))
                .map(|(_, info)| info),
        )?;
        let fixed_minimum = dense
            .checked_add(plan.attention_scratch)
            .context("fit fixed-state overflow")?;
        Ok(json!({
            "estimate":"payload_lower_bound; load validates actual fixed graph and working buffers",
            "configured_context_tokens":config.context_length,
            "kv_precision":options.kv_precision.as_str(),"kv_placement":options.kv_placement.as_str(),
            "kv_payload_bytes":plan.full_payload,"kv_growth_overlap_bytes":plan.growth_overlap,
            "attention_scratch_bytes":plan.attention_scratch,"resident_dense_weight_bytes":dense,
            "gpu_minimum_peak_bytes":if options.kv_placement.is_gpu() { plan.minimum_peak(fixed_minimum)? } else { fixed_minimum },
            "ram_context_minimum_peak_bytes":if options.kv_placement == KvPlacement::Ram { plan.minimum_peak(0)? } else { 0 },
        }))
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
        stats["kv_precision"] = json!(self.device.kv_precision.as_str());
        stats["full_context_kv_payload_bytes"] = json!(self.kv_plan.full_payload);
        stats["full_context_attention_scratch_bytes"] = json!(self.kv_plan.attention_scratch);
        stats["full_context_kv_growth_overlap_bytes"] = json!(self.kv_plan.growth_overlap);
        stats["full_context_minimum_peak_bytes"] =
            json!(self.full_context_minimum_peak.load(Ordering::Relaxed));
        stats["full_context_peak_bytes"] =
            json!(self.full_context_minimum_peak.load(Ordering::Relaxed));
        stats["full_context_gpu_peak_bytes"] =
            json!(self.full_context_minimum_peak.load(Ordering::Relaxed));
        stats["full_context_ram_peak_bytes"] =
            json!(self.full_context_ram_peak.load(Ordering::Relaxed));
        stats["kv_placement"] = json!(self.device.kv_placement.as_str());
        stats["configured_context_tokens"] = json!(self.context());
        stats["attention_history"] = json!("all_initialized_tokens");
        if !self.routing_poll_budget.is_zero() {
            stats["routing_fence_poll"] = json!({
                "budget_us": self.routing_poll_budget.as_micros() as u64,
                "completed": self.routing_poll_hits.load(Ordering::Relaxed),
                "blocked_after_poll": self.routing_poll_misses.load(Ordering::Relaxed),
                "host_poll_ms": self.routing_poll_ns.load(Ordering::Relaxed) as f64 / 1e6,
            });
        }
        stats["rope_rotation"] = json!(if self.rope_frequencies.is_some() {
            "cpu_precomputed_scaled_tables"
        } else {
            "gpu_trigonometry"
        });
        stats["execution"] = json!(if self.cache.is_some() {
            "ordinary_gpu_state_with_bounded_expert_cache"
        } else {
            "resident_gpu_graph"
        });
        if let Some(cache) = &self.cache {
            stats["expert_cache"] = cache.stats();
        }
        stats["cpu_expert_fallbacks"] = json!(self.cpu_experts.load(Ordering::Relaxed));
        stats["gpu_expert_executions"] = json!(self.gpu_experts.load(Ordering::Relaxed));
        stats["selected_experts_omitted"] = json!(0);
        stats["batched_expert_layers"] = json!(self.packed_layers.load(Ordering::Relaxed));
        stats["gpu_expert_pack_bytes"] = json!(self.packed_bytes.load(Ordering::Relaxed));
        stats["direct_expert_layers"] = json!(self.direct_layers.load(Ordering::Relaxed));
        stats["direct_expert_source_bytes"] =
            json!(self.direct_source_bytes.load(Ordering::Relaxed));
        stats["gpu_expert_pack_bytes_avoided"] =
            json!(self.direct_source_bytes.load(Ordering::Relaxed));
        stats["packed_graph_reuse_enabled"] = json!(self.reuse_packed_graphs);
        stats["packed_graph_builds"] = json!(self.packed_graph_builds.load(Ordering::Relaxed));
        stats["packed_graph_reuses"] = json!(self.packed_graph_reuses.load(Ordering::Relaxed));
        stats["fast_math"] = json!(self.device.fast_math);
        stats["activation_precision"] = json!(if self.device.fast_math { "Q8" } else { "F32" });
        if self.sparse_profile.enabled {
            stats["sparse_profile"] = json!({
                "timing": "host_wall_time; submit includes GPU fence wait; graph time includes CPU expert compute",
                "prefill": self.sparse_profile.prefill.stats(),
                "decode": self.sparse_profile.decode.stats(),
            });
        }
        stats["kv_allocation"] = json!("grows_with_initialized_context");
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
    fn quantize(
        &self,
        commands: &mut Commands,
        graph: &mut Vec<GraphOp>,
        quant: &QuantScratch,
        input: &Arc<Buffer>,
        status: &Arc<Buffer>,
        offset: usize,
        count: usize,
    ) -> Result<()> {
        let reused = graph.last().is_some_and(|last| {
            last.operation.push.0[0] == 0
                && last.operation.push.0[15] & 32 != 0
                && last.operation.push.0[1] as usize == count
                && last.operation.push.0[11] as usize == offset
                && Arc::ptr_eq(&last.operation.buffers[1], input)
                && !Arc::ptr_eq(&last.operation.buffers[3], input)
                && Arc::ptr_eq(&last.operation.buffers[2], &quant.packed)
        });
        if reused {
            return Ok(());
        }
        ensure!(
            offset % 32 == 0 && count % 32 == 0,
            "Q8 activation span must be 32-aligned"
        );
        let mut push = Push::new(16).n(0, count)?;
        push.0[10] = u32::try_from(offset)?;
        push.0[11] = u32::try_from(offset / 4)?;
        push.0[13] = u32::try_from(offset / 32)?;
        let groups = commands.groups(count.div_ceil(32))?;
        append(
            commands,
            graph,
            push,
            bindings(
                status,
                status,
                input,
                &quant.packed,
                status,
                &quant.scales,
                status,
                status,
            ),
            groups,
            Dynamic::Fixed,
        )
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
        quant: Option<&QuantScratch>,
    ) -> Result<()> {
        let s = self.shape(name)?;
        let mut groups = commands.groups(s.rows)?;
        groups[2] = if selected {
            self.config.experts_per_token
        } else {
            1
        };
        let mut p = Push::new(0)
            .n(0, s.width)?
            .n(1, s.rows)?
            .n(2, s.ty as usize)?
            .n(3, s.row_bytes)?
            .n(4, s.experts)?
            .flags(u32::from(selected) | u32::from(individual_inputs) * 2);
        let quant = quant.filter(|_| matches!(s.ty, 12 | 14) && s.width % 256 == 0);
        if let Some(quant) = quant {
            self.quantize(commands, graph, quant, input, status, 0, s.width)?;
            p.0[15] |= 32;
        }
        append(
            commands,
            graph,
            p,
            bindings(
                dummy,
                self.weight(name)?,
                input,
                quant.map(|q| &q.packed).unwrap_or(dummy),
                output,
                quant.map(|q| &q.scales).unwrap_or(dummy),
                selection,
                status,
            ),
            groups,
            Dynamic::Fixed,
        )
    }
    pub fn new_sequence(&self) -> Result<GpuSequence> {
        let c = &self.config;
        let compact = self.device.kv_precision.is_compact();
        let ram = self.device.kv_placement == KvPlacement::Ram;
        let kv_capacity = if self.cache.is_some() || compact || ram {
            c.context_length.min(256)
        } else {
            c.context_length
        };
        let mut commands = Commands::new(
            Arc::clone(&self.device),
            c.block_count
                .checked_mul(if self.device.fast_math { 40 } else { 24 })
                .and_then(|n| n.checked_add(8))
                .context("graph descriptor bound overflow")?,
        )?;
        let storage = |n: usize| {
            let bytes = float_bytes(n)?;
            if let Some(cache) = &self.cache {
                cache.make_room(
                    bytes
                        .checked_add(128 * 1024)
                        .context("sequence reserve overflow")?,
                )?;
            }
            self.device.storage(bytes)
        };
        let host = |bytes: usize| self.device.host(bytes);
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
        let max_chunks = c.context_length.div_ceil(1024);
        let scores = storage(
            if compact {
                c.head_count
                    .checked_mul(max_chunks)
                    .and_then(|n| n.checked_mul(c.head_dim.checked_add(2)?))
            } else {
                c.head_count.checked_mul(c.context_length)
            }
            .context("attention scratch overflow")?,
        )?;
        let logits = storage(c.vocab_size)?;
        let hidden = storage(c.embedding_length)?;
        let readback = host(float_bytes(
            c.vocab_size
                .checked_add(c.embedding_length)
                .and_then(|n| n.checked_add(1))
                .context("readback overflow")?,
        )?)?;
        let kv_width = c.head_count_kv * c.head_dim;
        let kv_bytes = self
            .kv_plan
            .row_bytes
            .checked_mul(kv_capacity)
            .context("KV capacity overflow")?;
        let mut keys = Vec::with_capacity(c.block_count);
        let mut values = Vec::with_capacity(c.block_count);
        for _ in 0..c.block_count {
            if ram {
                keys.push(self.device.host_storage(kv_bytes)?);
                values.push(self.device.host_storage(kv_bytes)?);
            } else {
                keys.push(storage(kv_bytes / 4)?);
                values.push(storage(kv_bytes / 4)?);
            }
        }
        if ram {
            ensure!(
                keys.iter()
                    .chain(&values)
                    .all(|buffer| buffer.is_system_ram()),
                "RAM context buffers must use system memory, not host-visible VRAM"
            );
        }
        let quant = if self.device.fast_math {
            let maximum = self
                .shapes
                .values()
                .map(|shape| shape.width)
                .max()
                .unwrap_or(1)
                .max(ffn * slots);
            let groups = maximum.div_ceil(32);
            Some(QuantScratch {
                packed: storage(groups * 8)?,
                scales: storage(groups)?,
            })
        } else {
            None
        };
        let rope = if self.rope_frequencies.is_some() {
            let elements = c
                .head_dim
                .checked_mul(32)
                .context("RoPE chunk table overflow")?;
            Some(RopeTables {
                device: storage(elements)?,
                upload: host(float_bytes(elements)?)?,
                values: vec![0.0; elements],
            })
        } else {
            None
        };
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
                quant.as_ref(),
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
                quant.as_ref(),
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
                quant.as_ref(),
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
                        .f(0, c.rope_base)
                        .f(1, c.rope_attention_factor())
                        .flags(if rope.is_some() { 256 } else { 0 }),
                    bindings(
                        &dummy,
                        &dummy,
                        buffer,
                        rope.as_ref().map(|tables| &tables.device).unwrap_or(&dummy),
                        buffer,
                        &dummy,
                        &dummy,
                        &status,
                    ),
                    groups,
                    Dynamic::Rope,
                )?;
            }
            let groups = elements_groups(&commands, kv_width)?;
            append(
                &mut commands,
                &mut forward,
                Push::new(3)
                    .n(0, kv_width)?
                    .flags(if compact { 64 } else { 0 }),
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
            if compact {
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(17)
                        .n(0, 1)?
                        .n(1, c.head_count)?
                        .n(2, c.head_count_kv)?
                        .n(3, c.head_dim)?
                        .n(4, max_chunks)?
                        .n(5, kv_width)?,
                    bindings(
                        &dummy,
                        &dummy,
                        &query,
                        &keys[layer],
                        &scores,
                        &values[layer],
                        &dummy,
                        &status,
                    ),
                    [1, c.head_count, 1],
                    Dynamic::CompactScores,
                )?;
                let groups = commands.groups(c.head_count)?;
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(18)
                        .n(0, 1)?
                        .n(1, c.head_count)?
                        .n(3, c.head_dim)?
                        .n(4, max_chunks)?,
                    bindings(
                        &dummy, &dummy, &scores, &dummy, &attention, &dummy, &dummy, &status,
                    ),
                    groups,
                    Dynamic::CompactAttention,
                )?;
            } else {
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
            }
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
                quant.as_ref(),
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
                    quant.as_ref(),
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
                forward.last_mut().unwrap().dynamic = Dynamic::Experts;
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
                    quant.as_ref(),
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
                    quant.as_ref(),
                )?;
            }
            if c.expert_count == 0 {
                let groups = elements_groups(&commands, ffn * slots)?;
                append(
                    &mut commands,
                    &mut forward,
                    Push::new(6).n(0, ffn * slots)?,
                    bindings(&dummy, &dummy, &gate, &up, &gate, &dummy, &dummy, &status),
                    groups,
                    Dynamic::Fixed,
                )?;
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
                    quant.as_ref(),
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
            quant.as_ref(),
        )?;
        let sparse = if self.cache.is_some() {
            let packed_size = |suffix: &str| -> Result<usize> {
                let mut bytes = 0;
                for layer in 0..c.block_count {
                    let shape = self.shape(&format!("blk.{layer}.{suffix}.weight"))?;
                    bytes = bytes.max(
                        shape
                            .row_bytes
                            .checked_mul(shape.rows)
                            .and_then(|n| n.checked_mul(slots))
                            .context("packed expert scratch overflow")?,
                    );
                }
                Ok(bytes.div_ceil(4))
            };
            let packed_weights = [
                storage(packed_size("ffn_gate_exps")?)?,
                storage(packed_size("ffn_up_exps")?)?,
                storage(packed_size("ffn_down_exps")?)?,
            ];
            let routes: Vec<f32> = (0..slots).flat_map(|slot| [slot as f32, 1.0]).collect();
            self.cache
                .as_ref()
                .unwrap()
                .make_room(routes.len() * 4 + 128 * 1024)?;
            let packed_selection = self.device.upload(bytes_of_f32(&routes))?;
            Some(SparseSequence {
                packed_commands: Commands::new(
                    Arc::clone(&self.device),
                    c.block_count
                        .checked_mul(if self.device.fast_math { 7 } else { 5 })
                        .and_then(|n| n.checked_add(1))
                        .context("persistent expert descriptor overflow")?,
                )?,
                packed_graphs: (0..c.block_count).map(|_| None).collect(),
                dynamic_descriptors_used: false,
                quant: quant.clone(),
                packed_weights,
                packed_selection,
                packed_leases: Vec::new(),
                commands: Commands::new(
                    Arc::clone(&self.device),
                    slots
                        .checked_mul(if self.device.fast_math { 6 } else { 4 })
                        .and_then(|n| n.checked_add(1))
                        .context("expert descriptor overflow")?,
                )?,
                routes: forward
                    .iter()
                    .enumerate()
                    .filter_map(|(index, op)| {
                        matches!(op.dynamic, Dynamic::Experts).then_some(index)
                    })
                    .collect(),
                dummy,
                input: normalized,
                gate,
                up,
                output: expert_output,
                projected,
                selection,
                upload: host(float_bytes(slots * c.embedding_length)?)?,
                routing_readback: host(float_bytes(slots * 2 + c.embedding_length + 1)?)?,
                weight_staging: host(
                    (self.device.budget / 32).clamp(4096, 64 * 1024 * 1024) as usize
                )?,
                pending_upload: None,
                cpu_routes: vec![0.0; slots * 2],
                cpu_input: vec![0.0; c.embedding_length],
                cpu_gate: vec![0.0; ffn],
                cpu_up: vec![0.0; ffn],
                cpu_output: vec![0.0; slots * c.embedding_length],
                selected: Vec::with_capacity(slots),
            })
        } else {
            None
        };
        if compact || ram {
            let evictable = self
                .cache
                .as_ref()
                .map(|cache| cache.stats())
                .and_then(|stats| stats["cached_packed_weight_bytes"].as_u64())
                .unwrap_or(0);
            let initialized_payload = (kv_bytes as u64)
                .checked_mul((c.block_count * 2) as u64)
                .context("initial KV byte overflow")?;
            if ram {
                let gpu_fixed = self.device.reserved().saturating_sub(evictable);
                ensure!(
                    gpu_fixed <= self.device.budget,
                    "fixed GPU state exceeds GPU allocation budget"
                );
                self.full_context_minimum_peak
                    .fetch_max(gpu_fixed, Ordering::Relaxed);
                let ram_fixed = self
                    .device
                    .host_reserved()
                    .saturating_sub(initialized_payload);
                let peak =
                    self.kv_plan
                        .check_location(ram_fixed, self.device.host_budget, "RAM")?;
                self.full_context_ram_peak
                    .fetch_max(peak, Ordering::Relaxed);
            } else {
                let fixed = self
                    .device
                    .reserved()
                    .saturating_sub(evictable)
                    .saturating_sub(initialized_payload);
                let peak = self.kv_plan.check(fixed, self.device.budget)?;
                self.full_context_minimum_peak
                    .fetch_max(peak, Ordering::Relaxed);
            }
        }
        Ok(GpuSequence {
            commands,
            owner: Arc::clone(&self.owner),
            position: 0,
            poisoned: false,
            forward,
            sparse,
            kv_capacity,
            rope,
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
        if self.cache.is_some() {
            let result = self.execute_sparse(sequence, tokens, out);
            if let Err(error) = result {
                sequence.position = start;
                sequence.poisoned = true;
                return Err(error);
            }
            return Ok(());
        }
        for (chunk_index, chunk) in tokens.chunks(32).enumerate() {
            let last = (chunk_index + 1) * 32 >= tokens.len();
            let result = self
                .ensure_capacity(sequence, sequence.position + chunk.len())
                .and_then(|()| self.execute_chunk(sequence, chunk, last, out));
            if let Err(error) = result {
                sequence.position = start;
                sequence.poisoned = true;
                return Err(error);
            }
            sequence.position += chunk.len();
        }
        Ok(())
    }
    fn ensure_capacity(&self, s: &mut GpuSequence, required: usize) -> Result<()> {
        if required <= s.kv_capacity {
            return Ok(());
        }
        let capacity = required
            .max(s.kv_capacity.saturating_mul(2))
            .min(self.context());
        let bytes = self
            .kv_plan
            .row_bytes
            .checked_mul(capacity)
            .context("KV growth overflow")?;
        let prefix = self
            .kv_plan
            .row_bytes
            .checked_mul(s.position)
            .context("KV prefix overflow")?;
        // Replace one layer at a time, so the temporary overlap is a single old
        // layer rather than another copy of the entire attention cache.
        for layer in 0..s.keys.len() {
            if let Some(cache) = self
                .cache
                .as_ref()
                .filter(|_| self.device.kv_placement.is_gpu())
            {
                cache.make_room(
                    bytes
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(128 * 1024))
                        .context("KV allocation overflow")?,
                )?;
            }
            let (key, value) = if self.device.kv_placement == KvPlacement::Ram {
                (
                    self.device.host_storage(bytes)?,
                    self.device.host_storage(bytes)?,
                )
            } else {
                (self.device.storage(bytes)?, self.device.storage(bytes)?)
            };
            let old_key = Arc::clone(&s.keys[layer]);
            let old_value = Arc::clone(&s.values[layer]);
            if prefix > 0 {
                s.commands.begin()?;
                s.commands.compute_to_transfer();
                unsafe {
                    for (old, new) in [(&old_key, &key), (&old_value, &value)] {
                        self.device.raw.cmd_copy_buffer(
                            s.commands.raw,
                            old.raw,
                            new.raw,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: prefix as u64,
                            }],
                        );
                    }
                }
                s.commands.transfer_to_compute();
                s.commands.submit()?;
            }
            for op in &mut s.forward {
                let mut changed = false;
                for binding in &mut op.operation.buffers {
                    if Arc::ptr_eq(binding, &old_key) {
                        *binding = Arc::clone(&key);
                        changed = true;
                    } else if Arc::ptr_eq(binding, &old_value) {
                        *binding = Arc::clone(&value);
                        changed = true;
                    }
                }
                if changed {
                    s.commands
                        .update_set(op.operation.set, &op.operation.buffers);
                }
            }
            s.keys[layer] = key;
            s.values[layer] = value;
        }
        s.kv_capacity = capacity;
        Ok(())
    }

    fn prepare_rope_tables(&self, s: &mut GpuSequence, rows: usize) -> Result<()> {
        let Some(tables) = &mut s.rope else {
            return Ok(());
        };
        ensure!(rows > 0 && rows <= 32, "RoPE upload exceeds chunk capacity");
        let frequencies = self
            .rope_frequencies
            .as_ref()
            .context("missing CPU RoPE frequencies")?;
        let width = self.config.head_dim;
        let half = width / 2;
        for (offset, row) in tables.values[..rows * width].chunks_mut(width).enumerate() {
            let (cosine, sine) = row.split_at_mut(half);
            math::rope_angles_scaled(
                s.position + offset,
                frequencies,
                self.config.rope_attention_factor(),
                cosine,
                sine,
            )?;
        }
        tables
            .upload
            .write(0, bytes_of_f32(&tables.values[..rows * width]))
    }

    /// Records the small table transfer before the existing transfer-to-compute
    /// barrier. It does not submit or add a synchronization point.
    fn record_rope_upload(&self, commands: &Commands, s: &GpuSequence, rows: usize) {
        if let Some(tables) = &s.rope {
            let bytes = rows * self.config.head_dim * 4;
            unsafe {
                self.device.raw.cmd_copy_buffer(
                    commands.raw,
                    tables.upload.raw,
                    tables.device.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: bytes as u64,
                    }],
                );
            }
            self.device
                .uploads
                .fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }

    fn record_range(commands: &Commands, graph: &[GraphOp], token: u32, position: usize) {
        for (index, op) in graph.iter().enumerate() {
            let mut push = op.operation.push.clone();
            let mut groups = op.operation.groups;
            match op.dynamic {
                Dynamic::Fixed | Dynamic::Experts => {}
                Dynamic::Embedding => push.0[2] = token,
                Dynamic::Rope => {
                    push.0[3] = position as u32;
                    push.0[10] = 0;
                }
                Dynamic::Store => push.0[2] = position as u32,
                Dynamic::Scores => {
                    push.0[1] = (position + 1) as u32;
                    groups[0] = (position + 1) as u32;
                }
                Dynamic::Attention => push.0[1] = (position + 1) as u32,
                Dynamic::CompactScores => {
                    push.0[1] = (position + 1) as u32;
                    groups[0] = (position + 1).div_ceil(1024) as u32;
                }
                Dynamic::CompactAttention => push.0[1] = (position + 1).div_ceil(1024) as u32,
            }
            // A slice may end in the middle of a group (for example a
            // diagnostic readback). Always close the group at that boundary.
            let barrier_after = !graph.get(index + 1).is_some_and(|next| next.continues_barrier_group);
            commands.record_dispatch(&op.operation, &push, groups, barrier_after);
        }
    }

    fn copy_routing(&self, commands: &Commands, s: &GpuSequence, sparse: &SparseSequence) {
        commands.compute_to_transfer();
        let routes = sparse.cpu_routes.len() * 4;
        let input = sparse.cpu_input.len() * 4;
        unsafe {
            for (source, offset, size) in [
                (&sparse.selection, 0, routes),
                (&sparse.input, routes, input),
                (&s.status, routes + input, 4),
            ] {
                self.device.raw.cmd_copy_buffer(
                    commands.raw,
                    source.raw,
                    sparse.routing_readback.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset as u64,
                        size: size as u64,
                    }],
                );
            }
        }
        commands.transfer_to_host();
    }

    fn read_routing(&self, sparse: &mut SparseSequence) -> Result<()> {
        let routes = sparse.cpu_routes.len() * 4;
        let input = sparse.cpu_input.len() * 4;
        let mut status = [0u8; 4];
        sparse.routing_readback.read(routes + input, &mut status)?;
        ensure!(
            u32::from_ne_bytes(status) == 0,
            "ordinary GPU routing produced a non-finite intermediate"
        );
        sparse
            .routing_readback
            .read(0, bytes_of_f32_mut(&mut sparse.cpu_routes))?;
        sparse
            .routing_readback
            .read(routes, bytes_of_f32_mut(&mut sparse.cpu_input))?;
        sparse.selected.clear();
        for pair in sparse.cpu_routes.chunks_exact(2) {
            let id = pair[0];
            ensure!(
                id.is_finite()
                    && id >= 0.0
                    && id.fract() == 0.0
                    && id < self.config.expert_count as f32
                    && pair[1].is_finite()
                    && pair[1] >= 0.0,
                "invalid ordinary GPU expert routing"
            );
            let id = id as usize;
            ensure!(
                !sparse.selected.contains(&id),
                "ordinary GPU router selected an expert twice"
            );
            sparse.selected.push(id);
        }
        ensure!(
            sparse.cpu_input.iter().all(|x| x.is_finite()),
            "ordinary GPU expert input is non-finite"
        );
        self.device
            .readbacks
            .fetch_add((routes + input + 4) as u64, Ordering::Relaxed);
        Ok(())
    }

    fn expert_matrix(
        &self,
        sparse: &mut SparseSequence,
        graph: &mut Vec<GraphOp>,
        view: Tensor<'_>,
        weight: Arc<Buffer>,
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        status: &Arc<Buffer>,
        input_offset: usize,
        output_offset: usize,
    ) -> Result<()> {
        let mut push = Push::new(0)
            .n(0, view.width())?
            .n(1, view.rows())?
            .n(2, view.quant_type().id() as usize)?
            .n(3, view.quant_type().row_bytes(view.width())?)?
            .n(4, 1)?;
        push.0[11] = u32::try_from(input_offset)?;
        push.0[12] = u32::try_from(output_offset)?;
        let quant = sparse.quant.as_ref().filter(|_| {
            matches!(view.quant_type().id(), 12 | 14)
                && view.width() % 256 == 0
                && input_offset % 32 == 0
        });
        if let Some(quant) = quant {
            self.quantize(
                &mut sparse.commands,
                graph,
                quant,
                &input,
                status,
                input_offset,
                view.width(),
            )?;
            push.0[15] |= 32;
        }
        let groups = sparse.commands.groups(view.rows())?;
        append(
            &mut sparse.commands,
            graph,
            push,
            bindings(
                &sparse.dummy,
                &weight,
                &input,
                quant.map(|q| &q.packed).unwrap_or(&sparse.dummy),
                &output,
                quant.map(|q| &q.scales).unwrap_or(&sparse.dummy),
                &sparse.dummy,
                status,
            ),
            groups,
            Dynamic::Fixed,
        )
    }

    fn experts(
        &self,
        s: &GpuSequence,
        sparse: &mut SparseSequence,
        layer: usize,
        admit: bool,
    ) -> Result<Arc<[GraphOp]>> {
        let cache = self.cache.as_ref().unwrap();
        let profile = self.sparse_profile.phase(admit);
        let started = self.sparse_profile.start();
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ];
        let slots = self.config.experts_per_token;
        let n = self.config.embedding_length;
        let ff = self.config.expert_feed_forward_length;
        let mut weights = vec![None; slots * 3];
        let mut leases = Vec::new();
        let mut missing = Vec::new();
        for slot in 0..slots {
            for (matrix, name) in names.iter().enumerate() {
                let expert = sparse.selected[slot];
                let key = self.source.tensor(name)?.expert(expert)?.cache_key();
                let index = slot * 3 + matrix;
                if let Some(hit) = cache.cached_weight(key, admit)? {
                    weights[index] = Some(hit);
                } else {
                    leases.push(if admit {
                        self.source.expert_cached(name, expert)?
                    } else {
                        self.source.expert_cached_for_prefill(name, expert)?
                    });
                    missing.push(index);
                }
            }
        }
        let views: Vec<_> = leases.iter().map(|lease| lease.tensor()).collect();
        SparseProfile::finish(started, &profile.lookup_ns);
        let started = self.sparse_profile.start();
        if sparse.dynamic_descriptors_used {
            sparse.commands.reset_descriptors()?;
            sparse.dynamic_descriptors_used = false;
        }
        sparse.commands.begin()?;
        let prepared = cache.prepare_batch_recorded(
            &views,
            admit,
            &sparse.commands,
            &sparse.weight_staging,
        )?;
        for (index, buffer) in missing.into_iter().zip(prepared.weights.iter().cloned()) {
            weights[index] = buffer;
        }
        let staged: BTreeMap<TensorKey, Tensor<'_>> =
            views.iter().map(|view| (view.cache_key(), *view)).collect();
        sparse.pending_upload = Some(prepared);
        SparseProfile::finish(started, &profile.prepare_ns);
        let started = self.sparse_profile.start();
        if weights.iter().all(Option::is_some)
            && names.iter().all(|name| {
                let shape = &self.shapes[name];
                (shape.rows * shape.row_bytes) % 4 == 0
            })
        {
            let result = self.packed_experts(s, sparse, layer, &names, &weights);
            SparseProfile::finish(started, &profile.graph_ns);
            return result;
        }
        sparse.dynamic_descriptors_used = true;
        let mut graph = Vec::with_capacity(slots * 4 + 1);
        let mut cpu_slots = Vec::new();
        for slot in 0..slots {
            let expert = sparse.selected[slot];
            let triplet = (
                weights[slot * 3].take(),
                weights[slot * 3 + 1].take(),
                weights[slot * 3 + 2].take(),
            );
            if let (Some(g), Some(u), Some(d)) = triplet {
                let gv = self.source.tensor(&names[0])?.expert(expert)?;
                let uv = self.source.tensor(&names[1])?.expert(expert)?;
                let dv = self.source.tensor(&names[2])?.expert(expert)?;
                self.expert_matrix(
                    sparse,
                    &mut graph,
                    gv,
                    g,
                    Arc::clone(&sparse.input),
                    Arc::clone(&sparse.gate),
                    &s.status,
                    0,
                    slot * ff,
                )?;
                self.expert_matrix(
                    sparse,
                    &mut graph,
                    uv,
                    u,
                    Arc::clone(&sparse.input),
                    Arc::clone(&sparse.up),
                    &s.status,
                    0,
                    slot * ff,
                )?;
                let mut push = Push::new(6).n(0, ff)?;
                push.0[10] = u32::try_from(slot * ff)?;
                push.0[11] = u32::try_from(slot * ff)?;
                push.0[12] = u32::try_from(slot * ff)?;
                let groups = elements_groups(&sparse.commands, ff)?;
                append(
                    &mut sparse.commands,
                    &mut graph,
                    push,
                    bindings(
                        &sparse.dummy,
                        &sparse.dummy,
                        &sparse.gate,
                        &sparse.up,
                        &sparse.gate,
                        &sparse.dummy,
                        &sparse.dummy,
                        &s.status,
                    ),
                    groups,
                    Dynamic::Fixed,
                )?;
                self.expert_matrix(
                    sparse,
                    &mut graph,
                    dv,
                    d,
                    Arc::clone(&sparse.gate),
                    Arc::clone(&sparse.output),
                    &s.status,
                    slot * ff,
                    slot * n,
                )?;
                self.gpu_experts.fetch_add(1, Ordering::Relaxed);
            } else {
                let cpu_started = self.sparse_profile.start();
                // Every routed expert is evaluated even when the device cap
                // cannot admit a complete packed triplet.
                for (name, output) in [
                    (&names[0], &mut sparse.cpu_gate),
                    (&names[1], &mut sparse.cpu_up),
                ] {
                    let key = self.source.tensor(name)?.expert(expert)?.cache_key();
                    if let Some(view) = staged.get(&key) {
                        view.matvec_into(&sparse.cpu_input, output)?;
                    } else {
                        let lease = if admit {
                            self.source.expert_cached(name, expert)?
                        } else {
                            self.source.expert_cached_for_prefill(name, expert)?
                        };
                        lease.tensor().matvec_into(&sparse.cpu_input, output)?;
                    }
                }
                math::swiglu(&mut sparse.cpu_gate, &sparse.cpu_up)?;
                let output = &mut sparse.cpu_output[slot * n..(slot + 1) * n];
                let key = self.source.tensor(&names[2])?.expert(expert)?.cache_key();
                if let Some(view) = staged.get(&key) {
                    view.matvec_into(&sparse.cpu_gate, output)?;
                } else {
                    let lease = if admit {
                        self.source.expert_cached(&names[2], expert)?
                    } else {
                        self.source.expert_cached_for_prefill(&names[2], expert)?
                    };
                    lease.tensor().matvec_into(&sparse.cpu_gate, output)?;
                }
                self.cpu_experts.fetch_add(1, Ordering::Relaxed);
                if self.sparse_profile.enabled {
                    profile.cpu_experts.fetch_add(1, Ordering::Relaxed);
                }
                SparseProfile::finish(cpu_started, &profile.cpu_expert_ns);
                cpu_slots.push(slot);
            }
        }
        let groups = elements_groups(&sparse.commands, n)?;
        append(
            &mut sparse.commands,
            &mut graph,
            Push::new(10).n(0, n)?.n(1, slots)?,
            bindings(
                &sparse.dummy,
                &sparse.dummy,
                &sparse.output,
                &sparse.dummy,
                &sparse.projected,
                &sparse.dummy,
                &sparse.selection,
                &s.status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        if !cpu_slots.is_empty() {
            sparse.upload.write(0, bytes_of_f32(&sparse.cpu_output))?;
            for slot in cpu_slots {
                unsafe {
                    self.device.raw.cmd_copy_buffer(
                        sparse.commands.raw,
                        sparse.upload.raw,
                        sparse.output.raw,
                        &[vk::BufferCopy {
                            src_offset: (slot * n * 4) as u64,
                            dst_offset: (slot * n * 4) as u64,
                            size: (n * 4) as u64,
                        }],
                    );
                }
                self.device
                    .uploads
                    .fetch_add((n * 4) as u64, Ordering::Relaxed);
            }
            sparse.commands.transfer_to_compute();
        }
        SparseProfile::finish(started, &profile.graph_ns);
        Ok(graph.into())
    }

    fn packed_experts(
        &self,
        s: &GpuSequence,
        sparse: &mut SparseSequence,
        layer: usize,
        names: &[String; 3],
        weights: &[Option<Arc<Buffer>>],
    ) -> Result<Arc<[GraphOp]>> {
        let slots = self.config.experts_per_token;
        let ff = self.config.expert_feed_forward_length;
        let n = self.config.embedding_length;
        let direct = self.device.direct_experts && slots <= 8;
        #[cfg(test)]
        let direct = direct && self.device.diagnostic_pipeline_mode != super::context::DiagnosticPipelineMode::AllArrayPacked;
        // Isolate shader compilation from descriptor indirection in an ignored
        // hardware test; this diagnostic is absent from production builds.
        #[cfg(test)]
        let direct = direct && std::env::var("NEXTORA_DIRECT_PACKED_CONTROL").as_deref() != Ok("1");
        sparse.packed_leases = weights
            .iter()
            .map(|buffer| Arc::clone(buffer.as_ref().unwrap()))
            .collect();
        if !direct {
            // New cache entries were copied earlier in this command buffer. Make
            // those writes available as transfer sources for the contiguous packs.
            unsafe {
                self.device.raw.cmd_pipeline_barrier(
                    sparse.commands.raw,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                    &[],
                    &[],
                );
                for (matrix, name) in names.iter().enumerate() {
                    let shape = &self.shapes[name];
                    let bytes = shape.rows * shape.row_bytes;
                    for slot in 0..slots {
                        self.device.raw.cmd_copy_buffer(
                            sparse.commands.raw,
                            weights[slot * 3 + matrix].as_ref().unwrap().raw,
                            sparse.packed_weights[matrix].raw,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: (slot * bytes) as u64,
                                size: bytes as u64,
                            }],
                        );
                    }
                    self.packed_bytes
                        .fetch_add((bytes * slots) as u64, Ordering::Relaxed);
                }
            }
            sparse.commands.transfer_to_compute();
        } else {
            let bytes: usize = names
                .iter()
                .map(|name| {
                    let shape = &self.shapes[name];
                    shape.rows * shape.row_bytes * slots
                })
                .sum();
            self.direct_layers.fetch_add(1, Ordering::Relaxed);
            self.direct_source_bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
        }
        self.gpu_experts.fetch_add(slots as u64, Ordering::Relaxed);
        self.packed_layers.fetch_add(1, Ordering::Relaxed);
        if self.reuse_packed_graphs {
            if let Some(graph) = &sparse.packed_graphs[layer] {
                self.packed_graph_reuses.fetch_add(1, Ordering::Relaxed);
                if direct {
                    Self::update_expert_arrays(&sparse.commands, graph, weights, slots)?;
                }
                return Ok(Arc::clone(graph));
            }
        } else {
            sparse.dynamic_descriptors_used = true;
        }
        let commands = if self.reuse_packed_graphs {
            &mut sparse.packed_commands
        } else {
            &mut sparse.commands
        };
        let mut graph = Vec::with_capacity(if self.device.fast_math { 7 } else { 5 });
        for matrix in 0..2 {
            let shape = &self.shapes[&names[matrix]];
            let mut groups = commands.groups(shape.rows)?;
            groups[2] = slots;
            let quant = sparse
                .quant
                .as_ref()
                .filter(|_| matches!(shape.ty, 12 | 14) && shape.width % 256 == 0);
            if let Some(quant) = quant {
                self.quantize(
                    commands,
                    &mut graph,
                    quant,
                    &sparse.input,
                    &s.status,
                    0,
                    shape.width,
                )?;
            }
            append(
                commands,
                &mut graph,
                Push::new(0)
                    .n(0, shape.width)?
                    .n(1, shape.rows)?
                    .n(2, shape.ty as usize)?
                    .n(3, shape.row_bytes)?
                    .n(4, slots)?
                    .flags(1 | if quant.is_some() { 32 } else { 0 } | if direct { 512 } else { 0 }),
                bindings(
                    &sparse.dummy,
                    if direct {
                        &sparse.dummy
                    } else {
                        &sparse.packed_weights[matrix]
                    },
                    &sparse.input,
                    quant.map(|q| &q.packed).unwrap_or(&sparse.dummy),
                    if matrix == 0 {
                        &sparse.gate
                    } else {
                        &sparse.up
                    },
                    quant.map(|q| &q.scales).unwrap_or(&sparse.dummy),
                    &sparse.packed_selection,
                    &s.status,
                ),
                groups,
                Dynamic::Fixed,
            )?;
        }
        let groups = elements_groups(commands, ff * slots)?;
        append(
            commands,
            &mut graph,
            Push::new(6).n(0, ff * slots)?,
            bindings(
                &sparse.dummy,
                &sparse.dummy,
                &sparse.gate,
                &sparse.up,
                &sparse.gate,
                &sparse.dummy,
                &sparse.dummy,
                &s.status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        let shape = &self.shapes[&names[2]];
        let quant = sparse
            .quant
            .as_ref()
            .filter(|_| matches!(shape.ty, 12 | 14) && shape.width % 256 == 0);
        if let Some(quant) = quant {
            self.quantize(
                commands,
                &mut graph,
                quant,
                &sparse.gate,
                &s.status,
                0,
                ff * slots,
            )?;
        }

        let mut groups = commands.groups(shape.rows)?;
        groups[2] = slots;
        append(
            commands,
            &mut graph,
            Push::new(0)
                .n(0, shape.width)?
                .n(1, shape.rows)?
                .n(2, shape.ty as usize)?
                .n(3, shape.row_bytes)?
                .n(4, slots)?
                .flags(3 | if quant.is_some() { 32 } else { 0 } | if direct { 512 } else { 0 }),
            bindings(
                &sparse.dummy,
                if direct {
                    &sparse.dummy
                } else {
                    &sparse.packed_weights[2]
                },
                &sparse.gate,
                quant.map(|q| &q.packed).unwrap_or(&sparse.dummy),
                &sparse.output,
                quant.map(|q| &q.scales).unwrap_or(&sparse.dummy),
                &sparse.packed_selection,
                &s.status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        let groups = elements_groups(commands, n)?;
        append(
            commands,
            &mut graph,
            Push::new(10).n(0, n)?.n(1, slots)?,
            bindings(
                &sparse.dummy,
                &sparse.dummy,
                &sparse.output,
                &sparse.dummy,
                &sparse.projected,
                &sparse.dummy,
                &sparse.selection,
                &s.status,
            ),
            groups,
            Dynamic::Fixed,
        )?;
        self.packed_graph_builds.fetch_add(1, Ordering::Relaxed);
        let graph: Arc<[GraphOp]> = graph.into();
        if direct {
            Self::update_expert_arrays(&sparse.commands, &graph, weights, slots)?;
        }
        if self.reuse_packed_graphs {
            sparse.packed_graphs[layer] = Some(Arc::clone(&graph));
        }
        Ok(graph)
    }

    fn update_expert_arrays(
        commands: &Commands,
        graph: &[GraphOp],
        weights: &[Option<Arc<Buffer>>],
        slots: usize,
    ) -> Result<()> {
        let matrices: Vec<_> = graph
            .iter()
            .filter(|op| op.operation.push.0[0] == 0)
            .collect();
        ensure!(
            matrices.len() == 3,
            "direct expert graph requires gate/up/down matrices"
        );
        for (matrix, op) in matrices.into_iter().enumerate() {
            let sources: Vec<_> = (0..slots)
                .map(|slot| Arc::clone(weights[slot * 3 + matrix].as_ref().unwrap()))
                .collect();
            commands.update_weight_slots(op.operation.set, &sources)?;
        }
        Ok(())
    }

    fn copy_output(&self, commands: &Commands, s: &GpuSequence, last: bool) {
        commands.compute_to_transfer();
        let logits = s.vocabulary * 4;
        let hidden = s.embedding * 4;
        unsafe {
            if last {
                self.device.raw.cmd_copy_buffer(
                    commands.raw,
                    s.logits.raw,
                    s.readback.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: logits as u64,
                    }],
                );
                self.device.raw.cmd_copy_buffer(
                    commands.raw,
                    s.hidden.raw,
                    s.readback.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: logits as u64,
                        size: hidden as u64,
                    }],
                );
            }
            self.device.raw.cmd_copy_buffer(
                commands.raw,
                s.status.raw,
                s.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: (logits + hidden) as u64,
                    size: 4,
                }],
            );
        }
        commands.transfer_to_host();
    }

    fn submit_sparse_layer(&self, commands: &mut Commands, poll: bool) -> Result<()> {
        if !poll || self.routing_poll_budget.is_zero() {
            return commands.submit();
        }
        let (signaled, elapsed) = commands.submit_routing(self.routing_poll_budget)?;
        let counter = if signaled { &self.routing_poll_hits } else { &self.routing_poll_misses };
        counter.fetch_add(1, Ordering::Relaxed);
        self.routing_poll_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn execute_sparse(&self, s: &mut GpuSequence, tokens: &[u32], out: &mut Step) -> Result<()> {
        let mut sparse = s
            .sparse
            .take()
            .context("ordinary sparse sequence missing")?;
        let profile = self.sparse_profile.phase(tokens.len() == 1);
        let result = (|| -> Result<()> {
            for (index, &token) in tokens.iter().enumerate() {
                let token_started = self.sparse_profile.start();
                let mut stage_started = self.sparse_profile.start();
                self.ensure_capacity(s, s.position + 1)?;
                self.prepare_rope_tables(s, 1)?;
                let final_token = index + 1 == tokens.len();
                let first = *sparse
                    .routes
                    .first()
                    .context("ordinary sparse routing stage missing")?;
                s.commands.begin()?;
                unsafe {
                    self.device
                        .raw
                        .cmd_fill_buffer(s.commands.raw, s.status.raw, 0, 4, 0)
                };
                self.record_rope_upload(&s.commands, s, 1);
                s.commands.transfer_to_compute();
                Self::record_range(&s.commands, &s.forward[..=first], token, s.position);
                self.copy_routing(&s.commands, s, &sparse);
                SparseProfile::finish(stage_started, &profile.record_ns);
                stage_started = self.sparse_profile.start();
                let poll = tokens.len() == 1 && s.position < 2048;
                self.submit_sparse_layer(&mut s.commands, poll)?;
                SparseProfile::finish(stage_started, &profile.submit_ns);
                stage_started = self.sparse_profile.start();
                self.read_routing(&mut sparse)?;
                SparseProfile::finish(stage_started, &profile.routing_ns);
                for layer in 0..sparse.routes.len() {
                    let graph = self.experts(s, &mut sparse, layer, tokens.len() == 1)?;
                    stage_started = self.sparse_profile.start();
                    Self::record_range(&sparse.commands, &graph, token, s.position);
                    let end = sparse
                        .routes
                        .get(layer + 1)
                        .map(|n| n + 1)
                        .unwrap_or(s.forward.len());
                    let begin = sparse.routes[layer] + 1;
                    Self::record_range(&sparse.commands, &s.forward[begin..end], token, s.position);
                    if layer + 1 < sparse.routes.len() {
                        self.copy_routing(&sparse.commands, s, &sparse);
                    } else {
                        if final_token {
                            Self::record_range(&sparse.commands, &s.project, token, s.position);
                        }
                        self.copy_output(&sparse.commands, s, final_token);
                    }
                    SparseProfile::finish(stage_started, &profile.record_ns);
                    stage_started = self.sparse_profile.start();
                    // The last layer performs the vocabulary projection and
                    // returns logits to the host, so use its regular wait.
                    self.submit_sparse_layer(&mut sparse.commands, poll && layer + 1 < sparse.routes.len())?;
                    SparseProfile::finish(stage_started, &profile.submit_ns);
                    stage_started = self.sparse_profile.start();
                    drop(graph);
                    sparse.packed_leases.clear();
                    if let Some(prepared) = sparse.pending_upload.take() {
                        self.cache.as_ref().unwrap().commit_prepared(prepared)?;
                    }
                    SparseProfile::finish(stage_started, &profile.commit_ns);
                    stage_started = self.sparse_profile.start();
                    if layer + 1 < sparse.routes.len() {
                        self.read_routing(&mut sparse)?;
                    }
                    SparseProfile::finish(stage_started, &profile.routing_ns);
                    if self.sparse_profile.enabled {
                        profile.layers.fetch_add(1, Ordering::Relaxed);
                    }
                }
                let mut status = [0u8; 4];
                s.readback
                    .read((s.vocabulary + s.embedding) * 4, &mut status)?;
                ensure!(
                    u32::from_ne_bytes(status) == 0,
                    "ordinary GPU forward produced a non-finite intermediate"
                );
                if final_token {
                    out.logits.resize(s.vocabulary, 0.0);
                    out.hidden.resize(s.embedding, 0.0);
                    s.readback.read(0, bytes_of_f32_mut(&mut out.logits))?;
                    s.readback
                        .read(s.vocabulary * 4, bytes_of_f32_mut(&mut out.hidden))?;
                    ensure!(
                        out.logits.iter().chain(&out.hidden).all(|x| x.is_finite()),
                        "ordinary GPU output is non-finite"
                    );
                    self.device.readbacks.fetch_add(
                        ((s.vocabulary + s.embedding) * 4 + 4) as u64,
                        Ordering::Relaxed,
                    );
                } else {
                    self.device.readbacks.fetch_add(4, Ordering::Relaxed);
                }
                s.position += 1;
                SparseProfile::finish(token_started, &profile.total_ns);
                if self.sparse_profile.enabled {
                    profile.tokens.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        })();
        s.sparse = Some(sparse);
        result
    }
    fn execute_chunk(
        &self,
        s: &mut GpuSequence,
        tokens: &[u32],
        last: bool,
        out: &mut Step,
    ) -> Result<()> {
        self.prepare_rope_tables(s, tokens.len())?;
        s.commands.begin()?;
        unsafe {
            self.device
                .raw
                .cmd_fill_buffer(s.commands.raw, s.status.raw, 0, 4, 0)
        };
        self.record_rope_upload(&s.commands, s, tokens.len());
        s.commands.transfer_to_compute();
        for (offset, &token) in tokens.iter().enumerate() {
            let position = s.position + offset;
            for op in &s.forward {
                let mut push = op.operation.push.clone();
                let mut groups = op.operation.groups;
                match op.dynamic {
                    Dynamic::Fixed | Dynamic::Experts => {}
                    Dynamic::Embedding => push.0[2] = token,
                    Dynamic::Rope => {
                        push.0[3] = position as u32;
                        if push.0[15] & 256 != 0 {
                            push.0[10] = (offset * self.config.head_dim) as u32;
                        }
                    }
                    Dynamic::Store => push.0[2] = position as u32,
                    Dynamic::Scores => {
                        push.0[1] = (position + 1) as u32;
                        groups[0] = (position + 1) as u32;
                    }
                    Dynamic::Attention => push.0[1] = (position + 1) as u32,
                    Dynamic::CompactScores => {
                        push.0[1] = (position + 1) as u32;
                        groups[0] = (position + 1).div_ceil(1024) as u32;
                    }
                    Dynamic::CompactAttention => push.0[1] = (position + 1).div_ceil(1024) as u32,
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

#[cfg(test)]
mod memory_tests {
    use super::*;
    use crate::quant::QuantType;

    fn tensor(dims: Vec<u64>, byte_len: usize) -> TensorInfo {
        TensorInfo {
            name: "memory-plan-test".into(),
            dims,
            ty: QuantType::F16,
            offset: 0,
            byte_len,
            file_offset: 0,
            shard_index: 0,
        }
    }

    #[test]
    fn payload_preflight_accounts_for_f32_vector_expansion() {
        let tensors = [tensor(vec![1024], 2048), tensor(vec![1024, 4], 8192)];
        assert_eq!(resident_weight_bytes(&tensors).unwrap(), 4096 + 8192);
        let overflowing = [tensor(vec![u64::MAX], 0)];
        assert!(resident_weight_bytes(&overflowing).is_err());
    }

    #[test]
    fn oversized_resident_checkpoint_explains_explicit_hybrid_recovery() {
        // The 30B Q4 checkpoint that failed after uploading almost the full GPU
        // budget must instead be rejected before any tensor is uploaded.
        let error = check_resident_weight_budget(18_556_686_304, 11_967_188_172)
            .unwrap_err()
            .to_string();
        assert!(error.contains("17.28 GiB"));
        assert!(error.contains("11.15 GiB"));
        assert!(error.contains("--backend hybrid"));
        assert!(error.contains("request state need additional memory"));
        assert!(check_resident_weight_budget(1024, 2048).is_ok());
    }
    fn sparse_fixture(context: u32) -> Arc<Gguf> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe.gguf");
        let mut source = Gguf::open(path).unwrap();
        source.meta.insert(
            "qwen3moe.context_length".into(),
            crate::gguf::Value::U32(context),
        );
        Arc::new(source)
    }
    fn compare(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && (a - b).abs() <= 0.002 + 0.0003 * b.abs(),
                "GPU {a} CPU {b}"
            );
        }
    }

    #[test]
    #[ignore = "requires explicit Vulkan execution"]
    fn software_vulkan_ordinary_sparse_grows_kv_and_preserves_prefix() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let source = sparse_fixture(40960);
        let cpu = Model::load(Arc::clone(&source), 40960).unwrap();
        let gpu = GpuModel::load(
            source,
            40960,
            DeviceOptions {
                allow_software: true,
                memory_budget_bytes: 8 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let mut cs = cpu.new_sequence();
        let mut gs = gpu.new_sequence().unwrap();
        assert_eq!(gs.kv_capacity, 256);
        let tokens: Vec<u32> = (0..257).map(|i| [2, 19, 77, 5][i % 4]).collect();
        let expected = cpu.prefill(&mut cs, &tokens).unwrap();
        let actual = gpu.prefill(&mut gs, &tokens).unwrap();
        assert_eq!(gs.kv_capacity, 512);
        compare(&actual.logits, &expected.logits);
        compare(&actual.hidden, &expected.hidden);
        for layer in 0..cpu.config().block_count {
            let (k, v) = gs.read_kv(layer).unwrap();
            let (ck, cv) = cs.kv(layer).unwrap();
            compare(&k, ck);
            compare(&v, cv);
        }
        let expected = cpu.step(&mut cs, 8).unwrap();
        let actual = gpu.step(&mut gs, 8).unwrap();
        compare(&actual.logits, &expected.logits);
        let mut other = gpu.new_sequence().unwrap();
        assert_eq!(other.kv_capacity, 256);
        gpu.step(&mut other, 11).unwrap();
        let expected = cpu.step(&mut cs, 12).unwrap();
        let actual = gpu.step(&mut gs, 12).unwrap();
        compare(&actual.logits, &expected.logits);
        assert_eq!(gpu.memory_stats()["selected_experts_omitted"], 0);
    }

    #[test]
    #[ignore = "requires explicit Vulkan execution"]
    fn software_vulkan_ordinary_sparse_cache_pressure_retains_every_expert() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let source = sparse_fixture(32);
        let cpu = Model::load(Arc::clone(&source), 32).unwrap();
        let gpu = GpuModel::load(
            source,
            32,
            DeviceOptions {
                allow_software: true,
                memory_budget_bytes: 4 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let mut cs = cpu.new_sequence();
        let mut gs = gpu.new_sequence().unwrap();
        let c = cpu.step(&mut cs, 2).unwrap();
        let g = gpu.step(&mut gs, 2).unwrap();
        compare(&g.logits, &c.logits);
        let cached = gpu.cache.as_ref().unwrap().stats()["cached_packed_weight_bytes"]
            .as_u64()
            .unwrap();
        assert!(cached > 0);
        let room = gpu.device.budget - gpu.device.reserved() + cached - 4096;
        gpu.cache
            .as_ref()
            .unwrap()
            .make_room(room as usize)
            .unwrap();
        let pressure = gpu.device.storage(room as usize).unwrap();
        assert!(
            gpu.cache.as_ref().unwrap().stats()["cache_evictions"]
                .as_u64()
                .unwrap()
                > 0
        );
        let before = gpu.cpu_experts.load(Ordering::Relaxed);
        for token in [19, 77, 5] {
            let c = cpu.step(&mut cs, token).unwrap();
            let g = gpu.step(&mut gs, token).unwrap();
            compare(&g.logits, &c.logits);
            compare(&g.hidden, &c.hidden);
        }
        assert_eq!(
            gpu.cpu_experts.load(Ordering::Relaxed) - before,
            (3 * gpu.config.block_count * gpu.config.experts_per_token) as u64
        );
        assert!(gpu.device.reserved() <= gpu.device.budget);
        assert_eq!(gpu.memory_stats()["selected_experts_omitted"], 0);
        drop(pressure);
        let c = cpu.step(&mut cs, 6).unwrap();
        let g = gpu.step(&mut gs, 6).unwrap();
        compare(&g.logits, &c.logits);
    }
    #[test]
    #[ignore = "requires explicit Vulkan integer-dot execution"]
    fn software_vulkan_ordinary_q8_graph_quantization_and_offsets() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let source = sparse_fixture(512);
        let mut gpu = GpuModel::load(
            source,
            512,
            DeviceOptions {
                allow_software: true,
                fast_math: true,
                memory_budget_bytes: 32 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nextora-q8-graph-{}-{stamp}.gguf",
            std::process::id()
        ));
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GGUF");
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&1u64.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&5u64.to_le_bytes());
        raw.extend_from_slice(b"probe");
        raw.extend_from_slice(&2u32.to_le_bytes());
        raw.extend_from_slice(&256u64.to_le_bytes());
        raw.extend_from_slice(&5u64.to_le_bytes());
        raw.extend_from_slice(&12u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.resize(raw.len().next_multiple_of(32), 0);
        for row in 0..5 {
            let mut block = vec![1u8; 144];
            block[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
            block[2..4].copy_from_slice(&0u16.to_le_bytes());
            for (i, q) in block[16..].iter_mut().enumerate() {
                *q = (i * 17 + row * 29) as u8;
            }
            raw.extend_from_slice(&block);
        }
        std::fs::write(&path, raw).unwrap();
        let model = Gguf::open(&path).unwrap();
        let view = model.tensor("probe").unwrap();
        let input: Vec<f32> = (0..512).map(|i| 0.1 + (i % 31) as f32 / 31.0).collect();
        let input_gpu = gpu.device.upload(bytes_of_f32(&input)).unwrap();
        let output = gpu.device.storage(10 * 4).unwrap();
        let readback = gpu.device.host(10 * 4).unwrap();
        let status = gpu.device.storage(4).unwrap();
        let packed = gpu.device.upload(view.bytes()).unwrap();
        gpu.weights.insert("probe".into(), Arc::clone(&packed));
        gpu.shapes.insert(
            "probe".into(),
            Shape {
                width: 256,
                rows: 5,
                experts: 1,
                ty: 12,
                row_bytes: 144,
            },
        );
        let quant = QuantScratch {
            packed: gpu.device.storage(512).unwrap(),
            scales: gpu.device.storage(16 * 4).unwrap(),
        };
        let mut commands = Commands::new(Arc::clone(&gpu.device), 2).unwrap();
        let mut graph = Vec::new();
        gpu.matrix(
            &mut commands,
            &mut graph,
            "probe",
            &input_gpu,
            &output,
            &status,
            &status,
            &status,
            false,
            false,
            Some(&quant),
        )
        .unwrap();
        assert_eq!(graph.len(), 2);
        assert_eq!(graph[0].operation.push.0[0], 16);
        assert_ne!(graph[1].operation.push.0[15] & 32, 0);
        commands.begin().unwrap();
        unsafe {
            gpu.device
                .raw
                .cmd_fill_buffer(commands.raw, status.raw, 0, 4, 0)
        };
        commands.transfer_to_compute();
        GpuModel::record_range(&commands, &graph, 0, 0);
        commands.compute_to_transfer();
        unsafe {
            gpu.device.raw.cmd_copy_buffer(
                commands.raw,
                output.raw,
                readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: 20,
                }],
            );
        }
        commands.transfer_to_host();
        commands.submit().unwrap();
        let mut actual = vec![0.0; 5];
        readback.read(0, bytes_of_f32_mut(&mut actual)).unwrap();
        let expected = view.matvec(&input[..256]).unwrap();
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!(
                (a - e).abs() <= 0.01 * e.abs() + 0.03,
                "Q8graph {a} F32 {e}"
            );
        }
        // Exercise the nonzero offsets used by single-expert GPU fallbacks.
        let mut sequence = gpu.new_sequence().unwrap();
        let mut sparse = sequence.sparse.take().unwrap();
        sparse.quant = Some(quant);
        let mut graph = Vec::new();
        gpu.expert_matrix(
            &mut sparse,
            &mut graph,
            view,
            packed,
            Arc::clone(&input_gpu),
            Arc::clone(&output),
            &status,
            256,
            5,
        )
        .unwrap();
        assert_eq!(graph[0].operation.push.0[10], 256);
        assert_eq!(graph[0].operation.push.0[11], 64);
        assert_eq!(graph[0].operation.push.0[13], 8);
        sparse.commands.begin().unwrap();
        unsafe {
            gpu.device
                .raw
                .cmd_fill_buffer(sparse.commands.raw, status.raw, 0, 4, 0)
        };
        sparse.commands.transfer_to_compute();
        GpuModel::record_range(&sparse.commands, &graph, 0, 0);
        sparse.commands.compute_to_transfer();
        unsafe {
            gpu.device.raw.cmd_copy_buffer(
                sparse.commands.raw,
                output.raw,
                readback.raw,
                &[vk::BufferCopy {
                    src_offset: 20,
                    dst_offset: 20,
                    size: 20,
                }],
            );
        }
        sparse.commands.transfer_to_host();
        sparse.commands.submit().unwrap();
        readback.read(20, bytes_of_f32_mut(&mut actual)).unwrap();
        let expected = view.matvec(&input[256..]).unwrap();
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!(
                (a - e).abs() <= 0.01 * e.abs() + 0.03,
                "Q8offset {a} F32 {e}"
            );
        }
        drop(model);
        let _ = std::fs::remove_file(path);
    }
    fn wide_q8_fixture() -> (std::path::PathBuf, Arc<Gguf>) {
        use crate::gguf::Value as V;
        fn string(s: &str, out: &mut Vec<u8>) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        fn value(v: &V) -> (u32, Vec<u8>) {
            match v {
                V::U32(n) => (4, n.to_le_bytes().to_vec()),
                V::F32(n) => (6, n.to_le_bytes().to_vec()),
                V::String(s) => {
                    let mut out = Vec::new();
                    string(s, &mut out);
                    (8, out)
                }
                V::Array(items) => {
                    let encoded: Vec<_> = items.iter().map(value).collect();
                    let ty = encoded.first().map(|v| v.0).unwrap_or(8);
                    let mut out = ty.to_le_bytes().to_vec();
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for (kind, bytes) in encoded {
                        assert_eq!(kind, ty);
                        out.extend(bytes);
                    }
                    (9, out)
                }
                _ => panic!("unsupported synthetic fixture metadata"),
            }
        }
        let mut meta = sparse_fixture(512).meta.clone();
        for (suffix, n) in [
            ("embedding_length", 256),
            ("feed_forward_length", 256),
            ("attention.head_count", 4),
            ("attention.head_count_kv", 4),
            ("attention.key_length", 64),
            ("rope.dimension_count", 64),
            ("expert_feed_forward_length", 256),
        ] {
            meta.insert(format!("qwen3moe.{suffix}"), V::U32(n));
        }
        let vocab = meta["tokenizer.ggml.tokens"].as_array().unwrap().len();
        let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
        fn tensor(
            tensors: &mut Vec<(String, Vec<usize>, u32, Vec<u8>)>,
            name: &str,
            dims: &[usize],
            ty: u32,
            salt: usize,
        ) {
            let elements: usize = dims.iter().product();
            let mut data = Vec::new();
            if ty == 0 {
                for i in 0..elements {
                    let f = if dims.len() == 1 {
                        1.0
                    } else {
                        0.5 + ((i * 3 + salt * 7) % 13) as f32 * 0.025
                    };
                    data.extend_from_slice(&f.to_le_bytes());
                }
            } else {
                for row in 0..elements / 256 {
                    let mut block = vec![1u8; if ty == 12 { 144 } else { 210 }];
                    if ty == 12 {
                        block[..2].copy_from_slice(
                            &half::f16::from_f32(0.015625).to_bits().to_le_bytes(),
                        );
                        block[2..4].fill(0);
                        for (i, q) in block[16..].iter_mut().enumerate() {
                            *q = (i * 17 + row * 29 + salt) as u8;
                        }
                    } else {
                        for (i, q) in block[..128].iter_mut().enumerate() {
                            *q = (i * 19 + row * 13 + salt) as u8;
                        }
                        block[128..192].fill(0x55);
                        block[192..208].fill(1);
                        block[208..].copy_from_slice(
                            &half::f16::from_f32(0.015625).to_bits().to_le_bytes(),
                        );
                    }
                    data.extend(block);
                }
            }
            tensors.push((name.into(), dims.to_vec(), ty, data));
        }
        tensor(&mut tensors, "token_embd.weight", &[256, vocab], 0, 0);
        tensor(&mut tensors, "output.weight", &[256, vocab], 12, 3);
        tensor(&mut tensors, "output_norm.weight", &[256], 0, 0);
        for name in ["attn_norm", "ffn_norm"] {
            tensor(&mut tensors, &format!("blk.0.{name}.weight"), &[256], 0, 0);
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            tensor(&mut tensors, &format!("blk.0.{name}.weight"), &[64], 0, 0);
        }
        for (i, name) in ["attn_q", "attn_k", "attn_v", "attn_output"]
            .iter()
            .enumerate()
        {
            tensor(
                &mut tensors,
                &format!("blk.0.{name}.weight"),
                &[256, 256],
                12,
                i,
            );
        }
        tensor(&mut tensors, "blk.0.ffn_gate_inp.weight", &[256, 4], 0, 3);
        tensor(
            &mut tensors,
            "blk.0.ffn_gate_exps.weight",
            &[256, 256, 4],
            12,
            5,
        );
        tensor(
            &mut tensors,
            "blk.0.ffn_up_exps.weight",
            &[256, 256, 4],
            12,
            9,
        );
        tensor(
            &mut tensors,
            "blk.0.ffn_down_exps.weight",
            &[256, 256, 4],
            14,
            11,
        );
        let mut raw = b"GGUF".to_vec();
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        raw.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        for (key, v) in meta {
            string(&key, &mut raw);
            let (ty, bytes) = value(&v);
            raw.extend_from_slice(&ty.to_le_bytes());
            raw.extend(bytes);
        }
        let mut payload = Vec::new();
        for (name, dims, ty, data) in tensors {
            string(&name, &mut raw);
            raw.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                raw.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            raw.extend_from_slice(&ty.to_le_bytes());
            raw.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            payload.extend(data);
            payload.resize(payload.len().next_multiple_of(32), 0);
        }
        raw.resize(raw.len().next_multiple_of(32), 0);
        raw.extend(payload);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nextora-q8-model-{}-{stamp}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, raw).unwrap();
        let source = Arc::new(Gguf::open(&path).unwrap());
        (path, source)
    }

    #[derive(Debug)]
    struct Q8Reference;
    impl crate::model::MatvecExecutor for Q8Reference {
        fn name(&self) -> &'static str {
            "q8-activation-reference"
        }
        fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
            if !matches!(tensor.quant_type().id(), 12 | 14) {
                return tensor.matvec_into(input, output);
            }
            let mut quantized = input.to_vec();
            for group in quantized.chunks_mut(32) {
                let peak = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let scale = peak / 127.0;
                for x in group {
                    *x = if scale > 0.0 {
                        (*x / scale).round_ties_even().clamp(-127.0, 127.0) * scale
                    } else {
                        0.0
                    };
                }
            }
            tensor.matvec_into(&quantized, output)
        }
    }

    #[test]
    #[ignore = "requires explicit Vulkan integer-dot execution"]
    fn software_vulkan_ordinary_q8_full_packed_moe_graph() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let (path, source) = wide_q8_fixture();
        let cpu =
            Model::load_with_executor(Arc::clone(&source), 512, Arc::new(Q8Reference)).unwrap();
        let gpu = GpuModel::load(
            source,
            512,
            DeviceOptions {
                allow_software: true,
                fast_math: true,
                memory_budget_bytes: 32 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        assert!(gpu.shapes.values().any(|s| s.ty == 12));
        assert!(gpu.shapes.values().any(|s| s.ty == 14));
        let mut cs = cpu.new_sequence();
        let mut gs = gpu.new_sequence().unwrap();
        assert!(gs.forward.iter().any(|op| op.operation.push.0[0] == 16));
        assert!(
            gs.forward
                .iter()
                .any(|op| op.operation.push.0[0] == 0 && op.operation.push.0[15] & 32 != 0)
        );
        for token in [2, 19, 77] {
            let expected = cpu.step(&mut cs, token).unwrap();
            let actual = gpu.step(&mut gs, token).unwrap();
            compare(&actual.logits, &expected.logits);
            compare(&actual.hidden, &expected.hidden);
            let (k, v) = gs.read_kv(0).unwrap();
            let (ck, cv) = cs.kv(0).unwrap();
            compare(&k, ck);
            compare(&v, cv);
        }
        assert_eq!(gpu.packed_layers.load(Ordering::Relaxed), 3);
        let grouped = gpu.memory_stats()["grouped_dispatches"].as_u64().unwrap();
        if gpu.device.group_compute_barriers {
            assert!(grouped >= 9, "QKV and independent expert projections should share barriers");
        } else {
            assert_eq!(grouped, 0);
        }
        if !gpu.routing_poll_budget.is_zero() {
            let polls = gpu.routing_poll_hits.load(Ordering::Relaxed)
                + gpu.routing_poll_misses.load(Ordering::Relaxed);
            assert_eq!(polls, 3, "one routing fence per single-layer decode token");
            assert!(gpu.routing_poll_ns.load(Ordering::Relaxed) > 0);
        }
        if gpu.device.direct_experts {
            assert_eq!(gpu.direct_layers.load(Ordering::Relaxed), 3);
            assert!(gpu.direct_source_bytes.load(Ordering::Relaxed) > 0);
            assert_eq!(gpu.packed_bytes.load(Ordering::Relaxed), 0);
            assert!(gs.sparse.as_ref().unwrap().packed_leases.is_empty());
        }
        if gpu.reuse_packed_graphs {
            assert_eq!(gpu.packed_graph_builds.load(Ordering::Relaxed), 1);
            assert_eq!(gpu.packed_graph_reuses.load(Ordering::Relaxed), 2);
            assert!(!gs.sparse.as_ref().unwrap().dynamic_descriptors_used);
        }
        if gpu.sparse_profile.enabled {
            assert_eq!(gpu.memory_stats()["sparse_profile"]["decode"]["tokens"], 3);
            assert!(
                gpu.memory_stats()["sparse_profile"]["decode"]["submit_and_fence_ms"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            );
        }
        assert_eq!(gpu.memory_stats()["selected_experts_omitted"], 0);
        drop(gs);
        drop(gpu);
        drop(cpu);
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn compact_512k_memory_plan_includes_scales_and_growth_overlap() {
        let model = Model::load(sparse_fixture(524288), 524288).unwrap();
        let mut config = model.config().clone();
        config.block_count = 48;
        config.head_count = 32;
        config.head_count_kv = 4;
        config.head_dim = 128;
        let row = super::super::context::KvPrecision::Q4_64
            .row_bytes(512)
            .unwrap();
        assert_eq!(row, 272);
        let plan = KvMemoryPlan::new(&config, row, true).unwrap();
        assert_eq!(plan.full_payload, 13_690_208_256);
        assert_eq!(plan.growth_overlap, 142_606_336);
        assert_eq!(plan.attention_scratch, 8_519_680);
        let fixed = 1024 * 1024 * 1024;
        let exact = plan.minimum_peak(fixed).unwrap();
        assert!(
            plan.check(fixed, exact - 1)
                .unwrap_err()
                .to_string()
                .contains("all KV history")
        );
        assert_eq!(plan.check(fixed, exact).unwrap(), exact);
    }

    #[test]
    fn compact_snapshot_decodes_signed_codes_and_independent_scales() {
        let mut raw = vec![0u8; 68 * 2];
        for (row, scales) in [(0, [0.5f32, 0.25f32]), (1, [1.0, 2.0])] {
            for (group, scale) in scales.into_iter().enumerate() {
                raw[row * 68 + group * 2..row * 68 + group * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            }
            for word in 0..16 {
                raw[row * 68 + 4 + word * 4..row * 68 + 8 + word * 4]
                    .copy_from_slice(&0x79f0_79f0u32.to_le_bytes());
            }
        }
        let decoded = decode_q4_64_prefix(&raw, 2, 128).unwrap();
        assert_eq!(&decoded[..4], &[0.0, -0.5, -3.5, 3.5]);
        assert_eq!(&decoded[64..68], &[0.0, -0.25, -1.75, 1.75]);
        assert_eq!(&decoded[128..132], &[0.0, -1.0, -7.0, 7.0]);
        assert_eq!(&decoded[192..196], &[0.0, -2.0, -14.0, 14.0]);
        assert!(decode_q4_64_prefix(&raw[..68], 2, 128).is_err());
    }

    fn raw_gpu_prefix(sequence: &mut GpuSequence, buffer: Arc<Buffer>, bytes: usize) -> Vec<u8> {
        let device = Arc::clone(&sequence.commands.device);
        let host = device.host(bytes).unwrap();
        sequence.commands.begin().unwrap();
        sequence.commands.compute_to_transfer();
        unsafe {
            device.raw.cmd_copy_buffer(
                sequence.commands.raw,
                buffer.raw,
                host.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                }],
            );
        }
        sequence.commands.transfer_to_host();
        sequence.commands.submit().unwrap();
        let mut raw = vec![0; bytes];
        host.read(0, &mut raw).unwrap();
        raw
    }

    #[test]
    #[ignore = "requires explicit Vulkan compact-cache execution"]
    fn software_vulkan_ordinary_compact_kv_growth_preserves_packed_prefix() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let gpu = GpuModel::load(
            sparse_fixture(512),
            512,
            DeviceOptions {
                allow_software: true,
                kv_precision: super::super::context::KvPrecision::Q4_64,
                memory_budget_bytes: 16 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let mut sequence = gpu.new_sequence().unwrap();
        let tokens: Vec<_> = (0..256).map(|i| [2, 19, 77, 5][i % 4]).collect();
        gpu.prefill(&mut sequence, &tokens).unwrap();
        assert_eq!(sequence.kv_capacity, 256);
        let bytes = gpu.kv_plan.row_bytes * 256;
        let mut prefixes = Vec::new();
        for layer in 0..gpu.config.block_count {
            let key = Arc::clone(&sequence.keys[layer]);
            let value = Arc::clone(&sequence.values[layer]);
            prefixes.push((
                raw_gpu_prefix(&mut sequence, key, bytes),
                raw_gpu_prefix(&mut sequence, value, bytes),
            ));
        }
        gpu.step(&mut sequence, 8).unwrap();
        assert_eq!(sequence.kv_capacity, 512);
        assert_eq!(sequence.position, 257);
        for (layer, (key_prefix, value_prefix)) in prefixes.into_iter().enumerate() {
            let key = Arc::clone(&sequence.keys[layer]);
            let value = Arc::clone(&sequence.values[layer]);
            assert_eq!(raw_gpu_prefix(&mut sequence, key, bytes), key_prefix);
            assert_eq!(raw_gpu_prefix(&mut sequence, value, bytes), value_prefix);
            let (keys, values) = sequence.read_kv(layer).unwrap();
            assert_eq!(keys.len(), sequence.position * sequence.kv_width);
            assert!(keys.iter().chain(&values).all(|v| v.is_finite()));
        }
        sequence.reset().unwrap();
        let reset = gpu.step(&mut sequence, 11).unwrap();
        let mut fresh = gpu.new_sequence().unwrap();
        let expected = gpu.step(&mut fresh, 11).unwrap();
        compare(&reset.logits, &expected.logits);
        compare(&reset.hidden, &expected.hidden);
        assert_eq!(gpu.memory_stats()["selected_experts_omitted"], 0);
    }

    #[test]
    #[ignore = "requires explicit Vulkan compact-cache execution"]
    fn software_vulkan_ordinary_compact_allocates_and_attends_full_512k_history() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        const CONTEXT: usize = 524288;
        let gpu = GpuModel::load(
            sparse_fixture(CONTEXT as u32),
            CONTEXT,
            DeviceOptions {
                allow_software: true,
                kv_precision: super::super::context::KvPrecision::Q4_64,
                memory_budget_bytes: 384 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let mut sequence = gpu.new_sequence().unwrap();
        assert_eq!(sequence.kv_capacity, 256);
        gpu.ensure_capacity(&mut sequence, CONTEXT).unwrap();
        assert_eq!(sequence.kv_capacity, CONTEXT);
        assert_eq!(
            sequence.keys[0].size as usize,
            gpu.kv_plan.row_bytes * CONTEXT
        );
        let first = sequence
            .forward
            .iter()
            .position(|op| op.operation.push.0[0] == 17)
            .unwrap();
        let query = Arc::clone(&sequence.forward[first].operation.buffers[1]);
        let attention = Arc::clone(&sequence.forward[first + 1].operation.buffers[3]);
        let keys = Arc::clone(&sequence.keys[0]);
        let values = Arc::clone(&sequence.values[0]);
        let bytes = values.size as usize;
        // Uniform zero keys make every history position equiprobable. The first
        // half contains value1 and the second half value0; attending only recent
        // positions would return0, whereas the full-history answer is0.5.
        let mut raw = vec![0u8; bytes];
        for token in 0..CONTEXT {
            for block in 0..sequence.kv_width.div_ceil(128) {
                let start = token * gpu.kv_plan.row_bytes + block * 68;
                raw[start..start + 4].copy_from_slice(&0x3800_3800u32.to_le_bytes());
                let code = if token < CONTEXT / 2 {
                    0x2222_2222u32
                } else {
                    0
                };
                for word in 0..16 {
                    raw[start + 4 + word * 4..start + 8 + word * 4]
                        .copy_from_slice(&code.to_le_bytes());
                }
            }
        }
        let host = gpu.device.host(bytes).unwrap();
        host.write(0, &raw).unwrap();
        drop(raw);
        sequence.commands.begin().unwrap();
        unsafe {
            gpu.device
                .raw
                .cmd_fill_buffer(sequence.commands.raw, keys.raw, 0, keys.size, 0);
            gpu.device.raw.cmd_fill_buffer(
                sequence.commands.raw,
                query.raw,
                0,
                query.size,
                1.0f32.to_bits(),
            );
            gpu.device
                .raw
                .cmd_fill_buffer(sequence.commands.raw, sequence.status.raw, 0, 4, 0);
            gpu.device.raw.cmd_copy_buffer(
                sequence.commands.raw,
                host.raw,
                values.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                }],
            );
        }
        sequence.commands.transfer_to_compute();
        GpuModel::record_range(
            &sequence.commands,
            &sequence.forward[first..first + 2],
            0,
            CONTEXT - 1,
        );
        sequence.commands.compute_to_transfer();
        unsafe {
            gpu.device.raw.cmd_copy_buffer(
                sequence.commands.raw,
                attention.raw,
                sequence.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: attention.size,
                }],
            );
        }
        sequence.commands.transfer_to_host();
        sequence.commands.submit().unwrap();
        let mut result = vec![0.0f32; gpu.config.head_count * gpu.config.head_dim];
        sequence
            .readback
            .read(0, bytes_of_f32_mut(&mut result))
            .unwrap();
        assert!(
            result
                .iter()
                .all(|v| v.is_finite() && (v - 0.5).abs() < 0.0001),
            "{result:?}"
        );
        assert_eq!(gpu.memory_stats()["configured_context_tokens"], CONTEXT);
        assert!(gpu.full_context_minimum_peak.load(Ordering::Relaxed) <= gpu.device.budget);
    }

    #[test]
    #[ignore = "requires explicit Vulkan compact-cache allocation"]
    fn software_vulkan_ordinary_compact_rejects_insufficient_full_context_budget() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let error = GpuModel::load(
            sparse_fixture(524288),
            524288,
            DeviceOptions {
                allow_software: true,
                kv_precision: super::super::context::KvPrecision::Q4_64,
                memory_budget_bytes: 8 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("full context requires"), "{error}");
    }

    #[test]
    #[ignore = "requires explicit Vulkan compact-cache allocation"]
    fn software_vulkan_ordinary_compact_load_validates_actual_fixed_state_budget() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let source = sparse_fixture(524288);
        let checked = Model::load(Arc::clone(&source), 524288).unwrap();
        let config = checked.config();
        let row = super::super::context::KvPrecision::Q4_64
            .row_bytes(config.head_count_kv * config.head_dim)
            .unwrap();
        let plan = KvMemoryPlan::new(config, row, true).unwrap();
        let dense = resident_weight_bytes(
            source
                .tensors
                .iter()
                .filter(|(name, _)| !routed_expert(name))
                .map(|(_, info)| info),
        )
        .unwrap();
        let lower_bound = plan.minimum_peak(dense + plan.attention_scratch).unwrap();
        // This clears the early payload-only check, but cannot hold the real
        // persistent GPU activations and packed-expert working buffers.
        let budget = lower_bound + 1;
        assert!(plan.check(dense + plan.attention_scratch, budget).is_ok());
        let error = GpuModel::load(
            source,
            524288,
            DeviceOptions {
                allow_software: true,
                kv_precision: super::super::context::KvPrecision::Q4_64,
                memory_budget_bytes: usize::try_from(budget).unwrap(),
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("memory validation failed"), "{error}");
        assert!(error.contains("full context requires"), "{error}");
        assert!(error.contains("fixed state"), "{error}");
    }

    fn yarn_graph_fixture(moe: bool, context: u32, original: u32) -> Arc<Gguf> {
        let name = if moe {
            "tiny-qwen3moe.gguf"
        } else {
            "tiny-qwen3.gguf"
        };
        let architecture = if moe { "qwen3moe" } else { "qwen3" };
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let mut source = Gguf::open(path).unwrap();
        source.meta.insert(
            format!("{architecture}.context_length"),
            crate::gguf::Value::U32(context),
        );
        source.meta.insert(
            format!("{architecture}.rope.scaling.type"),
            crate::gguf::Value::String("yarn".into()),
        );
        source.meta.insert(
            format!("{architecture}.rope.scaling.factor"),
            crate::gguf::Value::F32(context as f32 / original as f32),
        );
        source.meta.insert(
            format!("{architecture}.rope.scaling.original_context_length"),
            crate::gguf::Value::U32(original),
        );
        Arc::new(source)
    }

    #[test]
    #[ignore = "requires explicit Vulkan CPU-table RoPE execution"]
    fn software_vulkan_ordinary_host_rope_tables_match_cpu_across_prefill_and_growth() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        for moe in [false, true] {
            let source = yarn_graph_fixture(moe, 512, 128);
            let cpu = Model::load(Arc::clone(&source), 512).unwrap();
            let gpu = GpuModel::load(
                source,
                512,
                DeviceOptions {
                    allow_software: true,
                    memory_budget_bytes: 32 * 1024 * 1024,
                    ..Default::default()
                },
                &|_, _, _, _, _| {},
            )
            .unwrap();
            let mut cs = cpu.new_sequence();
            let mut gs = gpu.new_sequence().unwrap();
            let table = Arc::clone(&gs.rope.as_ref().unwrap().device);
            for op in gs
                .forward
                .iter()
                .filter(|op| matches!(op.dynamic, Dynamic::Rope))
            {
                assert_ne!(op.operation.push.0[15] & 256, 0);
                assert!(Arc::ptr_eq(&op.operation.buffers[2], &table));
            }
            let tokens: Vec<_> = (0..256).map(|i| [2, 19, 77, 5][i % 4]).collect();
            let expected = cpu.prefill(&mut cs, &tokens).unwrap();
            let actual = gpu.prefill(&mut gs, &tokens).unwrap();
            compare(&actual.logits, &expected.logits);
            compare(&actual.hidden, &expected.hidden);
            let expected = cpu.step(&mut cs, 8).unwrap();
            let actual = gpu.step(&mut gs, 8).unwrap();
            compare(&actual.logits, &expected.logits);
            compare(&actual.hidden, &expected.hidden);
            assert_eq!(gs.position, 257);
            for layer in 0..gpu.config.block_count {
                let (keys, values) = gs.read_kv(layer).unwrap();
                let (expected_keys, expected_values) = cs.kv(layer).unwrap();
                compare(&keys, expected_keys);
                compare(&values, expected_values);
            }
            let half = gpu.config.head_dim / 2;
            let mut cosine = vec![0.0; half];
            let mut sine = vec![0.0; half];
            math::rope_angles_scaled(
                256,
                gpu.rope_frequencies.as_ref().unwrap(),
                gpu.config.rope_attention_factor(),
                &mut cosine,
                &mut sine,
            )
            .unwrap();
            let uploaded = raw_gpu_prefix(&mut gs, Arc::clone(&table), gpu.config.head_dim * 4);
            let expected_bytes = bytes_of_f32(&[cosine, sine].concat()).to_vec();
            assert_eq!(uploaded, expected_bytes);
        }
    }

    #[test]
    #[ignore = "requires explicit Vulkan CPU-table RoPE execution"]
    fn software_vulkan_ordinary_host_rope_tables_rotate_at_524287() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let gpu = GpuModel::load(
            yarn_graph_fixture(true, 524288, 32768),
            524288,
            DeviceOptions {
                allow_software: true,
                kv_precision: super::super::context::KvPrecision::Q4_64,
                memory_budget_bytes: 384 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let mut gs = gpu.new_sequence().unwrap();
        gs.position = 524287;
        gpu.prepare_rope_tables(&mut gs, 1).unwrap();
        let index = gs
            .forward
            .iter()
            .position(|op| matches!(op.dynamic, Dynamic::Rope))
            .unwrap();
        let input_buffer = Arc::clone(&gs.forward[index].operation.buffers[1]);
        let input: Vec<_> = (0..gpu.config.head_count * gpu.config.head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03125)
            .collect();
        let mut expected = input.clone();
        let row = &gs.rope.as_ref().unwrap().values[..gpu.config.head_dim];
        let (cosine, sine) = row.split_at(gpu.config.head_dim / 2);
        math::rope_heads(&mut expected, gpu.config.head_dim, cosine, sine).unwrap();
        let upload = gpu.device.host(input.len() * 4).unwrap();
        upload.write(0, bytes_of_f32(&input)).unwrap();
        gs.commands.begin().unwrap();
        unsafe {
            gpu.device
                .raw
                .cmd_fill_buffer(gs.commands.raw, gs.status.raw, 0, 4, 0);
            gpu.device.raw.cmd_copy_buffer(
                gs.commands.raw,
                upload.raw,
                input_buffer.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (input.len() * 4) as u64,
                }],
            );
        }
        gpu.record_rope_upload(&gs.commands, &gs, 1);
        gs.commands.transfer_to_compute();
        GpuModel::record_range(&gs.commands, &gs.forward[index..index + 1], 0, gs.position);
        gs.commands.compute_to_transfer();
        unsafe {
            gpu.device.raw.cmd_copy_buffer(
                gs.commands.raw,
                input_buffer.raw,
                gs.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (input.len() * 4) as u64,
                }],
            );
        }
        gs.commands.transfer_to_host();
        gs.commands.submit().unwrap();
        let mut actual = vec![0.0; input.len()];
        gs.readback.read(0, bytes_of_f32_mut(&mut actual)).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && (a - b).abs() < 1e-6, "GPU{a} CPU{b}");
        }
    }

    #[test]
    #[ignore = "requires explicit Vulkan RAM KV execution"]
    fn software_vulkan_ordinary_ram_kv_preserves_history_and_gpu_parity() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let source = sparse_fixture(512);
        let base = DeviceOptions {
            allow_software: true,
            kv_precision: super::super::context::KvPrecision::Q4_64,
            memory_budget_bytes: 16 * 1024 * 1024,
            host_memory_budget_bytes: 32 * 1024 * 1024,
            ..Default::default()
        };
        let gpu =
            GpuModel::load(Arc::clone(&source), 512, base.clone(), &|_, _, _, _, _| {}).unwrap();
        let ram = GpuModel::load(
            source,
            512,
            DeviceOptions {
                kv_placement: KvPlacement::Ram,
                ..base
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let mut gs = gpu.new_sequence().unwrap();
        let mut rs = ram.new_sequence().unwrap();
        assert!(
            rs.keys
                .iter()
                .chain(&rs.values)
                .all(|buffer| buffer.is_system_ram())
        );
        assert!(
            gs.keys
                .iter()
                .chain(&gs.values)
                .all(|buffer| !buffer.is_system_ram())
        );
        let tokens: Vec<_> = (0..256).map(|i| [2, 19, 77, 5][i % 4]).collect();
        let expected = gpu.prefill(&mut gs, &tokens).unwrap();
        let actual = ram.prefill(&mut rs, &tokens).unwrap();
        compare(&actual.logits, &expected.logits);
        compare(&actual.hidden, &expected.hidden);
        let bytes = ram.kv_plan.row_bytes * 256;
        let first = Arc::clone(&rs.keys[0]);
        let prefix = raw_gpu_prefix(&mut rs, first, bytes);
        let gpu_before = ram.device.reserved();
        let host_before = ram.device.host_reserved();
        ram.ensure_capacity(&mut rs, 512).unwrap();
        assert_eq!(ram.device.reserved(), gpu_before);
        assert!(ram.device.host_reserved() > host_before);
        let first = Arc::clone(&rs.keys[0]);
        assert_eq!(raw_gpu_prefix(&mut rs, first, bytes), prefix);
        let expected = gpu.step(&mut gs, 8).unwrap();
        let actual = ram.step(&mut rs, 8).unwrap();
        compare(&actual.logits, &expected.logits);
        compare(&actual.hidden, &expected.hidden);
        for layer in 0..gpu.config.block_count {
            let (gk, gv) = gs.read_kv(layer).unwrap();
            let (rk, rv) = rs.read_kv(layer).unwrap();
            compare(&rk, &gk);
            compare(&rv, &gv);
        }
        assert_eq!(ram.memory_stats()["kv_placement"], "ram");
        assert!(
            ram.full_context_ram_peak.load(Ordering::Relaxed)
                > ram.full_context_minimum_peak.load(Ordering::Relaxed)
        );
    }

    #[test]
    #[ignore = "requires explicit Vulkan full RAM context allocation"]
    fn software_vulkan_ordinary_ram_kv_full_capacity_probe_uses_independent_ram_budget() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let gpu = GpuModel::load(
            sparse_fixture(524288),
            524288,
            DeviceOptions {
                allow_software: true,
                kv_precision: super::super::context::KvPrecision::Q4_64,
                kv_placement: KvPlacement::Ram,
                memory_budget_bytes: 8 * 1024 * 1024,
                host_memory_budget_bytes: 384 * 1024 * 1024,
                ..Default::default()
            },
            &|_, _, _, _, _| {},
        )
        .unwrap();
        let report = gpu.probe_capacity().unwrap();
        assert_eq!(report["capacity_proof"], "full_kv_allocated_and_touched");
        assert_eq!(report["context_tokens"], 524288);
        assert_eq!(report["kv_buffers_on_system_ram"], true);
        assert_eq!(report["history_tokens_processed"], 0);
        assert!(
            report["memory"]["host_live_allocation_bytes"]
                .as_u64()
                .unwrap()
                > gpu.device.budget
        );
        assert!(gpu.device.reserved() <= gpu.device.budget);
        assert!(gpu.device.host_reserved() < gpu.kv_plan.full_payload);
    }
}

#[cfg(test)]
#[path = "direct_trajectory_tests.rs"]
mod direct_trajectory_tests;

#[cfg(test)]
#[path = "direct_layer_trace_tests.rs"]
mod direct_layer_trace_tests;
