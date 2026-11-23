//! Bounded packed-weight cache shared by the resident Qwen3.8 graph and the
//! explicit CPU-state/GPU-projection hybrid accelerator.
use super::context::{
    BudgetExceeded, Buffer, Commands, Device, DeviceOptions, Operation, Push, bytes_of_f32,
    bytes_of_f32_mut, float_bytes,
};
use crate::{
    gguf::{Gguf, Tensor, TensorKey},
    model::MatvecExecutor,
};
use anyhow::{Context, Result, ensure};
use ash::vk;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::Ordering},
};

#[derive(Debug)]
struct Cached {
    buffer: Arc<Buffer>,
    last_use: u64,
    pinned: bool,
}
#[derive(Debug, Default)]
struct Counters {
    hits: u64,
    misses: u64,
    evictions: u64,
    gpu_calls: u64,
    cpu_small: u64,
    cpu_format: u64,
    cpu_budget: u64,
    pin_skips: u64,
    batch_uploads: u64,
    transient_uploads: u64,
}
#[derive(Debug)]
struct Io {
    commands: Commands,
    operation: Operation,
    input: Arc<Buffer>,
    output: Arc<Buffer>,
    upload: Arc<Buffer>,
    readback: Arc<Buffer>,
    status: Arc<Buffer>,
    dummy: Arc<Buffer>,
    width: usize,
    rows: usize,
}
#[derive(Debug, Default)]
struct State {
    cache: HashMap<TensorKey, Cached>,
    clock: u64,
    io: Option<Io>,
    counts: Counters,
}
#[derive(Debug)]
pub struct GpuMatvecExecutor {
    device: Arc<Device>,
    state: Mutex<State>,
}

impl State {
    fn evict_one(&mut self) -> bool {
        let oldest = self
            .cache
            .iter()
            .filter(|(_, v)| !v.pinned && Arc::strong_count(&v.buffer) == 1)
            .min_by_key(|(_, v)| v.last_use)
            .map(|(k, _)| *k);
        if let Some(key) = oldest {
            self.cache.remove(&key);
            self.counts.evictions += 1;
            true
        } else {
            false
        }
    }
    fn cached_payload(&self) -> u64 {
        self.cache.values().map(|v| v.buffer.size).sum()
    }
    fn pinned_payload(&self) -> u64 {
        self.cache
            .values()
            .filter(|v| v.pinned)
            .map(|v| v.buffer.size)
            .sum()
    }
}
impl GpuMatvecExecutor {
    pub fn new(options: DeviceOptions) -> Result<Self> {
        Ok(Self {
            device: Device::new(&options)?,
            state: Mutex::new(State::default()),
        })
    }
    pub(crate) fn device(&self) -> Arc<Device> {
        Arc::clone(&self.device)
    }
    pub(crate) fn cached_weight(
        &self,
        key: TensorKey,
        refresh: bool,
    ) -> Result<Option<Arc<Buffer>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        state.clock = state.clock.wrapping_add(1);
        let tick = state.clock;
        if let Some(hit) = state.cache.get_mut(&key) {
            if refresh {
                hit.last_use = tick;
            }
            let buffer = Arc::clone(&hit.buffer);
            state.counts.hits += 1;
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }
    #[cfg(test)]
    pub(crate) fn cache_recencies(&self) -> std::collections::BTreeMap<TensorKey, u64> {
        self.state
            .lock()
            .unwrap()
            .cache
            .iter()
            .map(|(key, entry)| (*key, entry.last_use))
            .collect()
    }
    pub(crate) fn make_room(&self, bytes: usize) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        while self.device.reserved().saturating_add(bytes as u64) > self.device.budget {
            if !state.evict_one() {
                return Err(BudgetExceeded(format!(
                    "Vulkan fixed/state memory needs {bytes} bytes beyond available owned budget"
                ))
                .into());
            }
        }
        Ok(())
    }
    /// Upload several selected packed matrices using bounded staging windows.
    /// A fence protects each reused staging window, not each matrix. Returned
    /// immutable leases remain valid through the caller's compute submission.
    /// Prefill misses never evict existing cache entries or enter the cache.
    /// Synchronous compatibility path for callers without independent CPU work.
    #[allow(dead_code)]
    pub(crate) fn prepare_batch(
        &self,
        tensors: &[Tensor<'_>],
        admit: bool,
    ) -> Result<Vec<Option<Arc<Buffer>>>> {
        self.prepare_batch_with_host_work(tensors, admit, |_, _| Ok(()))
    }

    /// Overlap CPU-only fallback with the final outstanding Vulkan upload batch.
    /// Prepared GPU buffer handles are for classification only until this returns:
    /// they must not be submitted for compute before the transfer fence completes.
    /// The callback runs exactly once, outside the cache mutex, including on
    /// cache hits and out-of-budget paths. A callback failure still joins the
    /// outstanding GPU submission before any staging or output buffer is dropped.
    pub(crate) fn prepare_batch_with_host_work(
        &self,
        tensors: &[Tensor<'_>],
        admit: bool,
        host_work: impl FnOnce(&[Option<Arc<Buffer>>], bool) -> Result<()>,
    ) -> Result<Vec<Option<Arc<Buffer>>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        let mut result = vec![None; tensors.len()];
        let mut misses = Vec::new();
        for (i, tensor) in tensors.iter().enumerate() {
            state.clock = state.clock.wrapping_add(1);
            let tick = state.clock;
            if let Some(hit) = state.cache.get_mut(&tensor.cache_key()) {
                if admit {
                    hit.last_use = tick;
                }
                result[i] = Some(Arc::clone(&hit.buffer));
                state.counts.hits += 1;
            } else {
                state.counts.misses += 1;
                if super::supports_quant(tensor.quant_type().id())
                    && tensor.bytes().len()
                        <= self.device.properties.limits.max_storage_buffer_range as usize
                    && tensor.bytes().len() <= u32::MAX as usize
                {
                    misses.push(i);
                }
            }
        }
        if misses.is_empty() {
            drop(state);
            host_work(&result, false)?;
            return Ok(result);
        }
        let bytes = misses.iter().try_fold(0usize, |sum, &i| {
            sum.checked_add(tensors[i].bytes().len())
                .context("expert upload batch size overflow")
        })?;
        // Reserve the transfer scratch before admitting weights so temporary
        // staging cannot violate the same owned-allocation budget.
        let mut staging_bytes = bytes.min(16 * 1024 * 1024).next_multiple_of(4).max(4);
        let staging = loop {
            if self
                .device
                .reserved()
                .saturating_add(staging_bytes as u64)
                .saturating_add(64 * 1024)
                > self.device.budget
            {
                if admit && state.evict_one() {
                    continue;
                }
                if staging_bytes > 4096 {
                    staging_bytes = (staging_bytes / 2).max(4096).next_multiple_of(4);
                    continue;
                }
                drop(state);
                host_work(&result, false)?;
                return Ok(result);
            }
            match self.device.host(staging_bytes) {
                Ok(buffer) => break buffer,
                Err(error) if error.downcast_ref::<BudgetExceeded>().is_some() => {
                    if admit && state.evict_one() {
                        continue;
                    }
                    drop(state);
                    host_work(&result, false)?;
                    return Ok(result);
                }
                Err(error) => return Err(error),
            }
        };
        let mut pending = Vec::new();
        for i in misses {
            let length = tensors[i].bytes().len();
            loop {
                if self
                    .device
                    .reserved()
                    .saturating_add(length as u64)
                    .saturating_add(4096)
                    > self.device.budget
                {
                    if admit && state.evict_one() {
                        continue;
                    }
                    break;
                }
                match self.device.storage(length) {
                    Ok(buffer) => {
                        pending.push((i, buffer));
                        break;
                    }
                    Err(error) if error.downcast_ref::<BudgetExceeded>().is_some() => {
                        if admit && state.evict_one() {
                            continue;
                        }
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        if pending.is_empty() {
            drop(state);
            host_work(&result, false)?;
            return Ok(result);
        }
        // These are immutable handles to allocated destinations, not evidence
        // that their transfer has completed. CPU-only fallback can use the
        // allocation map while Vulkan is writing to those buffers.
        for (i, buffer) in &pending {
            result[*i] = Some(Arc::clone(buffer));
        }
        let mut commands = Commands::new(Arc::clone(&self.device), 1)?;
        commands.begin()?;
        let mut used = 0usize;
        let mut copied_total = 0usize;
        for (index, destination) in &pending {
            let data = tensors[*index].bytes();
            let mut offset = 0usize;
            while offset < data.len() {
                if used == staging_bytes {
                    commands.transfer_to_compute();
                    commands.submit()?;
                    commands.begin()?;
                    used = 0;
                }
                let count = (data.len() - offset).min(staging_bytes - used);
                let padded = count.next_multiple_of(4);
                staging.write(used, &data[offset..offset + count])?;
                if padded > count {
                    staging.write(used + count, &[0u8; 3][..padded - count])?;
                }
                unsafe {
                    self.device.raw.cmd_copy_buffer(
                        commands.raw,
                        staging.raw,
                        destination.raw,
                        &[vk::BufferCopy {
                            src_offset: used as u64,
                            dst_offset: offset as u64,
                            size: padded as u64,
                        }],
                    )
                };
                offset += count;
                used += padded;
                copied_total += count;
            }
        }
        // Release shared cache state before CPU fallback, preventing lengthy
        // host computation from serialising all other inference requests.
        // A buffer held in pending/result is safe against cache eviction.
        drop(state);
        if used > 0 {
            commands.transfer_to_compute();
            commands.submit_with_host_work(|| host_work(&result, true))?;
        } else {
            host_work(&result, false)?;
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        self.device
            .uploads
            .fetch_add(copied_total as u64, Ordering::Relaxed);
        state.counts.batch_uploads += 1;
        for (i, buffer) in pending {
            if admit {
                state.clock = state.clock.wrapping_add(1);
                let tick = state.clock;
                // Another request may have inserted this same expert during
                // our transfer. Keep its cached object; this caller still owns
                // the completed upload through result and other Arc leases.
                state.cache.entry(tensors[i].cache_key()).or_insert_with(|| Cached {
                    buffer: Arc::clone(&buffer),
                    last_use: tick,
                    pinned: false,
                });
            } else {
                state.counts.transient_uploads += 1;
            }
            // result[i] already holds the prepared (now complete) GPU buffer.
        }
        Ok(result)
    }
    pub fn adapter_info(&self) -> Value {
        self.device.info()
    }
    pub fn stats(&self) -> Value {
        let mut report = self.device.stats();
        match self.state.lock() {
            Ok(s) => {
                report["execution"] = json!("hybrid_cpu_state_gpu_projections");
                report["cached_packed_weight_bytes"] = json!(s.cached_payload());
                report["pinned_dense_bytes"] = json!(s.pinned_payload());
                report["cached_tensors"] = json!(s.cache.len());
                report["cache_hits"] = json!(s.counts.hits);
                report["cache_misses"] = json!(s.counts.misses);
                report["cache_evictions"] = json!(s.counts.evictions);
                report["gpu_projection_calls"] = json!(s.counts.gpu_calls);
                report["cpu_small_projection_calls"] = json!(s.counts.cpu_small);
                report["cpu_unsupported_format_calls"] = json!(s.counts.cpu_format);
                report["cpu_budget_fallback_calls"] = json!(s.counts.cpu_budget);
                report["dense_pin_skips"] = json!(s.counts.pin_skips);
                report["synchronous_projection_transfers"] = json!(true);
                report["batched_expert_upload_calls"] = json!(s.counts.batch_uploads);
                report["transient_prefill_tensor_uploads"] = json!(s.counts.transient_uploads);
                report["transfer_compute_overlap"] = json!(false);
                report["weight_format_preserved"] = json!(true);
            }
            Err(_) => report["cache_lock_poisoned"] = json!(true),
        }
        report
    }
    pub fn memory_stats(&self) -> Value {
        self.stats()
    }
    /// Reserve reusable dense projections before expert admission. Embedding
    /// tables stay lazy and expert stacks are admitted as selected views. Dense
    /// pinning is bounded to 70% of the total budget, retaining room for expert
    /// residency and transfer/state buffers. Skips are explicit in the report.
    pub fn pin_dense(&self, source: &Gguf) -> Result<Value> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        let mut pinned = 0usize;
        let mut skipped = 0usize;
        for (name, info) in &source.tensors {
            if info.dims.len() != 2
                || name.contains("embd")
                || name.contains("embedding")
                || name.contains("_exps.")
            {
                continue;
            }
            let tensor = source.tensor(name)?;
            if tensor.elements() < 32768 {
                continue;
            }
            if !super::supports_quant(tensor.quant_type().id())
                || tensor.bytes().len()
                    > self.device.properties.limits.max_storage_buffer_range as usize
                || tensor.bytes().len() > u32::MAX as usize
            {
                skipped += 1;
                continue;
            }
            let key = tensor.cache_key();
            if let Some(hit) = state.cache.get_mut(&key) {
                hit.pinned = true;
                pinned += 1;
                continue;
            }
            if state
                .pinned_payload()
                .saturating_add(tensor.bytes().len() as u64)
                > self.device.budget.saturating_mul(7) / 10
            {
                skipped += 1;
                continue;
            }
            match self.admit(&mut state, &tensor, true)? {
                Some(_) => pinned += 1,
                None => skipped += 1,
            }
        }
        state.counts.pin_skips += skipped as u64;
        Ok(
            json!({"pinned_dense_matrices":pinned,"skipped_dense_matrices":skipped,"pinned_dense_bytes":state.pinned_payload(),"embedding_tables_loaded":false,"expert_stacks_loaded":false}),
        )
    }
    fn admit(
        &self,
        state: &mut State,
        tensor: &Tensor<'_>,
        pinned: bool,
    ) -> Result<Option<Arc<Buffer>>> {
        state.clock = state.clock.wrapping_add(1);
        let key = tensor.cache_key();
        if let Some(hit) = state.cache.get_mut(&key) {
            hit.last_use = state.clock;
            hit.pinned |= pinned;
            state.counts.hits += 1;
            return Ok(Some(Arc::clone(&hit.buffer)));
        }
        state.counts.misses += 1;
        let length = tensor.bytes().len();
        // Device::upload uses a bounded staging block, counted by the same
        // allocator. Leave alignment allowance so near-full caches can evict.
        let needed = (length as u64)
            .saturating_add(length.min(16 * 1024 * 1024) as u64)
            .saturating_add(128 * 1024);
        while self.device.reserved().saturating_add(needed) > self.device.budget {
            if !state.evict_one() {
                return Ok(None);
            }
        }
        loop {
            match self.device.upload(tensor.bytes()) {
                Ok(buffer) => {
                    state.cache.insert(
                        key,
                        Cached {
                            buffer: Arc::clone(&buffer),
                            last_use: state.clock,
                            pinned,
                        },
                    );
                    return Ok(Some(buffer));
                }
                Err(e) if e.downcast_ref::<BudgetExceeded>().is_some() => {
                    if !state.evict_one() {
                        return Ok(None);
                    }
                }
                Err(e) => return Err(e).context("upload packed projection tensor"),
            }
        }
    }
    fn ensure_io(&self, state: &mut State, width: usize, rows: usize) -> Result<bool> {
        if state
            .io
            .as_ref()
            .is_some_and(|io| io.width >= width && io.rows >= rows)
        {
            return Ok(true);
        }
        let (width, rows) = state
            .io
            .as_ref()
            .map(|io| (width.max(io.width), rows.max(io.rows)))
            .unwrap_or((width, rows));
        state.io = None;
        let input_bytes = float_bytes(width)?;
        let output_bytes = float_bytes(rows)?;
        let needed = input_bytes
            .checked_mul(2)
            .and_then(|n| output_bytes.checked_mul(2).and_then(|o| n.checked_add(o)))
            .and_then(|n| n.checked_add(256 * 1024))
            .context("GPU projection scratch overflow")?;
        while self.device.reserved().saturating_add(needed as u64) > self.device.budget {
            if !state.evict_one() {
                return Ok(false);
            }
        }
        let mut commands = Commands::new(Arc::clone(&self.device), 1)?;
        let input = self.device.storage(input_bytes)?;
        let output = self.device.storage(output_bytes)?;
        let upload = self.device.host(input_bytes)?;
        let readback = self.device.host(
            output_bytes
                .checked_add(4)
                .context("projection readback overflow")?,
        )?;
        let status = self.device.storage(4)?;
        let dummy = self.device.storage(4)?;
        let groups = commands.groups(rows)?;
        let operation = commands.operation(
            Push::new(0).n(0, width)?.n(1, rows)?,
            [
                Arc::clone(&dummy),
                Arc::clone(&input),
                Arc::clone(&dummy),
                Arc::clone(&output),
                Arc::clone(&dummy),
                Arc::clone(&dummy),
                Arc::clone(&status),
            ],
            groups,
        )?;
        state.io = Some(Io {
            commands,
            operation,
            input,
            output,
            upload,
            readback,
            status,
            dummy,
            width,
            rows,
        });
        Ok(true)
    }
    /// Force the GPU projection path for validation of small packed matrices.
    /// Production MatvecExecutor calls keep very small matrices on CPU to avoid
    /// making transfer/synchronization overhead dominate useful computation.
    pub fn matvec_gpu(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.run(tensor, input, output, true)
    }
    fn run(
        &self,
        tensor: &Tensor<'_>,
        input: &[f32],
        output: &mut [f32],
        force: bool,
    ) -> Result<()> {
        ensure!(
            (1..=2).contains(&tensor.dimensions().len()),
            "GPU projection requires a vector/one-row matrix or two-dimensional matrix view"
        );
        ensure!(
            input.len() == tensor.width() && output.len() == tensor.rows(),
            "GPU projection dimensions disagree"
        );
        ensure!(
            input.iter().all(|v| v.is_finite()),
            "GPU projection input contains non-finite values"
        );
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        if !super::supports_quant(tensor.quant_type().id())
            || tensor.bytes().len()
                > self.device.properties.limits.max_storage_buffer_range as usize
            || tensor.bytes().len() > u32::MAX as usize
        {
            ensure!(!force, "GPU tensor format/size is unsupported");
            state.counts.cpu_format += 1;
            drop(state);
            return tensor.matvec_into(input, output);
        }
        if !force && (tensor.elements() < 32768 || tensor.rows() < 16) {
            state.counts.cpu_small += 1;
            drop(state);
            return tensor.matvec_into(input, output);
        }
        if !self.ensure_io(&mut state, input.len(), output.len())? {
            ensure!(!force, "GPU projection buffers exceed owned memory budget");
            state.counts.cpu_budget += 1;
            drop(state);
            return tensor.matvec_into(input, output);
        }
        let Some(weight) = self.admit(&mut state, tensor, false)? else {
            ensure!(
                !force,
                "GPU packed tensor cannot fit the owned cache budget"
            );
            state.counts.cpu_budget += 1;
            drop(state);
            return tensor.matvec_into(input, output);
        };
        let io = state.io.as_mut().unwrap();
        let groups = io.commands.groups(output.len())?;
        let mut push = Push::new(0)
            .n(0, input.len())?
            .n(1, output.len())?
            .n(2, tensor.quant_type().id() as usize)?
            .n(3, tensor.quant_type().row_bytes(input.len())?)?
            .n(4, 1)?;
        push.0[14] = groups[0] as u32;
        io.operation.buffers[0] = weight;
        io.commands
            .update_set(io.operation.set, &io.operation.buffers);
        io.upload.write(0, bytes_of_f32(input))?;
        io.commands.begin()?;
        unsafe {
            self.device
                .raw
                .cmd_fill_buffer(io.commands.raw, io.status.raw, 0, 4, 0);
            self.device.raw.cmd_copy_buffer(
                io.commands.raw,
                io.upload.raw,
                io.input.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: float_bytes(input.len())? as u64,
                }],
            );
        }
        io.commands.transfer_to_compute();
        io.commands.record(
            &io.operation,
            &push,
            [groups[0] as u32, groups[1] as u32, groups[2] as u32],
        );
        io.commands.compute_to_transfer();
        let output_bytes = float_bytes(output.len())?;
        unsafe {
            self.device.raw.cmd_copy_buffer(
                io.commands.raw,
                io.output.raw,
                io.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: output_bytes as u64,
                }],
            );
            self.device.raw.cmd_copy_buffer(
                io.commands.raw,
                io.status.raw,
                io.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: output_bytes as u64,
                    size: 4,
                }],
            );
        }
        io.commands.transfer_to_host();
        io.commands.submit()?;
        let mut status = [0u8; 4];
        io.readback.read(output_bytes, &mut status)?;
        // The fence completed; remove the descriptor's lease so eviction of
        // this cache entry can actually free its memory before the next miss.
        io.operation.buffers[0] = Arc::clone(&io.dummy);
        io.commands
            .update_set(io.operation.set, &io.operation.buffers);
        ensure!(
            u32::from_ne_bytes(status) == 0,
            "GPU projection produced a non-finite result"
        );
        io.readback.read(0, bytes_of_f32_mut(output))?;
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "GPU projection returned non-finite data"
        );
        self.device
            .uploads
            .fetch_add(float_bytes(input.len())? as u64, Ordering::Relaxed);
        self.device
            .readbacks
            .fetch_add((output_bytes + 4) as u64, Ordering::Relaxed);
        state.counts.gpu_calls += 1;
        Ok(())
    }
}
impl MatvecExecutor for GpuMatvecExecutor {
    fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.run(tensor, input, output, false)
    }
    fn name(&self) -> &'static str {
        "nextora-native-hybrid-vulkan"
    }
}
