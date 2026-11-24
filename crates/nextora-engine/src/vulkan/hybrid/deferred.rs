//! Packed expert uploads recorded beside their consuming GPU operations.
use super::*;
use crate::gguf::TensorKey;

#[derive(Debug)]
pub(crate) struct PreparedBatch {
    pub weights: Vec<Option<Arc<Buffer>>>,
    pending: Vec<(TensorKey, Arc<Buffer>)>,
    bytes: usize,
    admit: bool,
}

impl GpuMatvecExecutor {
    /// The caller begins `commands` first and retains the returned object until
    /// its submission fence. Pending cache entries are published only afterwards.
    pub(crate) fn prepare_batch_recorded(
        &self,
        tensors: &[Tensor<'_>],
        admit: bool,
        commands: &Commands,
        staging: &Arc<Buffer>,
    ) -> Result<PreparedBatch> {
        let total = tensors.iter().try_fold(0usize, |sum, tensor| {
            sum.checked_add(tensor.bytes().len().next_multiple_of(4))
                .context("deferred expert upload overflow")
        })?;
        if total > staging.size as usize {
            return Ok(PreparedBatch {
                weights: if admit {
                    self.prepare_batch(tensors, true)?
                } else {
                    self.prepare_batch_prefill_free(tensors)?
                },
                pending: Vec::new(),
                bytes: 0,
                admit,
            });
        }
        let may_evict = admit || self.prefill_needs_gpu_weights();
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        let mut weights = vec![None; tensors.len()];
        let mut pending = Vec::new();
        let mut used = 0;
        let mut bytes = 0;
        for (index, tensor) in tensors.iter().enumerate() {
            state.clock = state.clock.wrapping_add(1);
            if let Some(buffer) = state.cached(tensor.cache_key(), admit, false) {
                weights[index] = Some(buffer);
                state.counts.hits += 1;
                continue;
            }
            state.counts.misses += 1;
            let length = tensor.bytes().len();
            if !super::super::supports_quant(tensor.quant_type().id())
                || length > self.device.properties.limits.max_storage_buffer_range as usize
                || length > u32::MAX as usize
            {
                continue;
            }
            let buffer = loop {
                if self
                    .device
                    .reserved()
                    .saturating_add(length as u64)
                    .saturating_add(4096)
                    > self.device.budget
                {
                    if may_evict && state.evict_one() {
                        continue;
                    }
                    break None;
                }
                match self.device.storage(length) {
                    Ok(buffer) => break Some(buffer),
                    Err(error) if error.downcast_ref::<BudgetExceeded>().is_some() => {
                        if may_evict && state.evict_one() {
                            continue;
                        }
                        break None;
                    }
                    Err(error) => return Err(error),
                }
            };
            let Some(buffer) = buffer else { continue };
            let padded = length.next_multiple_of(4);
            staging.write(used, tensor.bytes())?;
            if padded > length {
                staging.write(used + length, &[0u8; 3][..padded - length])?;
            }
            unsafe {
                self.device.raw.cmd_copy_buffer(
                    commands.raw,
                    staging.raw,
                    buffer.raw,
                    &[vk::BufferCopy {
                        src_offset: used as u64,
                        dst_offset: 0,
                        size: padded as u64,
                    }],
                );
            }
            used += padded;
            bytes += length;
            weights[index] = Some(Arc::clone(&buffer));
            pending.push((tensor.cache_key(), buffer));
        }
        if bytes > 0 {
            commands.transfer_to_compute();
        }
        Ok(PreparedBatch {
            weights,
            pending,
            bytes,
            admit,
        })
    }

    /// Called only after the consuming submission completed successfully.
    pub(crate) fn commit_prepared(&self, batch: PreparedBatch) -> Result<()> {
        if batch.bytes == 0 {
            return Ok(());
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU projection cache lock poisoned"))?;
        state.counts.batch_uploads += 1;
        self.device
            .uploads
            .fetch_add(batch.bytes as u64, Ordering::Relaxed);
        for (key, buffer) in batch.pending {
            if batch.admit || self.device.reserved() <= self.device.budget * 9 / 10 {
                state.publish_upload(key, buffer, batch.admit);
            } else {
                state.counts.transient_uploads += 1;
            }
        }
        Ok(())
    }
}
