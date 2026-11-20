//! First-party Vulkan execution for ordinary Qwen3 and Qwen3.8/qwen4exp.
//! Resident graphs keep activations and sequence state on the device. Qwen3.8
//! shares a bounded packed expert cache, batches cache-miss transfers and
//! computes every selected contribution even when the budget requires a CPU
//! expert fallback. A separate projection executor supports explicit hybrid
//! execution. None of these paths depends on Strata or imports its GPU shaders.
mod context;
mod facade;
mod hybrid;
mod qwen38;
mod resident;

pub use context::{DeviceOptions, discover, select_device};
pub use facade::{GpuModel, GpuSequence};
pub use hybrid::GpuMatvecExecutor;
pub use qwen38::{QwenGpuModel, QwenGpuSequence};

pub fn supports_quant(id: u32) -> bool {
    matches!(
        id,
        0 | 1 | 2 | 8 | 12 | 13 | 14 | 18 | 20 | 21 | 22 | 23 | 30 | 42
    )
}

#[cfg(test)]
mod tests;
