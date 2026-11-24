//! Conservative host-cache planning from actual tensor geometry and available RAM.
use crate::{gguf::Gguf, vulkan::{KvPlacement, KvPrecision}};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct HostMemoryPlan {
    pub available_bytes: u64,
    pub os_margin_bytes: u64,
    pub dense_page_allowance_bytes: u64,
    pub state_reserve_bytes: u64,
    pub state_slots: usize,
    pub ram_context_payload_bytes: u64,
    pub ram_context_peak_bytes: u64,
    pub vulkan_host_budget_bytes: u64,
    pub routed_expert_payload_bytes: u64,
    pub lazy_embedding_payload_bytes: u64,
    pub expert_cache_capacity_bytes: usize,
    pub policy: &'static str,
}

pub fn plan(source: &Gguf, context: usize, slots: usize, host_state: bool, explicit_mib: Option<usize>, margin_mib: usize) -> Result<HostMemoryPlan> {
    plan_with_kv(source, context, slots, host_state, explicit_mib, margin_mib,
        KvPrecision::F32, KvPlacement::Gpu, None)
}

pub fn plan_with_kv(source: &Gguf, context: usize, slots: usize, host_state: bool,
    explicit_mib: Option<usize>, margin_mib: usize, precision: KvPrecision,
    placement: KvPlacement, ram_cap_mib: Option<usize>) -> Result<HostMemoryPlan> {
    plan_available_kv(source, context, slots, host_state, explicit_mib, margin_mib,
        available_memory()?, precision, placement, ram_cap_mib)
}

// Existing CPU geometry remains independently testable without hardware/RAM allocation.
#[cfg(test)]
fn plan_available(source: &Gguf, context: usize, slots: usize, host_state: bool,
    explicit_mib: Option<usize>, margin_mib: usize, available: u64) -> Result<HostMemoryPlan> {
    plan_available_kv(source, context, slots, host_state, explicit_mib, margin_mib,
        available, KvPrecision::F32, KvPlacement::Gpu, None)
}
fn plan_available_kv(source: &Gguf, context: usize, slots: usize, host_state: bool,
    explicit_mib: Option<usize>, margin_mib: usize, available: u64,
    precision: KvPrecision, placement: KvPlacement, ram_cap_mib: Option<usize>) -> Result<HostMemoryPlan> {
    ensure!(
        (1..=4).contains(&slots),
        "memory plan requires 1..=4 sequence slots"
    );
    ensure!(
        (1..=1_048_576).contains(&context),
        "memory plan requires context in 1..=1048576"
    );
    let margin = u64::try_from(margin_mib)
        .context("RAM margin exceeds address range")?
        .checked_mul(1024 * 1024)
        .context("RAM margin overflow")?;
    let arch = source
        .meta
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .context("memory plan requires general.architecture")?;
    let mut experts = 0u64;
    let mut lazy = 0u64;
    let mut dense = 0u64;
    let mut kv_per_token = 0u64;
    let mut index_per_token = 0u64;
    let mut recurrent_layers = 0u64;
    let mut ram_kv_per_token = 0u64;
    let mut ram_layer_rows = std::collections::BTreeMap::<u64, u64>::new();
    let main_layers = if arch == "qwen35" {
        let total = source.meta.get("qwen35.block_count").and_then(|v| v.as_u64());
        let mtp = source.meta.get("qwen35.nextn_predict_layers").and_then(|v| v.as_u64()).unwrap_or(0);
        total.map(|total| total.checked_sub(mtp).context("Qwen3.5 MTP layers exceed block count")).transpose()?
    } else { None };
    ensure!(placement.is_gpu() || !host_state, "RAM Vulkan context placement cannot be applied to a CPU-state backend");
    for (name, t) in &source.tensors {
        let bytes = u64::try_from(t.byte_len).context("tensor payload exceeds address range")?;
        let kind = layer_tensor(name);
        if matches!(
            kind,
            Some("ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight")
        ) {
            experts = experts
                .checked_add(bytes)
                .context("expert payload overflow")?;
        } else if arch == "qwen4exp" && name == "per_layer_token_embd.weight" {
            // The 26.8 GiB Qwen3.8 hash table is accessed row-by-row. Its
            // projection/norm/convolution tensors are ordinary dense weights.
            lazy = lazy
                .checked_add(bytes)
                .context("lazy embedding payload overflow")?;
        } else {
            dense = dense.checked_add(bytes).context("dense payload overflow")?;
        }
        let layer_number = name.strip_prefix("blk.").and_then(|n| n.split_once('.')).and_then(|(n,_)| n.parse::<u64>().ok());
        // MTP prediction blocks are not executed by native text generation.
        if main_layers.zip(layer_number).is_some_and(|(main, layer)| layer >= main) { continue; }
        if matches!(kind, Some("attn_k.weight" | "attn_v.weight")) {
            ensure!(
                t.dims.len() == 2 && t.dims[1] > 0,
                "{name}: invalid KV projection dimensions"
            );
            if placement == KvPlacement::Ram {
                let bytes = precision.row_bytes(usize::try_from(t.dims[1]).context("KV width exceeds address range")?)? as u64;
                ram_kv_per_token = ram_kv_per_token.checked_add(bytes).context("RAM KV width overflow")?;
                let per_layer = ram_layer_rows.entry(layer_number.context("KV tensor missing layer number")?).or_default();
                *per_layer = per_layer.checked_add(bytes).context("RAM KV layer overflow")?;
            }
            kv_per_token = kv_per_token
                .checked_add(t.dims[1].checked_mul(4).context("KV width overflow")?)
                .context("KV total overflow")?;
        }
        if kind == Some("indexer.k_proj.weight") {
            ensure!(
                t.dims.len() == 2 && t.dims[1] > 0,
                "{name}: invalid indexer projection dimensions"
            );
            // Raw F32 keys plus one pooled F32 key per four tokens.
            index_per_token = index_per_token
                .checked_add(t.dims[1].checked_mul(5).context("indexer width overflow")?)
                .context("indexer total overflow")?;
        }
        if kind == Some("attn_qkv.weight") {
            recurrent_layers = recurrent_layers
                .checked_add(1)
                .context("recurrent layer count overflow")?;
        }
    }
    let get = |suffix: &str| -> Result<u128> {
        let key = format!("{arch}.{suffix}");
        source
            .meta
            .get(&key)
            .and_then(|v| v.as_u64())
            .filter(|n| *n > 0)
            .map(u128::from)
            .with_context(|| {
                format!("{key}: recurrent memory planning requires a positive integer")
            })
    };
    // Two recurrent working matrices plus a prefill rollback checkpoint. KV and
    // indexer allowance includes Vec growth above the initialized prefix.
    let product = |parts: &[u128]| -> Result<u128> {
        parts.iter().try_fold(1u128, |n, x| {
            n.checked_mul(*x).context("host memory geometry overflow")
        })
    };
    let recurrent_per_layer = if host_state && recurrent_layers > 0 {
        let inner = get("ssm.inner_size")?;
        let width = get("ssm.state_size")?;
        let matrix = product(&[inner, width, 4, 3])?;
        let conv_width = product(&[2, get("ssm.group_count")?, width])?
            .checked_add(inner)
            .context("convolution geometry overflow")?;
        matrix
            .checked_add(product(&[conv_width, get("ssm.conv_kernel")?, 4, 3])?)
            .context("recurrent reserve overflow")?
    } else {
        0
    };
    let ram_payload = ram_kv_per_token.checked_mul(context as u64).context("RAM context payload overflow")?;
    let mut capacity = context.min(256);
    let mut old_capacity = 0;
    while capacity < context { old_capacity = capacity; capacity = capacity.saturating_mul(2).min(context); }
    let ram_overlap = ram_layer_rows.values().copied().max().unwrap_or(0)
        .checked_mul(old_capacity as u64).context("RAM context growth overflow")?;
    let ram_peak = ram_payload.checked_add(ram_overlap).context("RAM context peak overflow")?;
    let per_sequence = if host_state {
        let attention = (kv_per_token as u128)
            .checked_add(index_per_token as u128)
            .context("attention reserve overflow")?;
        product(&[attention, context as u128, 2])?
            .checked_add(product(&[recurrent_per_layer, recurrent_layers as u128])?)
            .and_then(|n| n.checked_add(64 * 1024 * 1024))
            .context("sequence memory reserve overflow")?
    } else {
        // Host-visible staging is now charged to its actual RAM heap. Reserve
        // the maximum64MiB expert staging window, transfers, and small CPU scratch.
        (ram_peak as u128).checked_add(96 * 1024 * 1024).context("Vulkan host-state overflow")?
    };
    let state = u64::try_from(product(&[per_sequence, slots as u128])?)
        .context("host sequence reserve overflow")?;
    if let Some(mib) = ram_cap_mib {
        ensure!(placement == KvPlacement::Ram, "a context RAM cap requires RAM context placement");
        let cap = (mib as u64).checked_mul(1024 * 1024).context("context RAM budget overflow")?;
        ensure!(state <= cap, "RAM context and working buffers require {state} bytes; context RAM cap is {cap} bytes");
    }
    let reserve = margin
        .checked_add(dense)
        .and_then(|n| n.checked_add(state))
        .context("host reserve overflow")?;
    ensure!(
        dense.saturating_add(state) <= available,
        "dense-page and sequence-state allowance exceeds available RAM; reduce context or concurrent slots"
    );
    ensure!(placement.is_gpu() || reserve <= available,
        "RAM context requires {reserve} bytes including model pages and OS reserve; only {available} bytes are available");
    let safe = available.saturating_sub(reserve).min(experts);
    let cap = match explicit_mib {
        None => usize::try_from(safe).context("expert cache exceeds address range")?,
        Some(mib) => {
            let requested = mib
                .checked_mul(1024 * 1024)
                .context("expert cache budget overflow")?;
            // This option is a ceiling, not a request to allocate more bytes
            // than exist in the routed expert tensors.
            let bytes = requested.min(usize::try_from(experts).unwrap_or(usize::MAX));
            ensure!(
                u64::try_from(bytes).context("expert cache exceeds address range")?
                    <= available.saturating_sub(reserve)
                    || bytes == 0,
                "explicit expert cache requires {bytes} bytes plus {reserve} bytes of host reserves; only {available} bytes are currently available; lower expert RAM/context/concurrency"
            );
            bytes
        }
    };
    Ok(HostMemoryPlan {
        available_bytes: available,
        os_margin_bytes: margin,
        dense_page_allowance_bytes: dense,
        state_reserve_bytes: state,
        state_slots: slots,
        ram_context_payload_bytes: ram_payload.checked_mul(slots as u64).context("RAM context slot overflow")?,
        ram_context_peak_bytes: ram_peak.checked_mul(slots as u64).context("RAM context slot overflow")?,
        vulkan_host_budget_bytes: if host_state { 0 } else { state },
        routed_expert_payload_bytes: experts,
        lazy_embedding_payload_bytes: lazy,
        expert_cache_capacity_bytes: cap,
        policy: if explicit_mib.is_some() {
            "explicit_bounded_owned_expert_cache"
        } else {
            "automatic_owned_expert_cache_after_dense_state_and_os_reserves"
        },
    })
}

fn layer_tensor(name: &str) -> Option<&str> {
    let (layer, kind) = name.strip_prefix("blk.")?.split_once('.')?;
    (!layer.is_empty() && layer.bytes().all(|b| b.is_ascii_digit())).then_some(kind)
}

pub fn available_memory() -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo")?;
        let kib = text
            .lines()
            .find_map(|l| l.strip_prefix("MemAvailable:"))
            .and_then(|x| x.split_whitespace().next())
            .context("MemAvailable missing")?
            .parse::<u64>()?;
        let mut available = kib.checked_mul(1024).context("available memory overflow")?;
        if let Ok(cgroups) = std::fs::read_to_string("/proc/self/cgroup") {
            if let Some(path) = cgroups.lines().find_map(|l| l.strip_prefix("0::")) {
                let relative = path.trim_start_matches('/');
                let mut location = std::path::PathBuf::from("/sys/fs/cgroup").join(relative);
                loop {
                    if let (Ok(max), Ok(used)) = (
                        std::fs::read_to_string(location.join("memory.max")),
                        std::fs::read_to_string(location.join("memory.current")),
                    ) {
                        if let (Ok(max), Ok(used)) =
                            (max.trim().parse::<u64>(), used.trim().parse::<u64>())
                        {
                            available = available.min(max.saturating_sub(used));
                        }
                    }
                    if location == std::path::Path::new("/sys/fs/cgroup") || !location.pop() {
                        break;
                    }
                }
            }
        }
        return Ok(available);
    }
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct Status {
            length: u32,
            load: u32,
            total: u64,
            available: u64,
            total_page: u64,
            available_page: u64,
            total_virtual: u64,
            available_virtual: u64,
            extended: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(status: *mut Status) -> i32;
        }
        let mut status = Status {
            length: std::mem::size_of::<Status>() as u32,
            load: 0,
            total: 0,
            available: 0,
            total_page: 0,
            available_page: 0,
            total_virtual: 0,
            available_virtual: 0,
            extended: 0,
        };
        // SAFETY: the Windows API receives a correctly sized writable C layout.
        ensure!(
            unsafe { GlobalMemoryStatusEx(&mut status) } != 0,
            "GlobalMemoryStatusEx failed"
        );
        return Ok(status.available);
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        anyhow::bail!("automatic host memory planning supports Linux/WSL and Windows")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gguf::{TensorInfo, Value},
        quant::QuantType,
    };
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    // Keep a real tiny mapped owner, then replace only planning descriptors.
    // No large payload is mapped, read or allocated by these arithmetic tests.
    fn source(arch: &str) -> Gguf {
        let mut g = Gguf::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-qwen3moe.gguf"),
        )
        .unwrap();
        g.meta.clear();
        g.tensors.clear();
        g.meta
            .insert("general.architecture".into(), Value::String(arch.into()));
        g
    }
    fn tensor(g: &mut Gguf, name: &str, dims: &[u64], ty: QuantType) -> u64 {
        let bytes =
            ty.row_bytes(dims[0] as usize).unwrap() * dims[1..].iter().product::<u64>() as usize;
        g.tensors.insert(
            name.into(),
            TensorInfo {
                name: name.into(),
                dims: dims.to_vec(),
                ty,
                offset: 0,
                byte_len: bytes,
                file_offset: 0,
                shard_index: 0,
            },
        );
        bytes as u64
    }
    fn budget_source() -> Gguf {
        let mut g = source("qwen3moe");
        tensor(&mut g, "output.weight", &[1024, 256], QuantType::F32);
        for kind in ["gate", "up", "down"] {
            tensor(
                &mut g,
                &format!("blk.0.ffn_{kind}_exps.weight"),
                &[256, 256, 64],
                QuantType::F32,
            );
        }
        g
    }
    fn recurrent_metadata(g: &mut Gguf) {
        for (suffix, n) in [
            ("inner_size", 6144),
            ("state_size", 128),
            ("group_count", 16),
            ("conv_kernel", 4),
        ] {
            g.meta
                .insert(format!("qwen4exp.ssm.{suffix}"), Value::U64(n));
        }
    }

    #[test]
    fn actual_qwen38_ple_table_is_lazy_but_projections_are_dense() {
        let mut g = source("qwen4exp");
        // Names, shape and quantization from the pinned GSQ-RCO IQ3_S headers.
        let ple = tensor(
            &mut g,
            "per_layer_token_embd.weight",
            &[160, 320001536],
            QuantType::IQ4_NL,
        );
        assert_eq!(ple, 28_800_138_240);
        let projection = tensor(
            &mut g,
            "blk.1.ple_key.weight",
            &[2560, 10240],
            QuantType::BF16,
        );
        let experts = tensor(
            &mut g,
            "blk.0.ffn_gate_exps.weight",
            &[2560, 640, 512],
            QuantType::IQ3_XXS,
        );
        let lookalike = tensor(
            &mut g,
            "blk.1.ple_embd_projection.weight",
            &[160, 32],
            QuantType::F32,
        );
        let unknown = tensor(&mut g, "unrelated_exps.weight", &[32, 32], QuantType::F32);
        let plan = plan_available(&g, 32768, 1, true, None, 4096, 62 * GIB).unwrap();
        assert_eq!(plan.lazy_embedding_payload_bytes, ple);
        assert_eq!(
            plan.dense_page_allowance_bytes,
            projection + lookalike + unknown
        );
        assert_eq!(plan.routed_expert_payload_bytes, experts);
        assert_eq!(plan.expert_cache_capacity_bytes as u64, experts);
        // An unsupported name must never silently escape the dense allowance.
        let t = g.tensors.remove("per_layer_token_embd.weight").unwrap();
        g.tensors
            .insert("per_layer_token_embd_projection.weight".into(), t);
        let plan = plan_available(&g, 32768, 1, true, None, 4096, 62 * GIB).unwrap();
        assert_eq!(plan.lazy_embedding_payload_bytes, 0);
        assert_eq!(
            plan.dense_page_allowance_bytes,
            ple + projection + lookalike + unknown
        );
    }

    #[test]
    fn qwen38_attention_indexer_and_recurrent_reserves_scale_with_slots() {
        let mut g = source("qwen4exp");
        recurrent_metadata(&mut g);
        for layer in 0..48 {
            if layer % 4 == 3 {
                for kind in ["k", "v"] {
                    tensor(
                        &mut g,
                        &format!("blk.{layer}.attn_{kind}.weight"),
                        &[2560, 512],
                        QuantType::Q6_K,
                    );
                }
                tensor(
                    &mut g,
                    &format!("blk.{layer}.indexer.k_proj.weight"),
                    &[2560, 128],
                    QuantType::BF16,
                );
            } else {
                tensor(
                    &mut g,
                    &format!("blk.{layer}.attn_qkv.weight"),
                    &[2560, 10240],
                    QuantType::Q6_K,
                );
            }
        }
        let one = plan_available(&g, 32768, 1, true, None, 4096, 62 * GIB).unwrap();
        let four = plan_available(&g, 32768, 4, true, None, 4096, 62 * GIB).unwrap();
        assert_eq!(one.state_reserve_bytes, 4_149_084_160);
        assert_eq!(four.state_reserve_bytes, 16_596_336_640);
        assert_eq!(four.state_slots, 4);
        assert_eq!(
            four.dense_page_allowance_bytes,
            one.dense_page_allowance_bytes
        );
        let short = plan_available(&g, 4096, 1, true, None, 4096, 62 * GIB).unwrap();
        assert_eq!(short.state_reserve_bytes, 890_109_952);
        // Full-context host caches for four slots do not fit this machine.
        assert!(plan_available(&g, 262144, 4, true, None, 4096, 62 * GIB).is_err());
        // GPU-owned KV/recurrent state is not charged again; independent RAM
        // staging/working allocations retain a bounded96MiB allowance per slot.
        let device = plan_available(&g, 262144, 4, false, None, 4096, 62 * GIB).unwrap();
        assert_eq!(device.state_reserve_bytes, 96 * MIB * 4);
        assert_eq!(device.vulkan_host_budget_bytes, device.state_reserve_bytes);
        assert_eq!(device.ram_context_payload_bytes, 0);
    }

    #[test]
    fn automatic_and_explicit_cache_respect_the_same_headroom() {
        let g = budget_source();
        let auto = plan_available(&g, 128, 1, true, None, 8, 85 * MIB).unwrap();
        assert_eq!(auto.dense_page_allowance_bytes, MIB);
        assert_eq!(auto.state_reserve_bytes, 64 * MIB);
        assert_eq!(auto.routed_expert_payload_bytes, 48 * MIB);
        assert_eq!(auto.expert_cache_capacity_bytes as u64, 12 * MIB);
        assert!(auto.policy.starts_with("automatic"));
        for requested in [0usize, 4, 12] {
            let explicit = plan_available(&g, 128, 1, true, Some(requested), 8, 85 * MIB).unwrap();
            assert_eq!(
                explicit.expert_cache_capacity_bytes,
                requested * 1024 * 1024
            );
            assert!(explicit.policy.starts_with("explicit"));
        }
        assert!(plan_available(&g, 128, 1, true, Some(13), 8, 85 * MIB).is_err());
        // The same available RAM must reserve every configured active slot.
        let four = plan_available(&g, 128, 4, true, None, 8, 300 * MIB).unwrap();
        assert_eq!(four.state_reserve_bytes, 256 * MIB);
        assert_eq!(four.expert_cache_capacity_bytes as u64, 35 * MIB);
    }

    #[test]
    fn explicit_ceiling_is_clamped_before_checking_available_ram() {
        let g = budget_source();
        let plan = plan_available(&g, 128, 1, true, Some(4096), 8, 128 * MIB).unwrap();
        assert_eq!(plan.expert_cache_capacity_bytes as u64, 48 * MIB);
        assert!(plan_available(&g, 128, 1, true, Some(4096), 8, 100 * MIB).is_err());
    }

    #[test]
    fn no_cache_headroom_disables_cache_but_does_not_waive_state_reserves() {
        let g = budget_source();
        for explicit in [None, Some(0)] {
            let plan = plan_available(&g, 128, 1, true, explicit, 8, 65 * MIB).unwrap();
            assert_eq!(plan.expert_cache_capacity_bytes, 0);
            assert!(plan_available(&g, 128, 1, true, explicit, 0, 65 * MIB - 1).is_err());
        }
        assert!(plan_available(&g, 128, 1, true, Some(1), 8, 65 * MIB).is_err());
    }

    #[test]
    fn metadata_and_budget_overflow_return_errors_without_panicking() {
        let mut g = budget_source();
        assert!(plan_available(&g, 128, 1, true, None, usize::MAX, u64::MAX).is_err());
        assert!(plan_available(&g, 128, 1, true, Some(usize::MAX), 0, u64::MAX).is_err());
        tensor(&mut g, "blk.0.attn_k.weight", &[1, 1], QuantType::F32);
        g.tensors.get_mut("blk.0.attn_k.weight").unwrap().dims[1] = u64::MAX;
        assert!(plan_available(&g, 128, 1, true, None, 0, u64::MAX).is_err());
        let mut g = source("qwen4exp");
        recurrent_metadata(&mut g);
        tensor(&mut g, "blk.0.attn_qkv.weight", &[1, 1], QuantType::F32);
        for key in ["inner_size", "state_size"] {
            g.meta
                .insert(format!("qwen4exp.ssm.{key}"), Value::U64(u64::MAX));
        }
        assert!(plan_available(&g, 128, 1, true, None, 0, u64::MAX).is_err());
        // This product fits u128 but cannot be represented by the u64 plan.
        g.meta
            .insert("qwen4exp.ssm.inner_size".into(), Value::U64(1 << 35));
        g.meta
            .insert("qwen4exp.ssm.state_size".into(), Value::U64(1 << 28));
        assert!(plan_available(&g, 128, 1, true, None, 0, u64::MAX).is_err());
        let mut g = source("qwen3moe");
        tensor(&mut g, "a.weight", &[1], QuantType::F32);
        tensor(&mut g, "b.weight", &[1], QuantType::F32);
        g.tensors.get_mut("a.weight").unwrap().byte_len = usize::MAX;
        assert!(plan_available(&g, 128, 1, true, None, 0, u64::MAX).is_err());
    }

    #[test]
    fn recurrent_metadata_cannot_silently_become_zero_and_shapes_are_checked() {
        let mut g = source("qwen4exp");
        recurrent_metadata(&mut g);
        tensor(&mut g, "blk.0.attn_qkv.weight", &[1, 1], QuantType::F32);
        for bad in [
            None,
            Some(Value::String("6144".into())),
            Some(Value::I64(-1)),
            Some(Value::U64(0)),
        ] {
            g.meta.remove("qwen4exp.ssm.inner_size");
            if let Some(value) = bad {
                g.meta.insert("qwen4exp.ssm.inner_size".into(), value);
            }
            let err = plan_available(&g, 128, 1, true, None, 0, GIB).unwrap_err();
            assert!(err.to_string().contains("ssm.inner_size"));
        }
        let mut g = source("qwen3moe");
        tensor(&mut g, "blk.0.attn_k.weight", &[1], QuantType::F32);
        assert!(plan_available(&g, 128, 1, true, None, 0, GIB).is_err());
        for (context, slots) in [(0, 1), (1_048_577, 1), (128, 0), (128, 5)] {
            assert!(plan_available(&budget_source(), context, slots, true, None, 0, GIB).is_err());
        }
    }
    #[test]
    fn ram_context_reserves_full_quantized_history_before_expert_cache() {
        let mut g = budget_source();
        for layer in 0..48 {
            for kind in ["k", "v"] {
                tensor(&mut g, &format!("blk.{layer}.attn_{kind}.weight"), &[2048, 512], QuantType::Q4_K);
            }
        }
        let context = 524288;
        let preliminary = plan_available_kv(&g, context, 1, false, None, 1024, 64*GIB,
            KvPrecision::Q4_64, KvPlacement::Ram, None).unwrap();
        assert_eq!(preliminary.ram_context_payload_bytes, 13_690_208_256);
        assert_eq!(preliminary.ram_context_peak_bytes, 13_832_814_592);
        assert_eq!(preliminary.vulkan_host_budget_bytes, preliminary.ram_context_peak_bytes + 96*MIB);
        assert_eq!(preliminary.state_reserve_bytes, preliminary.vulkan_host_budget_bytes);
        let available = preliminary.os_margin_bytes + preliminary.dense_page_allowance_bytes
            + preliminary.state_reserve_bytes + preliminary.routed_expert_payload_bytes/2;
        let planned = plan_available_kv(&g, context, 1, false, None, 1024, available,
            KvPrecision::Q4_64, KvPlacement::Ram, None).unwrap();
        assert_eq!(planned.expert_cache_capacity_bytes as u64, planned.routed_expert_payload_bytes/2);
        assert!(plan_available_kv(&g, context, 1, false, None, 1024, 64*GIB,
            KvPrecision::Q4_64, KvPlacement::Ram, Some(1024)).unwrap_err().to_string().contains("RAM cap"));
        assert!(plan_available_kv(&g, context, 1, false, None, 1024, 64*GIB,
            KvPrecision::F32, KvPlacement::Ram, None).is_err());
    }

    #[test]
    fn qwen35_mtp_blocks_do_not_inflate_cpu_or_hybrid_context_state() {
        let mut g = source("qwen35");
        g.meta.insert("qwen35.block_count".into(), Value::U32(3));
        g.meta.insert("qwen35.nextn_predict_layers".into(), Value::U32(1));
        for (key, value) in [("inner_size", 64), ("state_size", 16), ("group_count", 2), ("conv_kernel", 4)] {
            g.meta.insert(format!("qwen35.ssm.{key}"), Value::U32(value));
        }
        tensor(&mut g, "blk.0.attn_k.weight", &[32,64], QuantType::F32);
        tensor(&mut g, "blk.0.attn_v.weight", &[32,64], QuantType::F32);
        tensor(&mut g, "blk.1.attn_qkv.weight", &[32,128], QuantType::F32);
        let reference = plan_available(&g, 32768, 2, true, None, 0, 8*GIB).unwrap();
        tensor(&mut g, "blk.2.attn_k.weight", &[32,65536], QuantType::F32);
        tensor(&mut g, "blk.2.attn_v.weight", &[32,65536], QuantType::F32);
        tensor(&mut g, "blk.2.attn_qkv.weight", &[32,128], QuantType::F32);
        let with_mtp = plan_available(&g, 32768, 2, true, None, 0, 8*GIB).unwrap();
        assert_eq!(reference.state_reserve_bytes, with_mtp.state_reserve_bytes);
        assert!(with_mtp.dense_page_allowance_bytes > reference.dense_page_allowance_bytes);
    }

}
