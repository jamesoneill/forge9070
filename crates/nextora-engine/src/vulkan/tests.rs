use super::*;
use crate::{
    gguf::Gguf,
    model::{Model, Step},
    quant::QuantType,
};
use half::{bf16, f16};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
static NEXT: AtomicU64 = AtomicU64::new(1);

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}[{i}] nonfinite {a}/{b}"
        );
        assert!(
            (a - b).abs() <= 0.002 + 0.0003 * b.abs(),
            "{label}[{i}] {a} != {b}"
        );
    }
}
fn fixture(moe: bool) -> Arc<Gguf> {
    let name = if moe {
        "tiny-qwen3moe.gguf"
    } else {
        "tiny-qwen3.gguf"
    };
    Arc::new(
        Gguf::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures")
                .join(name),
        )
        .unwrap(),
    )
}
fn options(bytes: usize) -> DeviceOptions {
    DeviceOptions {
        device_index: None,
        allow_software: true,
        fast_math: false,
        direct_experts: None,
        kv_precision: crate::vulkan::KvPrecision::F32,
        kv_placement: crate::vulkan::KvPlacement::Gpu,
        host_memory_budget_bytes: 0,
        memory_budget_bytes: bytes,
    }
}
struct FileGuard(PathBuf);
impl Drop for FileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn matrix_file(ty: QuantType, width: usize, rows: usize, salt: usize) -> FileGuard {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());
    data.extend_from_slice(b"matrix");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&(width as u64).to_le_bytes());
    data.extend_from_slice(&(rows as u64).to_le_bytes());
    data.extend_from_slice(&ty.id().to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.resize(data.len().next_multiple_of(32), 0);
    for row in 0..rows {
        if matches!(ty, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
            for col in 0..width {
                let value = (((col * 31 + row * 19 + salt * 7) % 251) as f32 - 125.0) / 128.0;
                match ty {
                    QuantType::F32 => data.extend_from_slice(&value.to_le_bytes()),
                    QuantType::F16 => data.extend_from_slice(&f16::from_f32(value).to_le_bytes()),
                    _ => data.extend_from_slice(&bf16::from_f32(value).to_le_bytes()),
                }
            }
        } else {
            for block in 0..width / ty.block_elements() {
                let mut bytes: Vec<u8> = (0..ty.block_bytes())
                    .map(|i| (i * 73 + row * 53 + block * 29 + salt * 11) as u8)
                    .collect();
                let scale = f16::from_f32(if (block + row) % 2 == 0 {
                    0.03125
                } else {
                    -0.0625
                })
                .to_le_bytes();
                if matches!(ty, QuantType::Q3_K | QuantType::Q6_K) {
                    let n = bytes.len();
                    bytes[n - 2..].copy_from_slice(&scale);
                } else {
                    bytes[..2].copy_from_slice(&scale);
                }
                if matches!(ty, QuantType::Q4_K | QuantType::Q5_K) {
                    bytes[2..4].copy_from_slice(&f16::from_f32(0.015625).to_le_bytes());
                }
                data.extend_from_slice(&bytes);
            }
        }
    }
    let path = std::env::temp_dir().join(format!(
        "forge-gpu-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, data).unwrap();
    FileGuard(path)
}

#[test]
fn packed_gpu_type_inventory_matches_public_types() {
    for id in [0, 1, 2, 8, 11, 12, 13, 14, 18, 20, 21, 22, 23, 30, 42] {
        assert!(supports_quant(id));
        assert!(QuantType::try_from(id).is_ok());
    }
    for id in [3, 99, u32::MAX] {
        assert!(!supports_quant(id));
    }
}

/// Run explicitly with NEXTORA_VULKAN_TEST=1 and a hardware or software ICD.
/// Software execution validates shader math/resources, never RX9070XT speed.
#[test]
#[ignore = "requires explicit Vulkan shader execution environment"]
fn software_vulkan_full_validation() {
    assert_eq!(
        std::env::var("NEXTORA_VULKAN_TEST").as_deref(),
        Ok("1"),
        "explicit NEXTORA_VULKAN_TEST=1 is required"
    );
    let report = discover().unwrap();
    if report["adapters"]
        .as_array()
        .unwrap()
        .iter()
        .all(|a| a["software"] == true)
    {
        assert!(
            super::context::Device::new(&DeviceOptions::default()).is_err(),
            "production defaults must reject software"
        );
    }
    // Every supported packed decoder, including odd half rows and two block
    // counts, is exercised by an actual GPU submission against CPU values.
    let executor = GpuMatvecExecutor::new(options(64 * 1024 * 1024)).unwrap();
    let formats = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q8_0,
        QuantType::Q2_0,
        QuantType::Q3_K,
        QuantType::Q4_K,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::IQ2_S,
        QuantType::IQ3_XXS,
        QuantType::IQ3_S,
        QuantType::IQ4_NL,
        QuantType::IQ4_XS,
    ];
    for (salt, ty) in formats.into_iter().enumerate() {
        for width in [256usize, 512] {
            let width = if ty.block_elements() == 1 {
                width + 3
            } else {
                width
            };
            let file = matrix_file(ty, width, 5, salt);
            let source = Gguf::open(&file.0).unwrap();
            let matrix = source.tensor("matrix").unwrap();
            let input: Vec<f32> = (0..width)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 53.0)
                .collect();
            let reference = matrix.matvec(&input).unwrap();
            let mut actual = vec![0.0; 5];
            executor.matvec_gpu(&matrix, &input, &mut actual).unwrap();
            close(&actual, &reference, &format!("{ty:?}/{width}"));
            executor.matvec_gpu(&matrix, &input, &mut actual).unwrap();
            close(&actual, &reference, "cached packed projection");
        }
    }
    // An uncached expert is submitted before independent host work. The
    // callback may inspect allocated handles, but the caller must wait for
    // the transfer fence before using those handles in GPU compute.
    let overlap_file = matrix_file(QuantType::F32, 256, 5, 193);
    let overlap_source = Gguf::open(&overlap_file.0).unwrap();
    let overlap_tensor = overlap_source.tensor("matrix").unwrap();
    let mut observed_inflight = false;
    let uploaded = executor
        .prepare_batch_with_host_work(&[overlap_tensor], true, |ready, in_flight| {
            assert!(in_flight, "uncached matrix should require GPU transfer");
            assert!(ready[0].is_some());
            observed_inflight = true;
            Ok(())
        })
        .unwrap();
    assert!(observed_inflight && uploaded[0].is_some());
    drop(uploaded);
    let mut observed_hit = false;
    let cached = executor
        .prepare_batch_with_host_work(&[overlap_tensor], true, |ready, in_flight| {
            assert!(!in_flight, "cached matrix needs no transfer");
            assert!(ready[0].is_some());
            observed_hit = true;
            Ok(())
        })
        .unwrap();
    assert!(observed_hit && cached[0].is_some());
    drop(cached);

    // Even an error in CPU-only host work must first join the GPU fence.
    let failed_file = matrix_file(QuantType::F32, 256, 5, 194);
    let failed_source = Gguf::open(&failed_file.0).unwrap();
    let failed_tensor = failed_source.tensor("matrix").unwrap();
    assert!(executor
        .prepare_batch_with_host_work(&[failed_tensor], false, |_, in_flight| {
            assert!(in_flight);
            anyhow::bail!("synthetic CPU fallback failure");
        })
        .is_err());
    assert!(executor
        .prepare_batch_with_host_work(&[failed_tensor], false, |ready, in_flight| {
            assert!(in_flight);
            assert!(ready[0].is_some());
            Ok(())
        })
        .unwrap()[0]
        .is_some());

    let stats = executor.stats();
    assert_eq!(stats["gpu_projection_calls"].as_u64(), Some(60));
    assert!(stats["cache_hits"].as_u64().unwrap() >= 30);
    assert!(
        stats["uploaded_bytes"].as_u64().unwrap() > 0
            && stats["readback_bytes"].as_u64().unwrap() > 0
    );
    drop(executor);
    for moe in [false, true] {
        let source = fixture(moe);
        let cpu = Model::load(Arc::clone(&source), 16).unwrap();
        let gpu = GpuModel::load(source, 16, options(64 * 1024 * 1024)).unwrap();
        let mut cs = cpu.new_sequence();
        let mut gs = gpu.new_sequence().unwrap();
        for token in [2, 19, 77, 5] {
            let c = cpu.step(&mut cs, token).unwrap();
            let g = gpu.step(&mut gs, token).unwrap();
            close(&g.logits, &c.logits, "resident logits");
            close(&g.hidden, &c.hidden, "resident hidden");
            for layer in 0..cpu.config().block_count {
                let (k, v) = gs.read_kv(layer).unwrap();
                let (ck, cv) = cs.kv(layer).unwrap();
                close(&k, ck, "resident keys");
                close(&v, cv, "resident values");
            }
        }
        let before = gs.position();
        assert!(gpu.step(&mut gs, u32::MAX).is_err());
        assert_eq!(gs.position(), before);
        gs.reset().unwrap();
        cs.reset();
        let c = cpu.prefill(&mut cs, &[3, 11, 71]).unwrap();
        let g = gpu.prefill(&mut gs, &[3, 11, 71]).unwrap();
        close(&g.logits, &c.logits, "batched prefill logits");
        close(&g.hidden, &c.hidden, "batched prefill hidden");
        let mut other = gpu.new_sequence().unwrap();
        gpu.prefill(&mut other, &[99, 101]).unwrap();
        let c = cpu.step(&mut cs, 8).unwrap();
        let g = gpu.step(&mut gs, 8).unwrap();
        close(&g.logits, &c.logits, "interleaved independent logits");
        close(&g.hidden, &c.hidden, "interleaved independent hidden");
        assert_eq!(other.position(), 2);
        assert_eq!(gs.position(), 4);
        let report = gpu.memory_stats();
        assert!(
            report["live_allocation_bytes"].as_u64().unwrap()
                <= report["allocation_budget_bytes"].as_u64().unwrap()
        );
        let reserved = report["live_allocation_bytes"].as_u64().unwrap();
        drop(other);
        assert!(
            gpu.memory_stats()["live_allocation_bytes"]
                .as_u64()
                .unwrap()
                < reserved
        );
        let mut out = Step::default();
        assert!(gpu.prefill_into(&mut gs, &[1; 16], &mut out).is_err());
        assert_eq!(gs.position(), 4);
    }
    // A deliberately small cache must evict by ownership rather than grow past
    // the limit. The same tensor keeps a stable key after another is evicted.
    let bounded = GpuMatvecExecutor::new(options(2 * 1024 * 1024)).unwrap();
    let files: Vec<_> = (0..4)
        .map(|salt| matrix_file(QuantType::F32, 512, 256, salt))
        .collect();
    let sources: Vec<_> = files.iter().map(|f| Gguf::open(&f.0).unwrap()).collect();
    let input = vec![0.125; 512];
    let mut output = vec![0.0; 256];
    for source in &sources {
        let tensor = source.tensor("matrix").unwrap();
        bounded.matvec_gpu(&tensor, &input, &mut output).unwrap();
        close(
            &output,
            &tensor.matvec(&input).unwrap(),
            "eviction projection",
        );
    }
    let tensor = sources[0].tensor("matrix").unwrap();
    bounded.matvec_gpu(&tensor, &input, &mut output).unwrap();
    close(
        &output,
        &tensor.matvec(&input).unwrap(),
        "reloaded projection",
    );
    let report = bounded.stats();
    assert!(report["cache_evictions"].as_u64().unwrap() > 0);
    assert!(report["live_allocation_bytes"].as_u64().unwrap() <= 2 * 1024 * 1024);
}


#[test]
#[ignore = "requires explicit Vulkan shader execution environment"]
fn software_vulkan_batched_routed_experts() {
    use crate::model::{CpuMatvec, MatvecExecutor, MoeScratch, RoutedExpert};
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    for ty in [QuantType::F32, QuantType::F16, QuantType::Q4_K, QuantType::Q6_K] {
        let files: Vec<_> = (0..9).map(|i| matrix_file(ty, 256, 256, i+37)).collect();
        let sources: Vec<_> = files.iter().map(|file| Gguf::open(&file.0).unwrap()).collect();
        let experts: Vec<_> = (0..3).map(|i| RoutedExpert {
            gate: sources[i*3].tensor("matrix").unwrap(),
            up: sources[i*3+1].tensor("matrix").unwrap(),
            down: sources[i*3+2].tensor("matrix").unwrap(),
            weight: [0.6,0.3,0.1][i],
        }).collect();
        let input: Vec<_> = (0..256).map(|i| ((i*7%31) as f32-15.0)/31.0).collect();
        let mut gate = vec![0.0;256]; let mut up = vec![0.0;256]; let mut partial = vec![0.0;256];
        let mut expected = vec![0.0;256];
        CpuMatvec.routed_moe(&experts, &input, &mut MoeScratch {gate:&mut gate,up:&mut up,expert_output:&mut partial,output:&mut expected},true).unwrap();
        let executor = GpuMatvecExecutor::new(options(64*1024*1024)).unwrap();
        let mut actual = vec![0.0;256];
        for _ in 0..2 {
            executor.routed_moe(&experts, &input, &mut MoeScratch {gate:&mut gate,up:&mut up,expert_output:&mut partial,output:&mut actual},true).unwrap();
            close(&actual,&expected,"batched routed experts");
        }
        let stats=executor.stats();
        assert_eq!(stats["batched_moe_calls"],2);
        assert_eq!(stats["gpu_projection_calls"],18);
        assert!(stats["queue_submissions"].as_u64().unwrap()<=3, "one upload then one submission per layer: {stats}");
        // Protected GPU expert leases are dropped after the fence, so a small
        // later cache budget can evict these entries rather than leaking them.
        assert_eq!(stats["cpu_moe_budget_fallback_calls"],0);
        let tiny = GpuMatvecExecutor::new(options(1024*1024)).unwrap();
        // Three F32 matrices 512x512 each exceed this entire 1 MiB budget.
        let bigfiles: Vec<_> = (0..3).map(|i| matrix_file(QuantType::F32,512,512,i+88)).collect();
        let big: Vec<_> = bigfiles.iter().map(|f|Gguf::open(&f.0).unwrap()).collect();
        let e=[RoutedExpert {gate:big[0].tensor("matrix").unwrap(),up:big[1].tensor("matrix").unwrap(),down:big[2].tensor("matrix").unwrap(),weight:1.0}];
        let input=vec![0.01;512]; let mut gate=vec![0.0;512];let mut up=vec![0.0;512];let mut part=vec![0.0;512];let mut output=vec![0.0;512];let mut truth=vec![0.0;512];
        tiny.routed_moe(&e,&input,&mut MoeScratch{gate:&mut gate,up:&mut up,expert_output:&mut part,output:&mut output},true).unwrap();
        CpuMatvec.routed_moe(&e,&input,&mut MoeScratch{gate:&mut gate,up:&mut up,expert_output:&mut part,output:&mut truth},true).unwrap();
        close(&output,&truth,"CPU mixture fallback");
        assert_eq!(tiny.stats()["cpu_moe_budget_fallback_calls"],1);
    }
}


#[test]
#[ignore = "requires explicit Vulkan shader execution environment"]
fn software_vulkan_prefill_free_cache() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let files: Vec<_> = (0..3).map(|i| matrix_file(QuantType::F32, 256, 256, i+131)).collect();
    let sources: Vec<_> = files.iter().map(|file|Gguf::open(&file.0).unwrap()).collect();
    let tensors: Vec<_> = sources.iter().map(|source|source.tensor("matrix").unwrap()).collect();
    let executor = GpuMatvecExecutor::new(options(64*1024*1024)).unwrap();
    drop(executor.prepare_batch(&tensors[..1],true).unwrap());
    let decode_recency = executor.cache_recencies()[&tensors[0].cache_key()];
    let initial_submissions = executor.stats()["queue_submissions"].as_u64().unwrap();
    drop(executor.prepare_batch_prefill_free(&tensors[1..2]).unwrap());
    let recencies = executor.cache_recencies();
    assert_eq!(recencies[&tensors[0].cache_key()],decode_recency);
    assert_eq!(recencies[&tensors[1].cache_key()],0);
    assert_eq!(recencies.len(),2);
    let uploaded_submissions = executor.stats()["queue_submissions"].as_u64().unwrap();
    assert_eq!(uploaded_submissions,initial_submissions+1);
    let hits = executor.prepare_batch_prefill_free(&tensors[..2]).unwrap();
    assert!(hits.iter().all(Option::is_some));
    drop(hits);
    assert_eq!(executor.cache_recencies(),recencies);
    assert_eq!(executor.stats()["queue_submissions"].as_u64().unwrap(),uploaded_submissions);

    // Allocation pressure switches further misses to transient uploads while
    // preserving all existing decode/prefill residents and their recencies.
    let device = executor.device();
    let filler = device.storage((device.budget*91/100-device.reserved()) as usize).unwrap();
    let near_full = executor.prepare_batch_prefill_free(&tensors[2..3]).unwrap();
    assert!(near_full[0].is_some());
    drop(near_full);
    assert_eq!(executor.cache_recencies(),recencies);
    assert_eq!(executor.stats()["cache_evictions"],0);
    assert_eq!(executor.stats()["transient_prefill_tensor_uploads"],1);
    drop(filler);

    // The pre-existing strict prefill API still avoids admitting a miss.
    drop(executor.prepare_batch(&tensors[2..3],false).unwrap());
    assert_eq!(executor.cache_recencies(),recencies);
    assert_eq!(executor.stats()["cache_evictions"],0);
}

#[test]
#[ignore = "requires explicit Vulkan execution"]
fn software_vulkan_duplicate_upload_keeps_decode_lease_and_recency() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let file = matrix_file(QuantType::Q4_K, 256, 5, 501);
    let source = Gguf::open(&file.0).unwrap();
    let weight = source.tensor("matrix").unwrap();
    let executor = GpuMatvecExecutor::new(options(32 * 1024 * 1024)).unwrap();
    let device = executor.device();
    let staging = device.host(weight.bytes().len().next_multiple_of(4)).unwrap();
    let mut commands = super::context::Commands::new(Arc::clone(&device), 1).unwrap();
    commands.begin().unwrap();
    let prefill = executor
        .prepare_batch_recorded(&[weight], false, &commands, &staging)
        .unwrap();

    // A decode request finishes its independent upload before the queued
    // prefill publishes its copy. Its current buffer and recency must survive.
    let decode = executor.prepare_batch(&[weight], true).unwrap();
    let first = decode[0].as_ref().unwrap();
    assert!(!Arc::ptr_eq(first, prefill.weights[0].as_ref().unwrap()));
    let recency = executor.cache_recencies();
    assert!(recency[&weight.cache_key()] > 0);
    commands.submit().unwrap();
    executor.commit_prepared(prefill).unwrap();
    let resident = executor.cached_weight(weight.cache_key(), false).unwrap().unwrap();
    assert!(Arc::ptr_eq(first, &resident));
    assert_eq!(executor.cache_recencies(), recency);
}

#[test]
#[ignore = "requires explicit direct Vulkan integer-dot execution"]
fn software_vulkan_direct_prefill_reclaims_idle_weights_under_pressure() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let files: Vec<_> = (0..4)
        .map(|i| matrix_file(QuantType::Q4_K, 2048, 256, i + 311))
        .collect();
    let sources: Vec<_> = files.iter().map(|f| Gguf::open(&f.0).unwrap()).collect();
    let tensors: Vec<_> = sources.iter().map(|s| s.tensor("matrix").unwrap()).collect();
    let size = tensors[3].bytes().len();
    // Cover both the recorded batch and a miss larger than its staging buffer.
    for staging_size in [size, 4096] {
        let executor = GpuMatvecExecutor::new(DeviceOptions {
            direct_experts: Some(true), fast_math: true, ..options(32 * 1024 * 1024)
        }).unwrap();
        let device = executor.device();
        assert!(device.direct_experts);
        let staging = device.host(staging_size).unwrap();
        let mut commands = super::context::Commands::new(Arc::clone(&device), 1).unwrap();
        let weights = executor.prepare_batch(&tensors[..3], true).unwrap();
        assert!(weights.iter().all(Option::is_some));
        let held = Arc::clone(weights[0].as_ref().unwrap());
        let filler = device.storage((device.budget - device.reserved() - 64 * 1024) as usize).unwrap();

        commands.begin().unwrap();
        let blocked = executor.prepare_batch_recorded(&tensors[3..], false, &commands, &staging).unwrap();
        assert!(blocked.weights[0].is_none(), "all cache entries are still leased");
        commands.submit().unwrap();
        executor.commit_prepared(blocked).unwrap();
        assert_eq!(executor.stats()["cache_evictions"], 0);

        drop(weights);
        commands.begin().unwrap();
        let batch = executor.prepare_batch_recorded(&tensors[3..], false, &commands, &staging).unwrap();
        assert!(batch.weights[0].is_some(), "an idle entry can supply the missing expert on GPU");
        commands.submit().unwrap();
        executor.commit_prepared(batch).unwrap();
        let cached = executor.cache_recencies();
        assert!(cached.contains_key(&tensors[0].cache_key()), "held weight must survive");
        assert!(!cached.contains_key(&tensors[3].cache_key()), "prefill miss remains transient");
        assert_eq!(executor.stats()["cache_evictions"], 1);
        assert!(device.reserved() <= device.budget);
        drop((filler, held));
    }
}

#[path = "fast_dot_tests.rs"]
mod fast_dot_tests;
#[path = "direct_expert_tests.rs"]
mod direct_expert_tests;

#[path = "fast_dot_bench.rs"]
mod fast_dot_bench;

#[path = "context_capacity_tests.rs"]
mod context_capacity_tests;

#[path = "compact_kv_tests.rs"]
mod compact_kv_tests;
#[path = "compact_kv_gpu_tests.rs"]
mod compact_kv_gpu_tests;

#[test]
#[ignore = "requires explicit Vulkan Q3_K decoder execution"]
fn software_vulkan_q3_k_matches_cpu_with_base_and_fast_shader_variants() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    for fast_math in [false, true] {
        let executor = GpuMatvecExecutor::new(DeviceOptions {
            allow_software: true, fast_math, memory_budget_bytes: 64 * 1024 * 1024, ..Default::default()
        }).unwrap();
        for width in [256usize, 512, 5120] {
            let file = matrix_file(QuantType::Q3_K, width, 7, 91);
            let source = Gguf::open(&file.0).unwrap();
            let matrix = source.tensor("matrix").unwrap();
            let input: Vec<f32> = (0..width).map(|i| ((i * 29 % 113) as f32 - 56.0) / 59.0).collect();
            let reference = matrix.matvec(&input).unwrap();
            let mut output = vec![0.0; 7];
            for _ in 0..2 {
                executor.matvec_gpu(&matrix, &input, &mut output).unwrap();
                close(&output, &reference, "Q3_K base/fast decoder");
            }
        }
        assert_eq!(executor.stats()["gpu_projection_calls"], 6);
    }
}

#[path = "attention_gpu_tests.rs"]
mod attention_gpu_tests;
