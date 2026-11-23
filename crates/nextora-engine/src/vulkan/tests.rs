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
                if ty == QuantType::Q6_K {
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
    for id in [0, 1, 2, 8, 12, 13, 14, 18, 20, 21, 22, 23, 30, 42] {
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
    assert_eq!(stats["gpu_projection_calls"].as_u64(), Some(56));
    assert!(stats["cache_hits"].as_u64().unwrap() >= 28);
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
