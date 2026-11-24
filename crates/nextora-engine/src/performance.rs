//! Reproducible warm-memory microbenchmark for the new CPU kernels.
//! This measures the checked row-dot API, not GPU or whole-model performance.
use crate::quant::{self, QuantType};
use anyhow::{ensure, Result};
use half::{bf16, f16};
use serde_json::{json, Value};
use std::{hint::black_box, time::Instant};

pub fn kernel_benchmark(
    width: usize,
    rows: usize,
    iterations: usize,
    reps: usize,
) -> Result<Value> {
    ensure!(
        (128..=16384).contains(&width) && width % 32 == 0,
        "kernel width must be a multiple of 32 in 128..=16384"
    );
    ensure!(
        (1..=4096).contains(&rows),
        "kernel rows must be in 1..=4096"
    );
    ensure!(
        width * rows <= 32 * 1024 * 1024,
        "kernel workload exceeds 128 MiB F32 storage limit"
    );
    ensure!(
        (1..=64).contains(&iterations),
        "kernel iterations must be in 1..=64"
    );
    ensure!((3..=20).contains(&reps), "kernel reps must be in 3..=20");
    let input: Vec<f32> = (0..width)
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 127.0)
        .collect();
    let mut reports = Vec::new();
    for ty in [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q8_0,
        QuantType::Q4_K,
        QuantType::Q6_K,
    ] {
        if width % ty.block_elements() != 0 {
            continue;
        }
        let encoded: Vec<Vec<u8>> = (0..rows).map(|r| encoded_row(ty, width, r)).collect();
        let mut max_abs_error = 0f32;
        let mut max_scaled_error = 0f32;
        let mut max_f64_errors = [0.0f64; 3];
        let mut max_f64_scaled_errors = [0.0f64; 3];
        let k_quant = matches!(ty, QuantType::Q4_K | QuantType::Q6_K);
        for (row_index, row) in encoded.iter().enumerate() {
            let scalar = quant::dot_scalar(ty, row, &input)?;
            let native = quant::dot(ty, row, &input)?;
            let previous = quant::dot_avx2_baseline(ty, row, &input)?;
            let delta = (native - scalar).abs();
            let scaled = delta / scalar.abs().max(1.0);
            max_abs_error = max_abs_error.max(delta);
            max_scaled_error = max_scaled_error.max(scaled);
            if k_quant {
                let mut decoded = vec![0.0; width];
                quant::decode_row(ty, row, &mut decoded)?;
                let reference: f64 = decoded
                    .iter()
                    .zip(&input)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum();
                let errors = [
                    (f64::from(scalar) - reference).abs(),
                    (f64::from(native) - reference).abs(),
                    (f64::from(previous) - reference).abs(),
                ];
                for i in 0..3 {
                    max_f64_errors[i] = max_f64_errors[i].max(errors[i]);
                    max_f64_scaled_errors[i] =
                        max_f64_scaled_errors[i].max(errors[i] / reference.abs().max(1.0));
                }
                // New K-format cases use independent F64 products as ground
                // truth. Scalar and previous AVX2 errors are reported; the
                // selected result must meet the unchanged 2e-4 scaled limit.
                ensure!(errors[1]/reference.abs().max(1.0)<=2e-4,
                    "{} kernel correctness gate failed at row {row_index}: selected {native}, scalar {scalar}, previous {previous}, f64 {reference}",ty.name());
            } else {
                ensure!(
                    scaled <= 2e-4,
                    "{} kernel correctness gate failed: native {native}, scalar {scalar}",
                    ty.name()
                );
            }
        }
        // Warm all three paths, then rotate measurement order so no path
        // consistently receives the cache or clock state of the last run.
        for path in 0..3 {
            timed(ty, &encoded, &input, 1, path)?;
        }
        let mut scalar_samples = Vec::new();
        let mut native_samples = Vec::new();
        let mut previous_samples = Vec::new();
        for rep in 0..reps {
            let mut timings = [0.0; 3];
            for offset in 0..3 {
                let path = (rep + offset) % 3;
                timings[path] = timed(ty, &encoded, &input, iterations, path)?;
            }
            scalar_samples.push(timings[0]);
            native_samples.push(timings[1]);
            previous_samples.push(timings[2]);
        }
        let scalar_median = median(&scalar_samples);
        let native_median = median(&native_samples);
        let previous_median = median(&previous_samples);
        reports.push(json!({
            "format": ty.name(), "selected_kernel": quant::dot_kernel(ty,width),
            "encoded_weight_bytes": encoded.iter().map(Vec::len).sum::<usize>(),
            "max_absolute_error": max_abs_error,
            "max_error_scaled_by_max_abs_reference_or_one": max_scaled_error,
            "correctness_scaled_error_limit": 2e-4,
            "correctness_reference":if k_quant {"f64_dot_of_independently_dequantized_weights"} else {"scalar_f32_dot"},
            "k_quant_f64_max_absolute_errors_scalar_selected_previous": if k_quant {Some(max_f64_errors)} else {None},
            "k_quant_f64_max_scaled_errors_scalar_selected_previous":if k_quant {Some(max_f64_scaled_errors)} else {None},
            "scalar_seconds": scalar_samples, "selected_kernel_seconds": native_samples,
            "previous_avx2_or_scalar_seconds": previous_samples,
            "previous_avx2_or_scalar_median_seconds": previous_median,
            "previous_avx2_over_selected_time_ratio": previous_median / native_median,
            "scalar_median_seconds": scalar_median, "selected_kernel_median_seconds": native_median,
            "scalar_over_selected_time_ratio": scalar_median / native_median,
            "selected_kernel_rows_per_second": (rows * iterations) as f64 / native_median,
        }));
    }
    Ok(json!({
        "engine": crate::ENGINE_ID, "version": crate::ENGINE_VERSION,
        "benchmark": "checked_cpu_row_dot_warm_memory",
        "host_arch": std::env::consts::ARCH, "host_os": std::env::consts::OS,
        "row_width": width, "rows": rows, "iterations_per_sample": iterations,
        "repetitions": reps, "samples": reports,
        "notes": [
            "Both scalar and selected paths include public row validation",
            "Serial row evaluation; excludes Rayon, attention and model orchestration",
            "Warm memory, synthetic weights, rotating scalar/selected/previous timing order; not a model token-rate benchmark",
            "K-quant formats are included only when width is divisible by their 256-value block size",
            "New K-quant cases require selected-versus-F64 scaled error <=2e-4; scalar and previous AVX2 errors are reported independently",
            "FMA and reduction ordering may round differently; comparison uses full output values",
            "A ratio above one means selected kernel took less median time on this host",
            "No claim about RX9070XT GPU throughput, real-model speed or answer quality"
        ]
    }))
}

fn timed(
    ty: QuantType,
    weights: &[Vec<u8>],
    input: &[f32],
    iterations: usize,
    path: usize,
) -> Result<f64> {
    let start = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..iterations {
        for row in weights {
            let value = match path {
                0 => quant::dot_scalar(ty, black_box(row), black_box(input))?,
                1 => quant::dot(ty, black_box(row), black_box(input))?,
                2 => quant::dot_avx2_baseline(ty, black_box(row), black_box(input))?,
                _ => unreachable!("three measured CPU implementations"),
            };
            checksum += black_box(value) as f64;
        }
    }
    black_box(checksum);
    Ok(start.elapsed().as_secs_f64().max(f64::MIN_POSITIVE))
}

fn median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let i = ordered.len() / 2;
    if ordered.len() % 2 == 0 {
        (ordered[i - 1] + ordered[i]) / 2.0
    } else {
        ordered[i]
    }
}

fn encoded_row(ty: QuantType, width: usize, row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ty.row_bytes(width).unwrap());
    let code = |col: usize| ((row * 53 + col * 29 + col / 7) % 251) as i32 - 125;
    match ty {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            for col in 0..width {
                let value = code(col) as f32 / 128.0;
                match ty {
                    QuantType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                    QuantType::F16 => bytes.extend_from_slice(&f16::from_f32(value).to_le_bytes()),
                    QuantType::BF16 => {
                        bytes.extend_from_slice(&bf16::from_f32(value).to_le_bytes())
                    }
                    _ => unreachable!(),
                }
            }
        }
        QuantType::Q8_0 => {
            for block in 0..width / 32 {
                bytes.extend_from_slice(&f16::from_f32(1.0 / 128.0).to_le_bytes());
                for offset in 0..32 {
                    bytes.push(code(block * 32 + offset) as i8 as u8);
                }
            }
        }
        QuantType::Q4_0 => {
            for block in 0..width / 32 {
                bytes.extend_from_slice(&f16::from_f32(1.0 / 8.0).to_le_bytes());
                for offset in 0..16 {
                    let low = code(block * 32 + offset).rem_euclid(16) as u8;
                    let high = code(block * 32 + offset + 16).rem_euclid(16) as u8;
                    bytes.push(low | high << 4);
                }
            }
        }
        QuantType::Q4_K | QuantType::Q6_K => {
            for block in 0..width / 256 {
                let start = bytes.len();
                bytes.extend((0..ty.block_bytes()).map(|i| (row * 53 + block * 97 + i * 29) as u8));
                let d = f16::from_f32(if (row + block) % 2 == 0 {
                    1.0 / 128.0
                } else {
                    -1.0 / 64.0
                });
                if ty == QuantType::Q4_K {
                    bytes[start..start + 2].copy_from_slice(&d.to_le_bytes());
                    bytes[start + 2..start + 4]
                        .copy_from_slice(&f16::from_f32(1.0 / 256.0).to_le_bytes());
                } else {
                    bytes[start + 208..start + 210].copy_from_slice(&d.to_le_bytes());
                }
            }
        }
        _ => unreachable!("the microbenchmark enumerates only generated formats"),
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_native_cpu_model_matches_scalar_projection_reference() {
        use crate::{
            gguf::{Gguf, Tensor},
            model::{MatvecExecutor, Model},
        };
        use std::sync::Arc;
        #[derive(Debug)]
        struct Scalar;
        impl MatvecExecutor for Scalar {
            fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> Result<()> {
                let stride = tensor.quant_type().row_bytes(input.len())?;
                for (row, out) in tensor.bytes().chunks_exact(stride).zip(output) {
                    *out = quant::dot_scalar(tensor.quant_type(), row, input)?;
                }
                Ok(())
            }
            fn name(&self) -> &'static str {
                "test_scalar"
            }
        }
        fn string(s: &str) -> Vec<u8> {
            [(s.len() as u64).to_le_bytes().as_slice(), s.as_bytes()].concat()
        }
        fn integer(key: &str, value: u32) -> Vec<u8> {
            [
                string(key),
                4u32.to_le_bytes().to_vec(),
                value.to_le_bytes().to_vec(),
            ]
            .concat()
        }
        fn float(key: &str, value: f32) -> Vec<u8> {
            [
                string(key),
                6u32.to_le_bytes().to_vec(),
                value.to_le_bytes().to_vec(),
            ]
            .concat()
        }
        struct Temporary(std::path::PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        for ty in [QuantType::Q4_K, QuantType::Q6_K] {
            let mut metadata = vec![[
                string("general.architecture"),
                8u32.to_le_bytes().to_vec(),
                string("qwen3"),
            ]
            .concat()];
            for (key, value) in [
                ("block_count", 1),
                ("embedding_length", 256),
                ("feed_forward_length", 256),
                ("attention.head_count", 8),
                ("attention.head_count_kv", 2),
                ("attention.key_length", 32),
                ("rope.dimension_count", 32),
                ("context_length", 16),
            ] {
                metadata.push(integer(&format!("qwen3.{key}"), value));
            }
            metadata.push(float("qwen3.attention.layer_norm_rms_epsilon", 1e-6));
            metadata.push(float("qwen3.rope.freq_base", 10000.0));
            let shapes = vec![
                ("token_embd.weight", vec![256, 16]),
                ("output.weight", vec![256, 16]),
                ("output_norm.weight", vec![256]),
                ("blk.0.attn_norm.weight", vec![256]),
                ("blk.0.attn_q_norm.weight", vec![32]),
                ("blk.0.attn_k_norm.weight", vec![32]),
                ("blk.0.ffn_norm.weight", vec![256]),
                ("blk.0.attn_q.weight", vec![256, 256]),
                ("blk.0.attn_k.weight", vec![256, 64]),
                ("blk.0.attn_v.weight", vec![256, 64]),
                ("blk.0.attn_output.weight", vec![256, 256]),
                ("blk.0.ffn_gate.weight", vec![256, 256]),
                ("blk.0.ffn_up.weight", vec![256, 256]),
                ("blk.0.ffn_down.weight", vec![256, 256]),
            ];
            let mut descriptors = Vec::new();
            let mut payload = Vec::new();
            for (name, dims) in &shapes {
                let kind = if dims.len() == 1 { QuantType::F32 } else { ty };
                descriptors.extend(string(name));
                descriptors.extend((dims.len() as u32).to_le_bytes());
                for dim in dims {
                    descriptors.extend((*dim as u64).to_le_bytes());
                }
                descriptors.extend(kind.id().to_le_bytes());
                descriptors.extend((payload.len() as u64).to_le_bytes());
                if dims.len() == 1 {
                    for _ in 0..dims[0] {
                        payload.extend(1f32.to_le_bytes());
                    }
                } else {
                    for row in 0..dims[1] {
                        payload.extend(encoded_row(ty, 256, row));
                    }
                }
                payload.resize(payload.len().next_multiple_of(32), 0);
            }
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend((shapes.len() as u64).to_le_bytes());
            bytes.extend((metadata.len() as u64).to_le_bytes());
            for entry in metadata {
                bytes.extend(entry);
            }
            bytes.extend(descriptors);
            bytes.resize(bytes.len().next_multiple_of(32), 0);
            bytes.extend(payload);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let file = Temporary(std::env::temp_dir().join(format!(
                "nextora-k-quant-cpu-{}-{stamp}.gguf",
                std::process::id()
            )));
            std::fs::write(&file.0, bytes).unwrap();
            let source = Arc::new(Gguf::open(&file.0).unwrap());
            let selected = Model::load(Arc::clone(&source), 16).unwrap();
            let scalar = Model::load_with_executor(source, 16, Arc::new(Scalar)).unwrap();
            let mut actual_state = selected.new_sequence();
            let mut reference_state = scalar.new_sequence();
            for token in [1, 3, 7, 2] {
                let actual = selected.step(&mut actual_state, token).unwrap();
                let expected = scalar.step(&mut reference_state, token).unwrap();
                for (a, b) in actual
                    .logits
                    .iter()
                    .chain(&actual.hidden)
                    .zip(expected.logits.iter().chain(&expected.hidden))
                {
                    assert!(
                        (a - b).abs() <= 5e-4 * b.abs().max(1.0),
                        "{ty:?}: {a} != {b}"
                    );
                }
            }
        }
    }
    #[test]
    fn k_quant_cancellation_checks_selected_against_f64_reference() {
        // Include all synthetic row phases: Q6_K has near cancellation at
        // 4096/row19 and5120/row205, so scalar F32 is not an exact oracle.
        for width in [4096, 5120] {
            let report = kernel_benchmark(width, 256, 1, 3).unwrap();
            let rows = report["samples"].as_array().unwrap();
            assert_eq!(rows.len(), 7);
            for row in rows
                .iter()
                .filter(|row| row["format"] == "Q4_K" || row["format"] == "Q6_K")
            {
                assert_eq!(
                    row["correctness_reference"],
                    "f64_dot_of_independently_dequantized_weights"
                );
                assert!(
                    row["k_quant_f64_max_scaled_errors_scalar_selected_previous"][1]
                        .as_f64()
                        .unwrap()
                        <= 2e-4
                );
            }
        }
    }
    #[test]
    fn benchmark_rejects_unbounded_work_before_allocating() {
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(kernel_benchmark(127, 128, 8, 5).is_err());
        assert!(kernel_benchmark(2048, 0, 8, 5).is_err());
        assert!(kernel_benchmark(16384, 4096, 8, 5).is_err());
        assert!(kernel_benchmark(2048, 128, 0, 5).is_err());
    }
}
