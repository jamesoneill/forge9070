//! Reproducible warm-memory microbenchmark for the new CPU kernels.
//! This measures the checked row-dot API, not GPU or whole-model performance.
use crate::quant::{self, QuantType};
use anyhow::{Result, ensure};
use half::{bf16, f16};
use serde_json::{Value, json};
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
    ] {
        let encoded: Vec<Vec<u8>> = (0..rows).map(|r| encoded_row(ty, width, r)).collect();
        let mut max_abs_error = 0f32;
        let mut max_scaled_error = 0f32;
        for row in &encoded {
            let scalar = quant::dot_scalar(ty, row, &input)?;
            let native = quant::dot(ty, row, &input)?;
            let delta = (native - scalar).abs();
            let scaled = delta / scalar.abs().max(1.0);
            max_abs_error = max_abs_error.max(delta);
            max_scaled_error = max_scaled_error.max(scaled);
            ensure!(
                scaled <= 2e-4,
                "{} kernel correctness gate failed: native {native}, scalar {scalar}",
                ty.name()
            );
        }
        // Warm both paths before measurement. Alternate order in measured pairs
        // to avoid always giving either implementation the second-run advantage.
        timed(ty, &encoded, &input, 1, false)?;
        timed(ty, &encoded, &input, 1, true)?;
        let mut scalar_samples = Vec::new();
        let mut native_samples = Vec::new();
        for rep in 0..reps {
            let (scalar, native) = if rep % 2 == 0 {
                let s = timed(ty, &encoded, &input, iterations, false)?;
                (s, timed(ty, &encoded, &input, iterations, true)?)
            } else {
                let n = timed(ty, &encoded, &input, iterations, true)?;
                (timed(ty, &encoded, &input, iterations, false)?, n)
            };
            scalar_samples.push(scalar);
            native_samples.push(native);
        }
        let scalar_median = median(&scalar_samples);
        let native_median = median(&native_samples);
        reports.push(json!({
            "format": ty.name(), "selected_kernel": quant::dot_kernel(ty,width),
            "encoded_weight_bytes": encoded.iter().map(Vec::len).sum::<usize>(),
            "max_absolute_error": max_abs_error,
            "max_error_scaled_by_max_abs_reference_or_one": max_scaled_error,
            "correctness_scaled_error_limit": 2e-4,
            "scalar_seconds": scalar_samples, "selected_kernel_seconds": native_samples,
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
            "Warm memory, synthetic weights, alternated timing order; not a model token-rate benchmark",
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
    native: bool,
) -> Result<f64> {
    let start = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..iterations {
        for row in weights {
            let value = if native {
                quant::dot(ty, black_box(row), black_box(input))?
            } else {
                quant::dot_scalar(ty, black_box(row), black_box(input))?
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
        _ => unreachable!("the microbenchmark enumerates only its five generated formats"),
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn benchmark_rejects_unbounded_work_before_allocating() {
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(kernel_benchmark(127, 128, 8, 5).is_err());
        assert!(kernel_benchmark(2048, 0, 8, 5).is_err());
        assert!(kernel_benchmark(16384, 4096, 8, 5).is_err());
        assert!(kernel_benchmark(2048, 128, 0, 5).is_err());
    }
}
