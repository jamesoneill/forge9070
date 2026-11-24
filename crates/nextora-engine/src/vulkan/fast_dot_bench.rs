//! Opt-in isolated kernel measurements; these are not model throughput results.
//! Run with NEXTORA_VULKAN_TEST=1 and --ignored --nocapture. For a quick software
//! smoke check also set NEXTORA_VULKAN_BENCH_SOFTWARE=1 and
//! NEXTORA_VULKAN_BENCH_SMALL=1. Iterations and samples can be set through
//! NEXTORA_VULKAN_BENCH_ITERATIONS and NEXTORA_VULKAN_BENCH_SAMPLES.

use super::super::context::{
    Buffer, Commands, Device, Operation, Push, bytes_of_f32, bytes_of_f32_mut,
};
use super::*;
use ash::vk;
use serde_json::{Value, json};
use std::time::Instant;

struct Timer {
    device: Arc<Device>,
    pool: vk::QueryPool,
}

impl Timer {
    fn new(device: &Arc<Device>) -> Option<Self> {
        if device.properties.limits.timestamp_compute_and_graphics == 0 {
            return None;
        }
        let pool = unsafe {
            device.raw.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .unwrap();
        Some(Self {
            device: Arc::clone(device),
            pool,
        })
    }

    fn begin(&self, commands: &Commands) {
        unsafe {
            self.device
                .raw
                .cmd_reset_query_pool(commands.raw, self.pool, 0, 2);
            self.device.raw.cmd_write_timestamp(
                commands.raw,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.pool,
                0,
            );
        }
    }

    fn end(&self, commands: &Commands) {
        unsafe {
            self.device.raw.cmd_write_timestamp(
                commands.raw,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                1,
            );
        }
    }

    fn elapsed_ms(&self) -> Option<f64> {
        let mut ticks = [0u64; 2];
        unsafe {
            self.device.raw.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .unwrap();
        // A short counter wrapping during the batch invalidates this sample;
        // submission-to-fence timing remains available independently.
        ticks[1]
            .checked_sub(ticks[0])
            .map(|delta| delta as f64 * self.device.properties.limits.timestamp_period as f64 / 1e6)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        unsafe { self.device.raw.destroy_query_pool(self.pool, None) };
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Gemv,
    Quantize,
    QuantizeAndGemv,
}

struct Kernel {
    device: Arc<Device>,
    commands: Commands,
    quantize: Option<Operation>,
    gemv: Operation,
    status: Arc<Buffer>,
    readback: Arc<Buffer>,
    timer: Option<Timer>,
    rows: usize,
}

impl Kernel {
    fn new(device: &Arc<Device>, matrix: &crate::gguf::Tensor<'_>, input: &[f32]) -> Self {
        let width = matrix.width();
        let rows = matrix.rows();
        let groups = width.div_ceil(32);
        let weight = device.upload(matrix.bytes()).unwrap();
        let source = device.upload(bytes_of_f32(input)).unwrap();
        let packed = device.storage(groups * 8 * 4).unwrap();
        let scales = device.storage(groups * 4).unwrap();
        let output = device.storage(rows * 4).unwrap();
        let status = device.storage(4).unwrap();
        let dummy = device.storage(4).unwrap();
        let readback = device.host(rows * 4 + 4).unwrap();
        let mut commands = Commands::new(Arc::clone(device), 2).unwrap();
        let quantize = if device.fast_math {
            let dispatch = commands.groups(groups).unwrap();
            Some(
                commands
                    .operation(
                        Push::new(16).n(0, width).unwrap(),
                        [
                            Arc::clone(&dummy),
                            Arc::clone(&source),
                            Arc::clone(&packed),
                            Arc::clone(&dummy),
                            Arc::clone(&scales),
                            Arc::clone(&dummy),
                            Arc::clone(&status),
                        ],
                        dispatch,
                    )
                    .unwrap(),
            )
        } else {
            None
        };
        let dispatch = commands.groups(rows).unwrap();
        let gemv = commands
            .operation(
                Push::new(0)
                    .n(0, width)
                    .unwrap()
                    .n(1, rows)
                    .unwrap()
                    .n(2, matrix.quant_type().id() as usize)
                    .unwrap()
                    .n(3, matrix.quant_type().row_bytes(width).unwrap())
                    .unwrap()
                    .n(4, 1)
                    .unwrap()
                    .flags(if device.fast_math { 32 } else { 0 }),
                [
                    weight,
                    source,
                    packed,
                    output,
                    scales,
                    dummy,
                    Arc::clone(&status),
                ],
                dispatch,
            )
            .unwrap();
        Self {
            device: Arc::clone(device),
            commands,
            quantize,
            gemv,
            status,
            readback,
            timer: Timer::new(device),
            rows,
        }
    }

    fn sample(&mut self, phase: Phase, iterations: usize) -> Value {
        self.commands.begin().unwrap();
        unsafe {
            self.device
                .raw
                .cmd_fill_buffer(self.commands.raw, self.status.raw, 0, 4, 0)
        };
        self.commands.transfer_to_compute();
        if let Some(timer) = &self.timer {
            timer.begin(&self.commands);
        }
        for _ in 0..iterations {
            if matches!(phase, Phase::Quantize | Phase::QuantizeAndGemv) {
                let operation = self.quantize.as_ref().unwrap();
                self.commands
                    .record(operation, &operation.push, operation.groups);
            }
            if matches!(phase, Phase::Gemv | Phase::QuantizeAndGemv) {
                self.commands
                    .record(&self.gemv, &self.gemv.push, self.gemv.groups);
            }
        }
        if let Some(timer) = &self.timer {
            timer.end(&self.commands);
        }
        let start = Instant::now();
        self.commands.submit().unwrap();
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let gpu_ms = self.timer.as_ref().and_then(Timer::elapsed_ms);
        json!({ "iterations": iterations, "gpu_ms": gpu_ms, "fence_ms": wall_ms,
                "gpu_us_per_iteration": gpu_ms.map(|ms| ms * 1e3 / iterations as f64),
                "fence_us_per_iteration": wall_ms * 1e3 / iterations as f64 })
    }

    fn output(&mut self) -> Vec<f32> {
        self.commands.begin().unwrap();
        self.commands.compute_to_transfer();
        unsafe {
            self.device.raw.cmd_copy_buffer(
                self.commands.raw,
                self.gemv.buffers[3].raw,
                self.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (self.rows * 4) as u64,
                }],
            );
            self.device.raw.cmd_copy_buffer(
                self.commands.raw,
                self.status.raw,
                self.readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: (self.rows * 4) as u64,
                    size: 4,
                }],
            );
        }
        self.commands.transfer_to_host();
        self.commands.submit().unwrap();
        let mut output = vec![0.0; self.rows];
        self.readback
            .read(0, bytes_of_f32_mut(&mut output))
            .unwrap();
        let mut status = [0u8; 4];
        self.readback.read(self.rows * 4, &mut status).unwrap();
        assert_eq!(status, [0; 4]);
        assert!(output.iter().all(|value| value.is_finite()));
        output
    }
}

fn check_rows(matrix: &crate::gguf::Tensor<'_>, input: &[f32], output: &[f32]) {
    for row in [0, matrix.rows() / 2, matrix.rows() - 1] {
        let weights = matrix.row(row).unwrap();
        let expected: f32 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
        let absolute_sum: f32 = weights.iter().zip(input).map(|(w, x)| (w * x).abs()).sum();
        assert!(
            (output[row] - expected).abs() <= 0.003 + 0.0005 * expected.abs() + 4e-6 * absolute_sum,
            "row {row}: GPU {} versus reference {expected}",
            output[row]
        );
    }
}

fn positive_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(default)
        .clamp(1, 4096)
}

#[test]
#[ignore = "explicit isolated GPU benchmark; reports kernel timing, not model token rate"]
fn paired_vulkan_f32_q8_gemv_microbenchmark() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let software = std::env::var("NEXTORA_VULKAN_BENCH_SOFTWARE").as_deref() == Ok("1");
    let small = std::env::var("NEXTORA_VULKAN_BENCH_SMALL").as_deref() == Ok("1");
    assert!(
        !software || small,
        "software smoke checks must use small matrices"
    );
    let device_index = std::env::var("NEXTORA_VULKAN_BENCH_DEVICE")
        .ok()
        .map(|s| s.parse().unwrap());
    let options = DeviceOptions {
        device_index,
        allow_software: software,
        fast_math: false,
                direct_experts: None,
        kv_precision: crate::vulkan::KvPrecision::F32,
        kv_placement: crate::vulkan::KvPlacement::Gpu,
        host_memory_budget_bytes: 0,
        memory_budget_bytes: 1024 * 1024 * 1024,
    };
    let f32_device = Device::new(&options).unwrap();
    let q8_device = Device::new(&DeviceOptions {
        device_index: Some(f32_device.info()["index"].as_u64().unwrap() as usize),
        fast_math: true,
        ..options
    })
    .unwrap();
    let iterations = positive_env(
        "NEXTORA_VULKAN_BENCH_ITERATIONS",
        if small { 2 } else { 64 },
    );
    let samples = positive_env("NEXTORA_VULKAN_BENCH_SAMPLES", if small { 1 } else { 3 });
    let shapes: &[(usize, usize)] = if small {
        &[(256, 7), (512, 11), (2048, 13)]
    } else {
        &[(2048, 2048), (2048, 768), (768, 2048), (2048, 151936)]
    };
    println!(
        "NEXTORA_KERNEL_BENCH {}",
        json!({ "adapter": f32_device.info(),
        "iterations": iterations, "samples": samples, "small": small,
        "measurement": "warm resident synthetic GGUF weights; no uploads or readbacks timed" })
    );
    for ty in [QuantType::Q4_K, QuantType::Q6_K] {
        for &(width, rows) in shapes {
            let file = matrix_file(ty, width, rows, 97);
            let source = Gguf::open(&file.0).unwrap();
            let matrix = source.tensor("matrix").unwrap();
            let input: Vec<f32> = (0..width)
                .map(|i| ((i * 43 % 211) as f32 - 105.0) / 107.0)
                .collect();
            let mut quantized = Vec::with_capacity(width);
            for group in input.chunks(32) {
                let scale = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 127.0;
                quantized.extend(
                    group
                        .iter()
                        .map(|x| (x / scale).round_ties_even().clamp(-127.0, 127.0) * scale),
                );
            }
            let mut f32 = Kernel::new(&f32_device, &matrix, &input);
            let mut q8 = Kernel::new(&q8_device, &matrix, &input);
            f32.sample(Phase::Gemv, 1);
            q8.sample(Phase::QuantizeAndGemv, 1);
            check_rows(&matrix, &input, &f32.output());
            check_rows(&matrix, &quantized, &q8.output());
            // Warm each measured phase, then alternate execution order to limit
            // a consistent bias from clock ramping or test ordering.
            f32.sample(Phase::Gemv, iterations);
            q8.sample(Phase::QuantizeAndGemv, iterations);
            let mut f32_samples = Vec::new();
            let mut q8_samples = Vec::new();
            let mut quantize_samples = Vec::new();
            let mut combined_samples = Vec::new();
            for sample in 0..samples {
                if sample % 2 == 0 {
                    f32_samples.push(f32.sample(Phase::Gemv, iterations));
                    q8_samples.push(q8.sample(Phase::Gemv, iterations));
                } else {
                    q8_samples.push(q8.sample(Phase::Gemv, iterations));
                    f32_samples.push(f32.sample(Phase::Gemv, iterations));
                }
                quantize_samples.push(q8.sample(Phase::Quantize, iterations));
                combined_samples.push(q8.sample(Phase::QuantizeAndGemv, iterations));
            }
            println!(
                "NEXTORA_KERNEL_BENCH {}",
                json!({ "quant_type": format!("{ty:?}"),
                "width": width, "rows": rows, "weight_bytes": matrix.bytes().len(),
                "f32_gemv": f32_samples, "q8_gemv": q8_samples,
                "q8_quantize": quantize_samples, "q8_quantize_and_gemv": combined_samples })
            );
        }
    }
}
