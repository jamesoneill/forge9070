use super::super::context::{Commands, Device, Push, bytes_of_f32, bytes_of_f32_mut};
use super::*;
use ash::vk;

fn quantized_reference(input: &[f32]) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut values = Vec::new();
    for group in input.chunks(32) {
        let maximum = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = maximum / 127.0;
        scales.push(scale);
        let mut quantized = [0i8; 32];
        for (i, &value) in group.iter().enumerate() {
            let q = if scale > 0.0 {
                (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            quantized[i] = q;
            values.push(q as f32 * scale);
        }
        for word in quantized.chunks_exact(4) {
            packed.push(
                word.iter()
                    .enumerate()
                    .fold(0u32, |n, (i, &q)| n | ((q as u8 as u32) << (i * 8))),
            );
        }
    }
    (packed, scales, values)
}

#[test]
fn q8_reference_rounding_zero_and_error_bound() {
    let mut input = vec![0.0; 32];
    input.extend([127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -127.0]);
    let (packed, scales, values) = quantized_reference(&input);
    assert_eq!(scales, vec![0.0, 1.0]);
    assert_eq!(
        &values[32..],
        &[127.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0, -127.0]
    );
    assert_eq!(packed.len(), 16);
    for (i, (&original, &quantized)) in input.iter().zip(&values).enumerate() {
        assert!((original - quantized).abs() <= scales[i / 32] * 0.5 + 1e-6);
    }
}

#[test]
#[ignore = "requires explicit Vulkan integer-dot execution environment"]
fn software_vulkan_fast_dot_matches_q8_reference() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let device = Device::new(&DeviceOptions {
        allow_software: true,
        fast_math: true,
        memory_budget_bytes: 64 * 1024 * 1024,
        ..Default::default()
    })
    .unwrap();
    for ty in [QuantType::Q4_K, QuantType::Q6_K, QuantType::Q5_K] {
        for width in [256usize, 512, 2048] {
            let rows = 7;
            let file = matrix_file(ty, width, rows, 31);
            let source = Gguf::open(&file.0).unwrap();
            let matrix = source.tensor("matrix").unwrap();
            let input: Vec<f32> = (0..width)
                .map(|i| {
                    if i < 32 {
                        0.0
                    } else if i == 32 {
                        1.0
                    } else if i < 64 {
                        -1.0 / 127.0
                    } else {
                        ((i * 43 % 211) as f32 - 105.0) / 107.0
                    }
                })
                .collect();
            // Deliberately exercise nonzero offsets and Q8 words whose raw
            // integer patterns would be NaNs if stored as floating-point data.
            let input_offset = 32;
            let prefix_groups = input_offset / 32;
            let mut source_values = vec![0.25; input_offset];
            source_values.extend_from_slice(&input);
            let (expected_words, scales, quantized) = quantized_reference(&input);
            let groups = width.div_ceil(32);
            let weights = device.upload(matrix.bytes()).unwrap();
            let source = device.upload(bytes_of_f32(&source_values)).unwrap();
            let packed = device.storage((groups + prefix_groups) * 8 * 4).unwrap();
            let scale = device.storage((groups + prefix_groups) * 4).unwrap();
            let output = device.storage(rows * 4).unwrap();
            let status = device.storage(4).unwrap();
            let dummy = device.storage(4).unwrap();
            let readback = device
                .host(rows * 4 + groups * 8 * 4 + groups * 4 + 4)
                .unwrap();
            let mut commands = Commands::new(Arc::clone(&device), 2).unwrap();
            let quant_groups = commands.groups(groups).unwrap();
            let mut quant_push = Push::new(16).n(0, width).unwrap();
            quant_push.0[10] = input_offset as u32;
            quant_push.0[11] = (prefix_groups * 8) as u32;
            quant_push.0[13] = prefix_groups as u32;
            let quant = commands
                .operation(
                    quant_push,
                    [
                        Arc::clone(&dummy),
                        Arc::clone(&source),
                        Arc::clone(&packed),
                        Arc::clone(&dummy),
                        Arc::clone(&scale),
                        Arc::clone(&dummy),
                        Arc::clone(&status),
                    ],
                    quant_groups,
                )
                .unwrap();
            let gemv_groups = commands.groups(rows).unwrap();
            let mut gemv_push = Push::new(0)
                .n(0, width)
                .unwrap()
                .n(1, rows)
                .unwrap()
                .n(2, ty.id() as usize)
                .unwrap()
                .n(3, ty.row_bytes(width).unwrap())
                .unwrap()
                .n(4, 1)
                .unwrap()
                .flags(32);
            gemv_push.0[11] = input_offset as u32;
            let gemv = commands
                .operation(
                    gemv_push,
                    [
                        Arc::clone(&weights),
                        Arc::clone(&source),
                        Arc::clone(&packed),
                        Arc::clone(&output),
                        Arc::clone(&scale),
                        Arc::clone(&dummy),
                        Arc::clone(&status),
                    ],
                    gemv_groups,
                )
                .unwrap();
            commands.begin().unwrap();
            unsafe {
                device
                    .raw
                    .cmd_fill_buffer(commands.raw, status.raw, 0, 4, 0)
            };
            commands.transfer_to_compute();
            commands.record(&quant, &quant.push, quant.groups);
            commands.record(&gemv, &gemv.push, gemv.groups);
            commands.compute_to_transfer();
            let mut offset = 0;
            for (buffer, bytes, source_offset) in [
                (&output, rows * 4, 0),
                (&packed, groups * 8 * 4, prefix_groups * 8 * 4),
                (&scale, groups * 4, prefix_groups * 4),
                (&status, 4, 0),
            ] {
                unsafe {
                    device.raw.cmd_copy_buffer(
                        commands.raw,
                        buffer.raw,
                        readback.raw,
                        &[vk::BufferCopy {
                            src_offset: source_offset as u64,
                            dst_offset: offset as u64,
                            size: bytes as u64,
                        }],
                    )
                };
                offset += bytes;
            }
            commands.transfer_to_host();
            commands.submit().unwrap();
            let mut actual = vec![0.0; rows];
            readback.read(0, bytes_of_f32_mut(&mut actual)).unwrap();
            let mut words = vec![0u8; expected_words.len() * 4];
            readback.read(rows * 4, &mut words).unwrap();
            let actual_words: Vec<_> = words
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual_words, expected_words, "Q8 activation packing");
            let mut actual_scales = vec![0.0; groups];
            readback
                .read(
                    rows * 4 + groups * 8 * 4,
                    bytes_of_f32_mut(&mut actual_scales),
                )
                .unwrap();
            close(&actual_scales, &scales, "Q8 activation scales");
            let mut bad = [0u8; 4];
            readback.read(offset - 4, &mut bad).unwrap();
            assert_eq!(bad, [0; 4]);
            let truth = matrix
                .matvec(if matches!(ty, QuantType::Q4_K | QuantType::Q6_K) {
                    &quantized
                } else {
                    &input
                })
                .unwrap();
            close(&actual, &truth, "packed integer dot / Q8 reference");
            if matches!(ty, QuantType::Q4_K | QuantType::Q6_K) {
                let original = matrix.matvec(&input).unwrap();
                let mut weights = vec![0.0; width];
                for row in 0..rows {
                    matrix.read_row(row, &mut weights).unwrap();
                    let bound: f32 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| w.abs() * scales[i / 32] * 0.5)
                        .sum();
                    assert!(
                        (actual[row] - original[row]).abs() <= bound * 1.01 + 0.005,
                        "Q8 worst-case bound"
                    );
                }
            }
        }
    }
}
