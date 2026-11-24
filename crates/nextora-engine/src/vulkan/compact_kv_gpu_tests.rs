use super::super::context::{Commands, Device, Push, bytes_of_f32, bytes_of_f32_mut};
use super::compact_kv_tests::{attention_reference, pack_row, row_words};
use super::*;
use ash::vk;

fn words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn device() -> Arc<Device> {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    Device::new(&DeviceOptions {
        allow_software: true,
        memory_budget_bytes: 256 * 1024 * 1024,
        ..Default::default()
    })
    .unwrap()
}

#[test]
#[ignore = "explicit Vulkan numerical test"]
fn compact_kv_gpu_store_preserves_prefix_padding_and_status() {
    let device = device();
    let width = 169;
    let key: Vec<f32> = (0..width)
        .map(|i| {
            if i < 64 {
                0.0
            } else {
                ((i * 29 % 97) as f32 - 48.0) / 17.0
            }
        })
        .collect();
    let value: Vec<f32> = (0..width)
        .map(|i| ((i * 31 % 101) as f32 - 50.0) / 31.0)
        .collect();
    let source_k = device.upload(bytes_of_f32(&key)).unwrap();
    let source_v = device.upload(bytes_of_f32(&value)).unwrap();
    let words = row_words(width);
    let initial = vec![0x1234_5678u32; words * 4];
    let keys = device.upload(&words_bytes(&initial)).unwrap();
    let values = device.upload(&words_bytes(&initial)).unwrap();
    let status = device.storage(4).unwrap();
    let readback = device.host(initial.len() * 8 + 4).unwrap();
    let mut commands = Commands::new(Arc::clone(&device), 1).unwrap();
    let groups = commands.groups(width.div_ceil(128)).unwrap();
    let store = commands
        .operation(
            Push::new(3).n(0, width).unwrap().n(1, 2).unwrap().flags(64),
            [
                Arc::clone(&status),
                source_k,
                source_v,
                Arc::clone(&keys),
                Arc::clone(&values),
                Arc::clone(&status),
                Arc::clone(&status),
            ],
            groups,
        )
        .unwrap();
    commands.begin().unwrap();
    unsafe {
        device
            .raw
            .cmd_fill_buffer(commands.raw, status.raw, 0, 4, 0)
    };
    commands.transfer_to_compute();
    commands.record(&store, &store.push, store.groups);
    commands.compute_to_transfer();
    unsafe {
        for (buffer, offset, bytes) in [
            (&keys, 0, initial.len() * 4),
            (&values, initial.len() * 4, initial.len() * 4),
            (&status, initial.len() * 8, 4),
        ] {
            device.raw.cmd_copy_buffer(
                commands.raw,
                buffer.raw,
                readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset as u64,
                    size: bytes as u64,
                }],
            );
        }
    }
    commands.transfer_to_host();
    commands.submit().unwrap();
    let mut actual = vec![0u8; initial.len() * 8 + 4];
    readback.read(0, &mut actual).unwrap();
    let mut expected_k = initial.clone();
    let mut expected_v = initial;
    expected_k[words * 2..words * 3].copy_from_slice(&pack_row(&key).unwrap());
    expected_v[words * 2..words * 3].copy_from_slice(&pack_row(&value).unwrap());
    for (label, bytes, expected) in [
        ("key", &actual[..expected_k.len() * 4], &expected_k),
        (
            "value",
            &actual[expected_k.len() * 4..actual.len() - 4],
            &expected_v,
        ),
    ] {
        for (i, (got, want)) in bytes.chunks_exact(4).zip(expected).enumerate() {
            let got = u32::from_le_bytes(got.try_into().unwrap());
            assert_eq!(
                got, *want,
                "{label} word {i}: GPU {got:#010x} / CPU {want:#010x}"
            );
        }
    }
    assert_eq!(&actual[actual.len() - 4..], &[0; 4]);
}

fn gpu_attention(
    device: &Arc<Device>,
    query: &[f32],
    keys: &[u32],
    values: &[u32],
    history: usize,
    heads: usize,
    kv_heads: usize,
    dimension: usize,
) -> Vec<f32> {
    let chunks = history.div_ceil(1024);
    let max_chunks = chunks + 1;
    let query = device.upload(bytes_of_f32(query)).unwrap();
    let keys = device.upload(&words_bytes(keys)).unwrap();
    let values = device.upload(&words_bytes(values)).unwrap();
    let partials = device
        .storage(heads * max_chunks * (dimension + 2) * 4)
        .unwrap();
    let output = device.storage(heads * dimension * 4).unwrap();
    let status = device.storage(4).unwrap();
    let readback = device.host((heads * dimension + 1) * 4).unwrap();
    let mut commands = Commands::new(Arc::clone(device), 2).unwrap();
    let partial = commands
        .operation(
            Push::new(17)
                .n(0, history)
                .unwrap()
                .n(1, heads)
                .unwrap()
                .n(2, kv_heads)
                .unwrap()
                .n(3, dimension)
                .unwrap()
                .n(4, max_chunks)
                .unwrap()
                .n(5, kv_heads * dimension)
                .unwrap(),
            [
                Arc::clone(&status),
                query,
                keys,
                Arc::clone(&partials),
                values,
                Arc::clone(&status),
                Arc::clone(&status),
            ],
            [chunks, heads, 1],
        )
        .unwrap();
    let merge_groups = commands.groups(heads).unwrap();
    let merge = commands
        .operation(
            Push::new(18)
                .n(0, chunks)
                .unwrap()
                .n(1, heads)
                .unwrap()
                .n(3, dimension)
                .unwrap()
                .n(4, max_chunks)
                .unwrap(),
            [
                Arc::clone(&status),
                partials,
                Arc::clone(&status),
                Arc::clone(&output),
                Arc::clone(&status),
                Arc::clone(&status),
                Arc::clone(&status),
            ],
            merge_groups,
        )
        .unwrap();
    commands.begin().unwrap();
    unsafe {
        device
            .raw
            .cmd_fill_buffer(commands.raw, status.raw, 0, 4, 0)
    };
    commands.transfer_to_compute();
    commands.record(&partial, &partial.push, partial.groups);
    commands.record(&merge, &merge.push, merge.groups);
    commands.compute_to_transfer();
    unsafe {
        device.raw.cmd_copy_buffer(
            commands.raw,
            output.raw,
            readback.raw,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: (heads * dimension * 4) as u64,
            }],
        );
        device.raw.cmd_copy_buffer(
            commands.raw,
            status.raw,
            readback.raw,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: (heads * dimension * 4) as u64,
                size: 4,
            }],
        );
    }
    commands.transfer_to_host();
    commands.submit().unwrap();
    let mut actual = vec![0.0; heads * dimension];
    readback.read(0, bytes_of_f32_mut(&mut actual)).unwrap();
    let mut status = [0u8; 4];
    readback.read(heads * dimension * 4, &mut status).unwrap();
    assert_eq!(status, [0; 4]);
    actual
}

#[test]
#[ignore = "explicit Vulkan numerical test"]
fn compact_kv_gpu_attention_matches_independent_cpu() {
    let device = device();
    for history in [1usize, 17, 1024, 1025] {
        let heads = 4;
        let kv_heads = 2;
        let dimension = 64;
        let width = kv_heads * dimension;
        let query: Vec<f32> = (0..heads * dimension)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 19.0)
            .collect();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for token in 0..history {
            keys.extend(
                pack_row(
                    &(0..width)
                        .map(|i| ((i * 17 + token * 31) % 101) as f32 / 47.0 - 1.0)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            );
            values.extend(
                pack_row(
                    &(0..width)
                        .map(|i| ((i * 11 + token * 37) % 97) as f32 / 51.0 - 1.0)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            );
        }
        let actual = gpu_attention(
            &device, &query, &keys, &values, history, heads, kv_heads, dimension,
        );
        let expected =
            attention_reference(&query, &keys, &values, history, heads, kv_heads, dimension);
        close(&actual, &expected, "Q4 KV fused attention");
    }
}

#[test]
#[ignore = "explicit large-history Vulkan test; retains all 65537 and 524288 tokens"]
fn compact_kv_gpu_attention_65537_and_524288_keeps_oldest_tokens() {
    let device = device();
    let dimension = 128;
    for history in [65537usize, 524288] {
        let words = row_words(dimension);
        let keys = vec![0u32; words * history];
        let mut values = vec![0u32; words * history];
        let positions = [0, 65535, 65536, history - 1];
        let sentinels = [28672.0f32, 14336.0, -28672.0, 7168.0];
        // Multiple sentinels can share a token at the65537 boundary.
        let mut rows = std::collections::BTreeMap::<usize, Vec<f32>>::new();
        for (i, (&position, &value)) in positions.iter().zip(&sentinels).enumerate() {
            rows.entry(position).or_insert_with(|| vec![0.0; dimension])[i] = value;
        }
        let mut expected = vec![0.0; dimension];
        for (position, row) in rows {
            let packed = pack_row(&row).unwrap();
            for i in 0..dimension {
                expected[i] +=
                    super::compact_kv_tests::unpack(&packed, dimension, 0, i) / history as f32;
            }
            values[position * words..(position + 1) * words].copy_from_slice(&packed);
        }
        let actual = gpu_attention(
            &device,
            &vec![0.0; dimension],
            &keys,
            &values,
            history,
            1,
            1,
            dimension,
        );
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "history {history} coordinate {i}: {a}/{b}"
            );
        }
    }
}

#[test]
#[ignore = "explicit Vulkan numerical test"]
fn compact_kv_gpu_precomputed_rope_matches_cpu() {
    let device = device();
    let dimension = 128;
    let heads = 2;
    let magnitude = 1.125;
    let input: Vec<f32> = (0..dimension * heads)
        .map(|i| ((i * 19 % 97) as f32 - 48.0) / 51.0)
        .collect();
    let inverse: Vec<f32> = (0..dimension / 2)
        .map(|i| 1_000_000.0f32.powf(-(i as f32) / (dimension / 2) as f32) / 16.0)
        .collect();
    for position in [0usize, 262143, 262144, 524287] {
        let source = device.upload(bytes_of_f32(&input)).unwrap();
        let table_offset = 11usize;
        let mut table = vec![31.0; table_offset + dimension];
        for i in 0..dimension / 2 {
            let angle = position as f32 * inverse[i];
            let (sine, cosine) = angle.sin_cos();
            table[table_offset + i] = cosine * magnitude;
            table[table_offset + dimension / 2 + i] = sine * magnitude;
        }
        let inverse_buffer = device.upload(bytes_of_f32(&table)).unwrap();
        let output = device.storage(input.len() * 4).unwrap();
        let status = device.storage(4).unwrap();
        let readback = device.host(input.len() * 4 + 4).unwrap();
        let mut commands = Commands::new(Arc::clone(&device), 1).unwrap();
        let groups = commands.groups((input.len() / 2).div_ceil(128)).unwrap();
        let mut push = Push::new(2)
            .n(0, dimension)
            .unwrap()
            .n(1, heads)
            .unwrap()
            .n(2, position)
            .unwrap()
            .n(3, dimension)
            .unwrap()
            .flags(256);
        push.0[10] = table_offset as u32;
        let rope = commands
            .operation(
                push,
                [
                    Arc::clone(&status),
                    source,
                    inverse_buffer,
                    Arc::clone(&output),
                    Arc::clone(&status),
                    Arc::clone(&status),
                    Arc::clone(&status),
                ],
                groups,
            )
            .unwrap();
        commands.begin().unwrap();
        unsafe {
            device
                .raw
                .cmd_fill_buffer(commands.raw, status.raw, 0, 4, 0)
        };
        commands.transfer_to_compute();
        commands.record(&rope, &rope.push, rope.groups);
        commands.compute_to_transfer();
        unsafe {
            device.raw.cmd_copy_buffer(
                commands.raw,
                output.raw,
                readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (input.len() * 4) as u64,
                }],
            );
            device.raw.cmd_copy_buffer(
                commands.raw,
                status.raw,
                readback.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: (input.len() * 4) as u64,
                    size: 4,
                }],
            );
        }
        commands.transfer_to_host();
        commands.submit().unwrap();
        let mut actual = vec![0.0; input.len()];
        readback.read(0, bytes_of_f32_mut(&mut actual)).unwrap();
        let mut expected = vec![0.0; input.len()];
        for head in 0..heads {
            for i in 0..dimension / 2 {
                let cosine = table[table_offset + i];
                let sine = table[table_offset + dimension / 2 + i];
                let start = head * dimension;
                let lower = input[start + i];
                let upper = input[start + i + dimension / 2];
                expected[start + i] = lower * cosine - upper * sine;
                expected[start + i + dimension / 2] = lower * sine + upper * cosine;
            }
        }
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "CPU-table RoPE position {position} index {i}: {actual}/{expected}"
            );
        }
        let mut status = [0u8; 4];
        readback.read(input.len() * 4, &mut status).unwrap();
        assert_eq!(status, [0; 4]);
    }
}
