//! Compare the production kernels to the original module on identical buffers.
//! Tail scores retain a sentinel so a partial final tile must mask every lane.
use super::*;
use crate::vulkan::context::{bytes_of_f32, Commands, Device, Push};
use ash::vk;

#[test]
#[ignore = "requires the hardware-validated RX 9070 XT Windows Vulkan driver"]
fn initialized_attention_matches_original_bits() {
    let lengths = [127, 128, 129, 255, 256, 513, 1024, 2048, 2051, 4096, 4099];
    compare_attention(4112, &lengths, 96);
}

#[test]
#[ignore = "requires the hardware-validated RX 9070 XT Windows Vulkan driver"]
fn full_context_attention_matches_original_bits() {
    compare_attention(40_960, &[40_959, 40_960], 256);
}

fn compare_attention(stride: usize, lengths: &[usize], budget_mib: usize) {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let device = Device::new(&DeviceOptions {
        direct_experts: Some(true),
        fast_math: true,
        ..options(budget_mib * 1024 * 1024)
    }).unwrap();
    let heads = 32;
    let kv_heads = 4;
    let width = 128;
    for op_code in [4, 5] {
        let (input_rows, result_rows, counter) = if op_code == 4 {
            (heads * width, heads * stride, "tiled_score_dispatches")
        } else {
            (heads * stride, heads * width, "long_attention_dispatches")
        };
        let input: Vec<_> = (0..input_rows).map(|i| ((i * 31 % 137) as f32 - 68.0) / 17.3317).collect();
        let cache: Vec<_> = (0..kv_heads * width * stride)
            .map(|i| ((i * 41 % 197) as f32 - 98.0) / 73.0331).collect();
        let dummy = device.storage(4).unwrap();
        let output = device.storage(result_rows * 4).unwrap();
        let status = device.storage(4).unwrap();
        let host = device.host(result_rows * 4 + 4).unwrap();
        let mut commands = Commands::new(Arc::clone(&device), 1).unwrap();
        let push = Push::new(op_code).n(0, 1).unwrap().n(1, heads).unwrap()
            .n(2, kv_heads).unwrap().n(3, width).unwrap().n(4, stride).unwrap();
        let groups = if op_code == 4 { [1, heads, 1] } else { [heads, 1, 1] };
        let op = commands.operation(push, [Arc::clone(&dummy), device.upload(bytes_of_f32(&input)).unwrap(),
            device.upload(bytes_of_f32(&cache)).unwrap(), Arc::clone(&output),
            Arc::clone(&dummy), dummy, Arc::clone(&status)], groups).unwrap();
        for &tokens in lengths {
            let push = op.push.clone().n(0, tokens).unwrap();
            let groups = if op_code == 4 { [tokens as u32, heads as u32, 1] } else { op.groups };
            let mut reference = Vec::new();
            let mut submit_us = [0u128; 2];
            for original in [true, false] {
                commands.begin().unwrap();
                unsafe {
                    device.raw.cmd_fill_buffer(commands.raw, status.raw, 0, 4, 0);
                    device.raw.cmd_fill_buffer(commands.raw, output.raw, 0, output.size, 0x7fc00000);
                }
                commands.transfer_to_compute();
                if original {
                    // Bypass production pipeline selection and keep the exact same
                    // ordinary module, allocated descriptors and push constants.
                    let raw_push = unsafe { std::slice::from_raw_parts(push.0.as_ptr().cast::<u8>(), 64) };
                    unsafe {
                        device.raw.cmd_bind_pipeline(commands.raw, vk::PipelineBindPoint::COMPUTE, op.pipeline);
                        device.raw.cmd_bind_descriptor_sets(commands.raw, vk::PipelineBindPoint::COMPUTE,
                            op.pipeline_layout, 0, &[op.set], &[]);
                        device.raw.cmd_push_constants(commands.raw, op.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE, 0, raw_push);
                        device.raw.cmd_dispatch(commands.raw, groups[0], groups[1], groups[2]);
                    }
                    commands.compute_barrier();
                } else {
                    commands.record(&op, &push, groups);
                }
                commands.compute_to_transfer();
                unsafe {
                    device.raw.cmd_copy_buffer(commands.raw, output.raw, host.raw, &[vk::BufferCopy {
                        src_offset: 0, dst_offset: 0, size: output.size,
                    }]);
                    device.raw.cmd_copy_buffer(commands.raw, status.raw, host.raw, &[vk::BufferCopy {
                        src_offset: 0, dst_offset: output.size, size: 4,
                    }]);
                }
                commands.transfer_to_host();
                let start = std::time::Instant::now();
                commands.submit().unwrap();
                submit_us[usize::from(!original)] = start.elapsed().as_micros();
                let mut result = vec![0; output.size as usize + 4];
                host.read(0, &mut result).unwrap();
                assert_eq!(&result[output.size as usize..], &[0; 4]);
                if original {
                    // Verify that the reference actually wrote every initialized
                    // score/component and left unused positions at the sentinel.
                    for (index, word) in result[..output.size as usize].chunks_exact(4).enumerate() {
                        let bits = u32::from_ne_bytes(word.try_into().unwrap());
                        if op_code == 4 && index % stride >= tokens {
                            assert_eq!(bits, 0x7fc00000);
                        } else {
                            assert!(f32::from_bits(bits).is_finite());
                        }
                    }
                    reference = result;
                } else {
                    let different = reference.chunks_exact(4).zip(result.chunks_exact(4))
                        .filter(|(expected, actual)| expected != actual).count();
                    assert_eq!(different, 0, "op {op_code} at {tokens} occupied tokens");
                }
            }
            eprintln!("ATTENTION_POINTWISE {}", serde_json::json!({
                "op": op_code, "initialized_tokens": tokens,
                "bit_exact_outputs": if op_code == 4 { heads * tokens } else { heads * width },
                "untouched_tail": if op_code == 4 { heads * (stride - tokens) } else { 0 },
                "reference_submit_and_copy_us": submit_us[0],
                "candidate_submit_and_copy_us": submit_us[1],
            }));
        }
        let selected_lengths = lengths.iter().filter(|&&n| n >= 128).count();
        assert_eq!(device.stats()[counter], selected_lengths);
    }
}
