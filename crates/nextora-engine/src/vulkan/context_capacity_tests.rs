//! Explicit hardware capacity proof; never part of routine test execution.
use super::super::context::{Commands, Device, DeviceOptions, KvPrecision};
use ash::vk;
use std::sync::Arc;

#[test]
#[ignore = "requires an idle hardware GPU and explicit 512K allocation permission"]
fn hardware_512k_full_qwen3_kv_capacity() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    assert_eq!(std::env::var("NEXTORA_GPU_CAPACITY_TEST").as_deref(), Ok("1"));
    let budget = 14800usize * 1024 * 1024;
    let device = Device::new(&DeviceOptions {
        device_index: Some(0), allow_software: false, fast_math: true,
        kv_precision: KvPrecision::Q4_64, memory_budget_bytes: budget,
        ..DeviceOptions::default()
    }).unwrap();
    assert_eq!(device.info()["software"], false);
    let row_bytes = KvPrecision::Q4_64.row_bytes(512).unwrap();
    let per_cache = row_bytes * 524288;
    let mut buffers = Vec::new();
    for _ in 0..48*2 { buffers.push(device.storage(per_cache).unwrap()); }
    let dense = device.storage(997_554_176).unwrap();
    let scratch = device.storage(96*1024*1024).unwrap();
    let growth = device.storage(row_bytes*262144*2).unwrap();
    let readback = device.host(4).unwrap();
    let mut commands = Commands::new(Arc::clone(&device), 1).unwrap();
    commands.begin().unwrap();
    unsafe {
        for buffer in &buffers { device.raw.cmd_fill_buffer(commands.raw, buffer.raw, 0, vk::WHOLE_SIZE, 0x11223344); }
        for buffer in [&dense, &scratch, &growth] { device.raw.cmd_fill_buffer(commands.raw, buffer.raw, 0, vk::WHOLE_SIZE, 0); }
        device.raw.cmd_pipeline_barrier(commands.raw, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ)], &[], &[]);
        device.raw.cmd_copy_buffer(commands.raw, buffers.last().unwrap().raw, readback.raw, &[vk::BufferCopy {src_offset:(per_cache-4) as u64,dst_offset:0,size:4}]);
    }
    commands.transfer_to_host(); commands.submit().unwrap();
    let mut bytes=[0u8;4];readback.read(0, &mut bytes).unwrap();
    assert_eq!(u32::from_ne_bytes(bytes),0x11223344);
    assert!(device.reserved() <= budget as u64);
    println!("NEXTORA_512K_CAPACITY {}", serde_json::json!({"adapter":device.info()["name"],"tokens":524288,"layers":48,"kv_heads":4,"head_dim":128,"precision":"Q4_64","kv_payload_bytes":per_cache*96,"dense_bytes":dense.size,"working_allowance_bytes":scratch.size,"growth_overlap_bytes":growth.size,"live_bytes":device.reserved(),"budget_bytes":budget,"last_layer_last_word_verified":true}));
}
