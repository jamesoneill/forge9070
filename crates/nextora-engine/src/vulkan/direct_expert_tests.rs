//! Opt-in descriptor-array validation. These tests deliberately use independent
//! source buffers and update an already-submitted graph with different leases.
use super::*;
use super::super::context::{bytes_of_f32, bytes_of_f32_mut, Commands, Device, Push};
use ash::vk;

fn quantized(input: &[f32]) -> Vec<f32> {
    input.chunks(32).flat_map(|group| {
        let scale = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 127.0;
        group.iter().map(move |&x| if scale == 0.0 { 0.0 } else {
            (x / scale).round_ties_even().clamp(-127.0,127.0) * scale
        })
    }).collect()
}

#[test]
#[ignore = "requires explicit Vulkan direct-expert execution environment"]
fn software_vulkan_direct_expert_arrays_all_formats_and_rebinding() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    for fast in [false, true] {
        let device = Device::new(&DeviceOptions {
            fast_math: fast,
            direct_experts: Some(true),
            ..options(64 * 1024 * 1024)
        }).unwrap();
        assert!(device.direct_experts, "test requires descriptor-array support");
        for id in [0,1,2,8,11,12,13,14,18,20,21,22,23,30,42] {
            let ty=QuantType::try_from(id).unwrap();
            let shapes: &[(usize,usize)] = if matches!(ty,QuantType::Q4_K|QuantType::Q6_K) { &[(256,7),(2048,768),(768,2048)] } else { &[(256,7)] };
            for &(width,rows) in shapes {
            let (slots,offset)=(8usize,32usize);
            let q8=fast && matches!(ty,QuantType::Q4_K|QuantType::Q6_K);
            let input:Vec<f32>=(0..width*slots).map(|i| ((i*43%211) as f32-105.0)/107.0).collect();
            let mut prefixed=vec![0.25;offset];prefixed.extend_from_slice(&input);
            let source=device.upload(bytes_of_f32(&prefixed)).unwrap();
            let dummy=device.storage(4).unwrap();
            let packed=device.storage((width*slots+offset)*4).unwrap();
            let scales=device.storage((width*slots+offset)/32*4).unwrap();
            let output=device.storage(rows*slots*4).unwrap();
            let status=device.storage(4).unwrap();
            let host=device.host(rows*slots*4+4).unwrap();
            let mut commands=Commands::new(Arc::clone(&device),2).unwrap();
            let mut qp=Push::new(16).n(0,width*slots).unwrap();
            qp.0[10]=offset as u32;qp.0[11]=(offset/4) as u32;qp.0[13]=(offset/32) as u32;
            let quant_groups=commands.groups((width*slots).div_ceil(32)).unwrap();
            let quant=commands.operation(qp,[Arc::clone(&dummy),Arc::clone(&source),Arc::clone(&packed),Arc::clone(&dummy),Arc::clone(&scales),Arc::clone(&dummy),Arc::clone(&status)],quant_groups).unwrap();
            let mut gp=Push::new(0).n(0,width).unwrap().n(1,rows).unwrap().n(2,id as usize).unwrap().n(3,ty.row_bytes(width).unwrap()).unwrap().n(4,slots).unwrap().flags(512|2|if q8 {32} else {0});
            gp.0[11]=offset as u32;
            let gemv=commands.operation(gp,[Arc::clone(&dummy),Arc::clone(&source),Arc::clone(&packed),Arc::clone(&output),Arc::clone(&scales),Arc::clone(&dummy),Arc::clone(&status)],[rows,1,slots]).unwrap();
            assert!(!quant.array_weights);
            assert_eq!(quant.pipeline,device.pipeline,"Q8 quantizer must retain the legacy module");
            assert_eq!(quant.pipeline_layout,device.pipeline_layout);
            assert!(gemv.array_weights);
            assert_ne!(gemv.pipeline,device.pipeline);
            assert_ne!(gemv.pipeline_layout,device.pipeline_layout);
            for generation in 0..2 {
                // Rebind the same graph after its fence, and discard all old
                // source leases. No graph Arc may pin historical expert data.
                let files:Vec<_>=(0..slots).map(|slot|matrix_file(ty,width,rows,slot+generation*23+71)).collect();
                let models:Vec<_>=files.iter().map(|f|Gguf::open(&f.0).unwrap()).collect();
                let weights:Vec<_>=models.iter().map(|m|device.upload(m.tensor("matrix").unwrap().bytes()).unwrap()).collect();
                let word_dispatches = device.stats()["expert_word_dispatches"].as_u64().unwrap();
                commands.begin().unwrap();
                commands.update_weight_slots(gemv.set,&weights).unwrap();
                unsafe { device.raw.cmd_fill_buffer(commands.raw,status.raw,0,4,0); }
                commands.transfer_to_compute();
                if q8 {commands.record(&quant,&quant.push,quant.groups);}
                commands.record(&gemv,&gemv.push,gemv.groups);
                commands.compute_to_transfer();
                for (buffer,size,dst) in [(&output,rows*slots*4,0),(&status,4,rows*slots*4)] {
                    unsafe {device.raw.cmd_copy_buffer(commands.raw,buffer.raw,host.raw,&[vk::BufferCopy{src_offset:0,dst_offset:dst as u64,size:size as u64}]);}
                }
                commands.transfer_to_host();commands.submit().unwrap();
                assert_eq!(
                    device.stats()["expert_word_dispatches"].as_u64().unwrap() - word_dispatches,
                    u64::from(q8),
                    "select the packed-word shader for Direct Q4_K/Q6_K with Q8 activations"
                );
                let mut actual=vec![0.0;rows*slots];host.read(0,bytes_of_f32_mut(&mut actual)).unwrap();
                let mut bad=[0u8;4];host.read(rows*slots*4,&mut bad).unwrap();assert_eq!(bad,[0;4]);
                for slot in 0..slots {
                    let original=&input[slot*width..(slot+1)*width];
                    let dequantized=quantized(original);
                    let expected=models[slot].tensor("matrix").unwrap().matvec(if q8 {&dequantized}else{original}).unwrap();
                    close(&actual[slot*rows..(slot+1)*rows],&expected,&format!("direct type={id} fast={fast} slot={slot} rebind={generation}"));
                }
                for weight in &weights {assert_eq!(Arc::strong_count(weight),1,"graph must not retain source weights");}
            }
            }
        }
    }
}
