//! Explicit numerical localization, not a relaxed inference acceptance gate.
//! Runs immutable legacy/array binaries against the same allocations and inputs.
use super::context::{bytes_of_f32, Buffer, Commands, Device, Push};
use super::DeviceOptions;
use ash::vk;
use half::f16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{ffi::CString, sync::Arc};

struct Harness {
    device: Arc<Device>,
    commands: Commands,
    sets: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    old: vk::Pipeline,
    array: vk::Pipeline,
    variants: Vec<(String, vk::Pipeline)>,
}
struct Op {
    set: vk::DescriptorSet,
    push: Push,
    groups: [u32; 3],
    _buffers: Vec<Arc<Buffer>>,
}
impl Harness {
    fn new() -> Self {
        let software = std::env::var("NEXTORA_NUMERICS_SOFTWARE").as_deref() == Ok("1");
        let device = Device::new(&DeviceOptions {
            fast_math: true,
            direct_experts: Some(true),
            allow_software: software,
            memory_budget_bytes: 256 * 1024 * 1024,
            ..Default::default()
        })
        .unwrap();
        let commands = Commands::new(Arc::clone(&device), 1).unwrap();
        let bindings: Vec<_> = (0..7)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(if i == 0 { 8 } else { 1 })
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            device.raw.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .unwrap();
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(64)];
        let layouts = [set_layout];
        let layout = unsafe {
            device.raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        }
        .unwrap();
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(128 * 14)];
        let sets = unsafe {
            device.raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(128)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .unwrap();
        let pipeline = |binary: &[u8]| {
            let words: Vec<_> = binary
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let module = unsafe {
                device
                    .raw
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            }
            .unwrap();
            let main = CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&main);
            let result = unsafe {
                device.raw.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
            }
            .unwrap()[0];
            unsafe { device.raw.destroy_shader_module(module, None) };
            result
        };
        let old = pipeline(include_bytes!("ops_fast.spv"));
        let array = pipeline(include_bytes!("ops_fast_direct.spv"));
        let variants = std::env::var_os("NEXTORA_NUMERICS_VARIANTS")
            .map(|path| {
                let specs: Vec<Value> =
                    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                specs
                    .into_iter()
                    .map(|spec| {
                        let name = spec["name"].as_str().unwrap().to_owned();
                        let bytes = std::fs::read(spec["path"].as_str().unwrap()).unwrap();
                        (name, pipeline(&bytes))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            device,
            commands,
            sets,
            set_layout,
            layout,
            old,
            array,
            variants,
        }
    }
    fn op(
        &self,
        mut push: Push,
        weights: &[Arc<Buffer>],
        others: [Arc<Buffer>; 6],
        groups: [u32; 3],
    ) -> Op {
        assert!(!weights.is_empty() && weights.len() <= 8);
        push.0[14] = groups[0];
        let layouts = [self.set_layout];
        let set = unsafe {
            self.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.sets)
                    .set_layouts(&layouts),
            )
        }
        .unwrap()[0];
        let mut infos: Vec<Vec<vk::DescriptorBufferInfo>> = vec![(0..8)
            .map(|i| {
                let b = &weights[i.min(weights.len() - 1)];
                vk::DescriptorBufferInfo::default()
                    .buffer(b.raw)
                    .offset(0)
                    .range(b.size)
            })
            .collect()];
        infos.extend(others.iter().map(|b| {
            vec![vk::DescriptorBufferInfo::default()
                .buffer(b.raw)
                .range(b.size)]
        }));
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { self.device.raw.update_descriptor_sets(&writes, &[]) };
        Op {
            set,
            push,
            groups,
            _buffers: weights.iter().chain(&others).cloned().collect(),
        }
    }
    fn execute(
        &mut self,
        pipeline: vk::Pipeline,
        ops: &[&Op],
        outputs: &[Arc<Buffer>],
        status: &Arc<Buffer>,
    ) -> Vec<Vec<u8>> {
        let bytes = outputs.iter().map(|b| b.size as usize).sum::<usize>();
        let host = self.device.host(bytes + 4).unwrap();
        self.commands.begin().unwrap();
        unsafe {
            self.device
                .raw
                .cmd_fill_buffer(self.commands.raw, status.raw, 0, 4, 0)
        };
        self.commands.transfer_to_compute();
        for op in ops {
            let constants =
                unsafe { std::slice::from_raw_parts(op.push.0.as_ptr().cast::<u8>(), 64) };
            unsafe {
                self.device.raw.cmd_bind_pipeline(
                    self.commands.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline,
                );
                self.device.raw.cmd_bind_descriptor_sets(
                    self.commands.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[op.set],
                    &[],
                );
                self.device.raw.cmd_push_constants(
                    self.commands.raw,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    constants,
                );
                self.device.raw.cmd_dispatch(
                    self.commands.raw,
                    op.groups[0],
                    op.groups[1],
                    op.groups[2],
                );
            }
            self.commands.compute_barrier();
        }
        self.commands.compute_to_transfer();
        let mut offset = 0u64;
        for buffer in outputs.iter().chain(std::iter::once(status)) {
            unsafe {
                self.device.raw.cmd_copy_buffer(
                    self.commands.raw,
                    buffer.raw,
                    host.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset,
                        size: buffer.size,
                    }],
                )
            };
            offset += buffer.size;
        }
        self.commands.transfer_to_host();
        self.commands.submit().unwrap();
        let mut all = vec![0u8; bytes + 4];
        host.read(0, &mut all).unwrap();
        assert_eq!(&all[bytes..], &[0; 4], "shader generated invalid values");
        let mut offset = 0;
        outputs
            .iter()
            .map(|buffer| {
                let end = offset + buffer.size as usize;
                let result = all[offset..end].to_vec();
                offset = end;
                result
            })
            .collect()
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        unsafe {
            for (_, pipeline) in &self.variants {
                self.device.raw.destroy_pipeline(*pipeline, None);
            }
            self.device.raw.destroy_pipeline(self.old, None);
            self.device.raw.destroy_pipeline(self.array, None);
            self.device.raw.destroy_descriptor_pool(self.sets, None);
            self.device.raw.destroy_pipeline_layout(self.layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn comparison(a: &[u8], b: &[u8], floating: bool) -> Value {
    assert_eq!(a.len(), b.len());
    let aw: Vec<_> = a
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let bw: Vec<_> = b
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let changed = aw.iter().zip(&bw).filter(|(x, y)| x != y).count();
    let first = aw.iter().zip(&bw).position(|(x, y)| x != y);
    let mut result = json!({"words":aw.len(),"changed_words":changed,"first_changed_index":first,"bitwise_equal":changed==0,
        "legacy_sha256":format!("{:x}",Sha256::digest(a)),"candidate_sha256":format!("{:x}",Sha256::digest(b))});
    if floating {
        let mut max_abs = 0.0f64;
        let mut scaled = 0.0f64;
        let mut square = 0.0f64;
        let mut ulp = 0u32;
        for (&x, &y) in aw.iter().zip(&bw) {
            let x = f32::from_bits(x);
            let y = f32::from_bits(y);
            assert!(x.is_finite() && y.is_finite());
            let delta = (f64::from(x) - f64::from(y)).abs();
            max_abs = max_abs.max(delta);
            scaled = scaled.max(delta / (1.0 + f64::from(x).abs()));
            square += delta * delta;
            let ordered = |f: f32| {
                let b = f.to_bits();
                if b >> 31 != 0 {
                    !b
                } else {
                    b ^ 0x80000000
                }
            };
            ulp = ulp.max(ordered(x).abs_diff(ordered(y)));
        }
        result["max_abs"] = json!(max_abs);
        result["max_scaled"] = json!(scaled);
        result["rms"] = json!((square / aw.len() as f64).sqrt());
        result["max_ulp"] = json!(ulp);
        if let Some(i) = first {
            result["first_legacy"] = json!(f32::from_bits(aw[i]));
            result["first_candidate"] = json!(f32::from_bits(bw[i]));
        }
    } else if let Some(i) = first {
        result["first_legacy_bits"] = json!(aw[i]);
        result["first_candidate_bits"] = json!(bw[i]);
    }
    result
}
fn input(width: usize, salt: usize) -> Vec<f32> {
    (0..width)
        .map(|i| ((i * 73 + salt * 31) % 997) as f32 / 251.731 - 1.937)
        .collect()
}
fn matrix(ty: u32, width: usize, rows: usize, salt: usize) -> Vec<u8> {
    let block_bytes = if ty == 12 { 144 } else { 210 };
    let mut out = Vec::with_capacity(width / 256 * block_bytes * rows);
    for row in 0..rows {
        for block in 0..width / 256 {
            let start = out.len();
            out.extend(
                (0..block_bytes)
                    .map(|i| (i * 37 + i / 3 + row * 17 + block * 53 + salt * 71) as u8),
            );
            let scale = f16::from_f32(
                0.000137319 + ((row * 7 + block * 11 + salt * 13) % 19) as f32 * 0.000019733,
            );
            if ty == 12 {
                out[start..start + 2].copy_from_slice(&scale.to_le_bytes());
                let min =
                    f16::from_f32(0.00029317 + ((row + block + salt) % 11) as f32 * 0.000023479);
                out[start + 2..start + 4].copy_from_slice(&min.to_le_bytes());
            } else {
                out[start + 208..start + 210].copy_from_slice(&scale.to_le_bytes());
            }
        }
    }
    out
}

#[test]
#[ignore = "diagnostic only; requires explicit idle Vulkan environment and no downloaded weights"]
fn diagnostic_legacy_array_math_pointwise() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let mut h = Harness::new();
    let device = Arc::clone(&h.device);
    let dummy = device.storage(4).unwrap();
    let status = device.storage(4).unwrap();
    let mut reports = Vec::new();
    for (width, heads) in [(2048usize, 1usize), (128, 32)] {
        let values = input(width * heads, 9);
        let weights: Vec<_> = (0..width)
            .map(|i| 0.83719 + (i % 29) as f32 * 0.013719)
            .collect();
        let source = device.upload(bytes_of_f32(&values)).unwrap();
        let weight = device.upload(bytes_of_f32(&weights)).unwrap();
        let output = device.storage(values.len() * 4).unwrap();
        let op = h.op(
            Push::new(1)
                .n(0, width)
                .unwrap()
                .n(1, heads)
                .unwrap()
                .f(0, 1e-6),
            &[weight],
            [
                source,
                Arc::clone(&dummy),
                Arc::clone(&output),
                Arc::clone(&dummy),
                Arc::clone(&dummy),
                Arc::clone(&status),
            ],
            [heads as u32, 1, 1],
        );
        let old = h.execute(h.old, &[&op], &[Arc::clone(&output)], &status);
        let array = h.execute(h.array, &[&op], &[output], &status);
        reports.push(json!({"operation":"rms","width":width,"heads":heads,"comparison":comparison(&old[0],&array[0],true)}));
    }
    for position in [0usize, 1, 21, 127, 1023, 40959] {
        let (width, heads) = (128usize, 32usize);
        let values = input(width * heads, position + 1);
        let source = device.upload(bytes_of_f32(&values)).unwrap();
        let output = device.storage(values.len() * 4).unwrap();
        let op = h.op(
            Push::new(2)
                .n(0, width)
                .unwrap()
                .n(1, heads)
                .unwrap()
                .n(2, position)
                .unwrap()
                .n(3, width)
                .unwrap()
                .f(0, 1_000_000.0),
            &[Arc::clone(&dummy)],
            [
                source,
                Arc::clone(&dummy),
                Arc::clone(&output),
                Arc::clone(&dummy),
                Arc::clone(&dummy),
                Arc::clone(&status),
            ],
            [(heads * width / 2).div_ceil(128) as u32, 1, 1],
        );
        let old = h.execute(h.old, &[&op], &[Arc::clone(&output)], &status);
        let array = h.execute(h.array, &[&op], &[output], &status);
        reports.push(json!({"operation":"rope_gpu_trig","width":width,"heads":heads,"position":position,"comparison":comparison(&old[0],&array[0],true)}));
    }
    let matrix_shapes: &[(usize, usize)] =
        if std::env::var("NEXTORA_NUMERICS_SMALL").as_deref() == Ok("1") {
            &[(256, 7)]
        } else {
            &[(2048, 768), (768, 2048)]
        };
    for &(width, rows) in matrix_shapes {
        let slots = 8usize;
        let total = width * slots;
        for ties in [false, true] {
            let values = if !ties {
                input(total, 17)
            } else {
                (0..total)
                    .map(|i| {
                        let amplitude = 1.003179 + (i / 32 % 31) as f32 * 0.000317;
                        if i % 32 == 0 {
                            amplitude
                        } else {
                            amplitude * (((i * 7 % 253) as f32 - 126.0) + 0.5) / 127.0
                        }
                    })
                    .collect()
            };
            let source = device.upload(bytes_of_f32(&values)).unwrap();
            let packed = device.storage(total).unwrap();
            let scales = device.storage(total / 32 * 4).unwrap();
            let quant = h.op(
                Push::new(16).n(0, total).unwrap(),
                &[Arc::clone(&dummy)],
                [
                    Arc::clone(&source),
                    Arc::clone(&packed),
                    Arc::clone(&dummy),
                    Arc::clone(&scales),
                    Arc::clone(&dummy),
                    Arc::clone(&status),
                ],
                [(total / 32) as u32, 1, 1],
            );
            let old = h.execute(
                h.old,
                &[&quant],
                &[Arc::clone(&packed), Arc::clone(&scales)],
                &status,
            );
            let array = h.execute(
                h.array,
                &[&quant],
                &[Arc::clone(&packed), Arc::clone(&scales)],
                &status,
            );
            reports.push(json!({"operation":"q8_activation","width":width,"slots":slots,"near_half_ties":ties,
                "packed_comparison":comparison(&old[0],&array[0],false),"scale_comparison":comparison(&old[1],&array[1],true)}));
            if ties {
                continue;
            }
            // Reuse legacy-generated packed inputs/scales for every GEMV so
            // activation rounding cannot disguise weight/math differences.
            let legacy_packed = device.upload(&old[0]).unwrap();
            let legacy_scales = device.upload(&old[1]).unwrap();
            let selection: Vec<f32> = (0..slots)
                .flat_map(|slot| [slot as f32, 1.0 / slots as f32])
                .collect();
            let selection = device.upload(bytes_of_f32(&selection)).unwrap();
            for ty in [12u32, 14u32] {
                let sources: Vec<_> = (0..slots)
                    .map(|slot| matrix(ty, width, rows, slot + 31))
                    .collect();
                let all = sources.concat();
                let packed_weight = device.upload(&all).unwrap();
                let separate: Vec<_> = sources
                    .iter()
                    .map(|bytes| device.upload(bytes).unwrap())
                    .collect();
                let output = device.storage(rows * slots * 4).unwrap();
                for fast in [false, true] {
                    let push = Push::new(0)
                        .n(0, width)
                        .unwrap()
                        .n(1, rows)
                        .unwrap()
                        .n(2, ty as usize)
                        .unwrap()
                        .n(3, width / 256 * if ty == 12 { 144 } else { 210 })
                        .unwrap()
                        .n(4, slots)
                        .unwrap()
                        .flags(3 | if fast { 32 } else { 0 });
                    let bindings = || {
                        [
                            Arc::clone(&source),
                            Arc::clone(&legacy_packed),
                            Arc::clone(&output),
                            Arc::clone(&legacy_scales),
                            Arc::clone(&selection),
                            Arc::clone(&status),
                        ]
                    };
                    let packed_op = h.op(
                        push.clone(),
                        &[Arc::clone(&packed_weight)],
                        bindings(),
                        [rows as u32, 1, slots as u32],
                    );
                    let direct_op = h.op(
                        push.clone().flags(push.0[15] | 512),
                        &separate,
                        bindings(),
                        [rows as u32, 1, slots as u32],
                    );
                    let old = h.execute(h.old, &[&packed_op], &[Arc::clone(&output)], &status);
                    let array = h.execute(h.array, &[&packed_op], &[Arc::clone(&output)], &status);
                    let direct = h.execute(h.array, &[&direct_op], &[Arc::clone(&output)], &status);
                    let values = floats(&old[0]);
                    assert!(values.iter().all(|v| v.is_finite()));
                    let mut variant_results = Vec::new();
                    if ty == 14 && fast {
                        for (name, pipeline) in h.variants.clone() {
                            let packed =
                                h.execute(pipeline, &[&packed_op], &[Arc::clone(&output)], &status);
                            let direct =
                                h.execute(pipeline, &[&direct_op], &[Arc::clone(&output)], &status);
                            variant_results.push(json!({"name":name,
                                "old_vs_variant_packed":comparison(&old[0],&packed[0],true),
                                "old_vs_variant_direct":comparison(&old[0],&direct[0],true),
                                "variant_packed_vs_variant_direct":comparison(&packed[0],&direct[0],true)}));
                        }
                    }
                    reports.push(json!({"operation":"gemv","type":ty,"fast_q8":fast,"width":width,"rows":rows,"slots":slots,"variants":variant_results,
                        "old_vs_array_packed":comparison(&old[0],&array[0],true),"array_packed_vs_array_direct":comparison(&array[0],&direct[0],true)}));
                }
            }
        }
    }
    let report = json!({"diagnostic":"legacy_array_same_input_pointwise","adapter":device.info(),
        "old_fast_spv_sha256":format!("{:x}",Sha256::digest(include_bytes!("ops_fast.spv"))),
        "array_fast_spv_sha256":format!("{:x}",Sha256::digest(include_bytes!("ops_fast_direct.spv"))),"cases":reports,
        "scope":"Localization only: exact bits/ULPs reported; no numerical tolerance or inference acceptance gate is changed."});
    if let Some(path) = std::env::var_os("NEXTORA_NUMERICS_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    eprintln!(
        "NUMERICAL_POINTWISE {}",
        serde_json::to_string(&report).unwrap()
    );
}
