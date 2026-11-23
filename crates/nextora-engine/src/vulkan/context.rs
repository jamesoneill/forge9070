//! Vulkan ownership, bounded allocation and reusable compute submissions.
use anyhow::{Context as _, Result, bail, ensure};
use ash::{Entry, Instance, vk};
use serde_json::{Value, json};
use std::{
    ffi::{CStr, CString},
    ptr::NonNull,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

const BINDINGS: u32 = 7;
pub const WORKGROUP: usize = 128;

#[derive(Clone, Debug)]
pub struct DeviceOptions {
    pub device_index: Option<usize>,
    /// Explicitly enable a CPU/software Vulkan implementation for shader tests.
    /// Production discovery never accepts one by default.
    pub allow_software: bool,
    /// Maximum total live Vulkan allocation bytes owned by this engine. Zero
    /// selects 70% of the largest usable device-local heap; this is not free VRAM.
    pub memory_budget_bytes: usize,
}
impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            device_index: None,
            allow_software: false,
            memory_budget_bytes: 0,
        }
    }
}

#[derive(Debug)]
pub struct BudgetExceeded(pub String);
impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BudgetExceeded {}

fn software(p: &vk::PhysicalDeviceProperties) -> bool {
    let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
        .to_string_lossy()
        .to_lowercase();
    p.device_type == vk::PhysicalDeviceType::CPU
        || name.contains("llvmpipe")
        || name.contains("lavapipe")
        || name.contains("swiftshader")
}
fn api(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}
fn open_instance() -> Result<(Entry, Instance)> {
    // SAFETY: the loaded Entry owns the dynamic loader for the Instance lifetime.
    let entry = unsafe { Entry::load() }
        .context("load Vulkan loader; install the platform Vulkan driver")?;
    let supported = unsafe { entry.try_enumerate_instance_version() }
        .context("query Vulkan loader version")?
        .unwrap_or(vk::API_VERSION_1_0);
    ensure!(
        supported >= vk::API_VERSION_1_1,
        "independent Vulkan engine requires Vulkan 1.1 or later"
    );
    let name = CString::new("Forge independent inference").unwrap();
    let app = vk::ApplicationInfo::default()
        .application_name(&name)
        .engine_name(&name)
        .api_version(vk::API_VERSION_1_1);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }
    .context("create Vulkan instance")?;
    Ok((entry, instance))
}
/// A discovery snapshot only: buffer memory-type compatibility and successful
/// pipeline/model creation are checked later, when those resources exist.
struct Adapter {
    properties: vk::PhysicalDeviceProperties,
    memory: vk::PhysicalDeviceMemoryProperties,
    compute_queue: Option<u32>,
}

impl Adapter {
    fn query(instance: &Instance, physical: vk::PhysicalDevice) -> Self {
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let compute_queue = queues
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .min_by_key(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(i, _)| i as u32);
        Self {
            properties,
            memory,
            compute_queue,
        }
    }

    fn name(&self) -> String {
        unsafe { CStr::from_ptr(self.properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn largest_usable_heap(&self) -> u64 {
        self.memory.memory_heaps[..self.memory.memory_heap_count as usize]
            .iter()
            .enumerate()
            .filter(|(index, heap)| {
                heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                    && self.memory.memory_types[..self.memory.memory_type_count as usize]
                        .iter()
                        .any(|ty| {
                            ty.heap_index as usize == *index
                                && ty
                                    .property_flags
                                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        })
            })
            .map(|(_, heap)| heap.size)
            .max()
            .unwrap_or(0)
    }

    fn has_staging_memory(&self) -> bool {
        self.memory.memory_types[..self.memory.memory_type_count as usize]
            .iter()
            .any(|ty| {
                ty.property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && ty.heap_index < self.memory.memory_heap_count
                    && self.memory.memory_heaps[ty.heap_index as usize].size > 0
            })
    }

    fn budget(&self, options: &DeviceOptions) -> u64 {
        if options.memory_budget_bytes == 0 {
            self.largest_usable_heap().saturating_mul(7) / 10
        } else {
            options.memory_budget_bytes as u64
        }
    }

    fn rejection_reasons(&self, options: &DeviceOptions) -> Vec<String> {
        let p = &self.properties;
        let mut reasons = Vec::new();
        if software(p) && !options.allow_software {
            reasons.push(
                "CPU/software Vulkan adapter requires the explicit test-only allow-software option"
                    .into(),
            );
        }
        if p.api_version < vk::API_VERSION_1_1 {
            reasons.push("Vulkan 1.1 or later is required".into());
        }
        if self.compute_queue.is_none() {
            reasons.push("adapter has no compute queue".into());
        }
        if p.limits.max_compute_work_group_invocations < WORKGROUP as u32
            || p.limits.max_compute_work_group_size[0] < WORKGROUP as u32
        {
            reasons.push("adapter cannot execute 128-lane workgroups".into());
        }
        if p.limits.max_compute_shared_memory_size < (WORKGROUP * 4) as u32 {
            reasons.push("adapter needs at least 512 bytes of compute shared memory".into());
        }
        if p.limits.max_per_stage_descriptor_storage_buffers < BINDINGS
            || p.limits.max_descriptor_set_storage_buffers < BINDINGS
        {
            reasons.push("adapter needs at least seven compute storage-buffer bindings".into());
        }
        if p.limits.max_push_constants_size < 64 {
            reasons.push("adapter needs at least 64 bytes of push constants".into());
        }
        let heap = self.largest_usable_heap();
        if heap == 0 {
            reasons.push("adapter has no usable device-local memory heap".into());
        }
        if !self.has_staging_memory() {
            reasons.push("adapter has no host-visible memory for staging transfers".into());
        }
        let budget = self.budget(options);
        if budget < 1024 * 1024 || budget > heap {
            reasons.push(format!("Vulkan allocation budget must be between 1 MiB and the largest usable device-local heap ({heap} bytes); requested {budget} bytes"));
        }
        reasons
    }

    fn rank(&self, index: usize) -> (u8, std::cmp::Reverse<u64>, usize) {
        let kind = if software(&self.properties) {
            4
        } else {
            match self.properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                _ => 3,
            }
        };
        // Heap capacity is a deterministic tie-break, not a free-memory or
        // throughput measurement. Neither the vendor ID nor model name ranks a GPU.
        (kind, std::cmp::Reverse(self.largest_usable_heap()), index)
    }
}

fn choose_adapter(adapters: &[Adapter], options: &DeviceOptions) -> Result<Option<usize>> {
    ensure!(
        options.memory_budget_bytes == 0 || options.memory_budget_bytes >= 1024 * 1024,
        "explicit Vulkan allocation budget must be at least 1 MiB"
    );
    if let Some(index) = options.device_index {
        let adapter = adapters.get(index).with_context(|| {
            format!(
                "Vulkan adapter index {index} is outside {} adapters",
                adapters.len()
            )
        })?;
        let reasons = adapter.rejection_reasons(options);
        ensure!(
            reasons.is_empty(),
            "selected Vulkan adapter {index} ({}) is incompatible: {}",
            adapter.name(),
            reasons.join("; ")
        );
        return Ok(Some(index));
    }
    let selected = adapters
        .iter()
        .enumerate()
        .filter(|(_, adapter)| adapter.rejection_reasons(options).is_empty())
        .min_by_key(|(index, adapter)| adapter.rank(*index))
        .map(|(index, _)| index);
    ensure!(
        selected.is_some() || options.memory_budget_bytes == 0,
        "no compatible Vulkan adapter satisfies the explicit allocation budget of {} bytes",
        options.memory_budget_bytes
    );
    Ok(selected)
}

/// Read-only device selection for the requested policy. This creates no logical
/// device, shader, buffers or model. `None` means no permitted adapter satisfies
/// discovery requirements; an explicit invalid device or allocation cap is an
/// error. Loader/discovery errors are returned to the caller without fallback.
pub fn select_device(options: &DeviceOptions) -> Result<Option<usize>> {
    let (_entry, instance) = open_instance()?;
    let result = (|| {
        let devices = unsafe { instance.enumerate_physical_devices() }
            .context("enumerate Vulkan adapters")?;
        let adapters = devices
            .into_iter()
            .map(|p| Adapter::query(&instance, p))
            .collect::<Vec<_>>();
        choose_adapter(&adapters, options)
    })();
    unsafe { instance.destroy_instance(None) };
    result
}

fn describe(adapter: &Adapter, index: usize, options: &DeviceOptions) -> Value {
    let p = &adapter.properties;
    let memory = &adapter.memory;
    let reasons = adapter.rejection_reasons(options);
    json!({"index":index,"name":unsafe{CStr::from_ptr(p.device_name.as_ptr())}.to_string_lossy(),
        "vendor_id":p.vendor_id,"device_id":p.device_id,"type":format!("{:?}",p.device_type),
        "vulkan_version":api(p.api_version),"software":software(p),"compute_queue":adapter.compute_queue,
        "device_local_heaps_bytes":memory.memory_heaps[..memory.memory_heap_count as usize].iter()
            .filter(|h|h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|h|h.size).collect::<Vec<_>>(),
        "largest_usable_device_local_heap_bytes":adapter.largest_usable_heap(),
        "host_visible_staging_memory":adapter.has_staging_memory(),
        "allocation_budget_bytes":adapter.budget(options),"allocation_budget_is_free_memory":false,
        "max_storage_buffer_bytes":p.limits.max_storage_buffer_range,
        "max_workgroup_invocations":p.limits.max_compute_work_group_invocations,
        "compatible":reasons.is_empty(),"rejection_reasons":reasons,
        "compatibility_scope":"discovery_requirements_only; model_allocation_and_compute_are_not_verified",
        "discovery_only":true,"compute_verified":false})
}
pub fn discover() -> Result<Value> {
    let (_entry, instance) = open_instance()?;
    let result = (|| {
        let devices = unsafe { instance.enumerate_physical_devices() }
            .context("enumerate Vulkan adapters")?;
        Ok(
            json!({"engine":"nextora-native-vulkan","adapters":devices.into_iter().enumerate().map(|(i,p)|describe(&Adapter::query(&instance,p),i,&DeviceOptions::default())).collect::<Vec<_>>(),
            "selection_policy":"capability_filtered_device_type_then_heap_capacity_then_index",
            "software_allowed_by_default":false,"compute_submitted":false,"hardware_performance_verified":false}),
        )
    })();
    unsafe { instance.destroy_instance(None) };
    result
}

pub(crate) struct Device {
    _entry: Entry,
    instance: Instance,
    pub raw: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub queue_lock: Mutex<()>,
    pub properties: vk::PhysicalDeviceProperties,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    info: Value,
    pub budget: u64,
    allocated: AtomicU64,
    high_water: AtomicU64,
    allocations: AtomicUsize,
    pub uploads: AtomicU64,
    pub readbacks: AtomicU64,
    pub submissions: AtomicU64,
    pub dispatches: AtomicU64,
}
impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("adapter", &self.info)
            .field("budget", &self.budget)
            .finish()
    }
}
impl Device {
    pub fn new(options: &DeviceOptions) -> Result<Arc<Self>> {
        let (entry, instance) = open_instance()?;
        let initialized = (|| -> Result<_> {
            let devices = unsafe { instance.enumerate_physical_devices() }
                .context("enumerate Vulkan adapters")?;
            let adapters = devices
                .iter()
                .map(|&p| Adapter::query(&instance, p))
                .collect::<Vec<_>>();
            let selected = choose_adapter(&adapters, options)?
                .context("no compatible Vulkan compute adapter; inspect engine doctor rejection_reasons; software adapters require the explicit test-only allow-software option")?;
            let physical = devices[selected];
            let adapter = &adapters[selected];
            let properties = adapter.properties;
            let queue_family = adapter
                .compute_queue
                .context("adapter has no compute queue")?;
            let memory = adapter.memory;
            let budget = adapter.budget(options);
            let priority = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priority)];
            let available = unsafe { instance.get_physical_device_features(physical) };
            let features = vk::PhysicalDeviceFeatures::default()
                .robust_buffer_access(available.robust_buffer_access != 0);
            let raw = unsafe {
                instance.create_device(
                    physical,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queue_info)
                        .enabled_features(&features),
                    None,
                )
            }
            .context("create Vulkan device")?;
            let created = (|| -> Result<_> {
                let bindings: Vec<_> = (0..BINDINGS)
                    .map(|binding| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    })
                    .collect();
                let set_layout = unsafe {
                    raw.create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                }?;
                let layouts = [set_layout];
                let ranges = [vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(64)];
                let pipeline_layout = match unsafe {
                    raw.create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&layouts)
                            .push_constant_ranges(&ranges),
                        None,
                    )
                } {
                    Ok(x) => x,
                    Err(e) => {
                        unsafe { raw.destroy_descriptor_set_layout(set_layout, None) };
                        return Err(e.into());
                    }
                };
                let binary = include_bytes!("ops.spv");
                ensure!(binary.len() % 4 == 0, "invalid embedded SPIR-V length");
                let words: Vec<u32> = binary
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let module = match unsafe {
                    raw.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&words),
                        None,
                    )
                } {
                    Ok(x) => x,
                    Err(e) => {
                        unsafe {
                            raw.destroy_pipeline_layout(pipeline_layout, None);
                            raw.destroy_descriptor_set_layout(set_layout, None)
                        };
                        return Err(e.into());
                    }
                };
                let main = CString::new("main").unwrap();
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(&main);
                let pipelines = unsafe {
                    raw.create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::ComputePipelineCreateInfo::default()
                            .stage(stage)
                            .layout(pipeline_layout)],
                        None,
                    )
                };
                unsafe { raw.destroy_shader_module(module, None) };
                match pipelines {
                    Ok(p) => Ok((set_layout, pipeline_layout, p[0])),
                    Err((partial, e)) => {
                        unsafe {
                            for p in partial {
                                raw.destroy_pipeline(p, None)
                            }
                            raw.destroy_pipeline_layout(pipeline_layout, None);
                            raw.destroy_descriptor_set_layout(set_layout, None)
                        };
                        Err(e.into())
                    }
                }
            })();
            match created {
                Ok((set_layout, pipeline_layout, pipeline)) => Ok((
                    raw,
                    queue_family,
                    properties,
                    memory,
                    set_layout,
                    pipeline_layout,
                    pipeline,
                    describe(adapter, selected, options),
                    budget,
                )),
                Err(e) => {
                    unsafe { raw.destroy_device(None) };
                    Err(e)
                }
            }
        })();
        let (
            raw,
            queue_family,
            properties,
            memory,
            set_layout,
            pipeline_layout,
            pipeline,
            mut info,
            budget,
        ) = match initialized {
            Ok(x) => x,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        info["discovery_only"] = json!(false);
        let queue = unsafe { raw.get_device_queue(queue_family, 0) };
        Ok(Arc::new(Self {
            _entry: entry,
            instance,
            raw,
            queue,
            queue_family,
            queue_lock: Mutex::new(()),
            properties,
            memory,
            set_layout,
            pipeline_layout,
            pipeline,
            info,
            budget,
            allocated: AtomicU64::new(0),
            high_water: AtomicU64::new(0),
            allocations: AtomicUsize::new(0),
            uploads: AtomicU64::new(0),
            readbacks: AtomicU64::new(0),
            submissions: AtomicU64::new(0),
            dispatches: AtomicU64::new(0),
        }))
    }
    pub fn info(&self) -> Value {
        self.info.clone()
    }
    pub fn reserved(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }
    pub fn stats(&self) -> Value {
        json!({"allocation_budget_bytes":self.budget,"live_allocation_bytes":self.reserved(),"high_water_allocation_bytes":self.high_water.load(Ordering::Relaxed),
        "live_allocations":self.allocations.load(Ordering::Relaxed),"uploaded_bytes":self.uploads.load(Ordering::Relaxed),"readback_bytes":self.readbacks.load(Ordering::Relaxed),
        "queue_submissions":self.submissions.load(Ordering::Relaxed),"compute_dispatches":self.dispatches.load(Ordering::Relaxed),
        "budget_is_owned_allocation_limit_not_free_vram":true,"software":self.info["software"]})
    }
    fn reserve(&self, bytes: u64) -> Result<()> {
        let result = self
            .allocated
            .try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(bytes).filter(|next| *next <= self.budget)
            });
        let old=result.map_err(|old|BudgetExceeded(format!("Vulkan owned-allocation budget exceeded: {old} live + {bytes} requested > {} bytes",self.budget)))?;
        self.high_water.fetch_max(old + bytes, Ordering::Relaxed);
        Ok(())
    }
    pub fn buffer(
        self: &Arc<Self>,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Arc<Buffer>> {
        ensure!(bytes > 0, "zero-sized Vulkan buffer");
        if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            ensure!(
                bytes <= self.properties.limits.max_storage_buffer_range as usize,
                "tensor/state buffer {bytes} exceeds adapter storage-buffer limit {}",
                self.properties.limits.max_storage_buffer_range
            );
        }
        let size = bytes.checked_add(3).context("buffer size overflow")? & !3;
        let raw = unsafe {
            self.raw.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .context("create Vulkan buffer")?;
        let req = unsafe { self.raw.get_buffer_memory_requirements(raw) };
        let required = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE
        } else {
            vk::MemoryPropertyFlags::empty()
        };
        let preferred = if host {
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let memory_type = (0..self.memory.memory_type_count)
            .filter(|i| {
                req.memory_type_bits & (1u32 << i) != 0
                    && self.memory.memory_types[*i as usize]
                        .property_flags
                        .contains(required)
            })
            .max_by_key(|i| {
                (self.memory.memory_types[*i as usize].property_flags & preferred)
                    .as_raw()
                    .count_ones()
            });
        let Some(memory_type) = memory_type else {
            unsafe { self.raw.destroy_buffer(raw, None) };
            bail!("no compatible Vulkan memory type (host visible={host})");
        };
        if let Err(e) = self.reserve(req.size) {
            unsafe { self.raw.destroy_buffer(raw, None) };
            return Err(e);
        }
        let count = self.allocations.fetch_add(1, Ordering::AcqRel);
        if count >= self.properties.limits.max_memory_allocation_count as usize {
            self.allocations.fetch_sub(1, Ordering::AcqRel);
            self.allocated.fetch_sub(req.size, Ordering::AcqRel);
            unsafe { self.raw.destroy_buffer(raw, None) };
            bail!("Vulkan memory allocation count limit reached");
        }
        let memory = match unsafe {
            self.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                self.allocations.fetch_sub(1, Ordering::AcqRel);
                self.allocated.fetch_sub(req.size, Ordering::AcqRel);
                unsafe { self.raw.destroy_buffer(raw, None) };
                return Err(e).context("allocate Vulkan memory");
            }
        };
        let initialized = (|| -> Result<Option<NonNull<u8>>> {
            unsafe { self.raw.bind_buffer_memory(raw, memory, 0) }
                .context("bind Vulkan buffer memory")?;
            if host {
                let ptr = unsafe {
                    self.raw
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                }
                .context("map host transfer memory")?;
                Ok(Some(
                    NonNull::new(ptr.cast()).context("Vulkan returned null map")?,
                ))
            } else {
                Ok(None)
            }
        })();
        let mapped = match initialized {
            Ok(m) => m,
            Err(e) => {
                unsafe {
                    self.raw.destroy_buffer(raw, None);
                    self.raw.free_memory(memory, None)
                }
                self.allocations.fetch_sub(1, Ordering::AcqRel);
                self.allocated.fetch_sub(req.size, Ordering::AcqRel);
                return Err(e);
            }
        };
        let coherent = self.memory.memory_types[memory_type as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        Ok(Arc::new(Buffer {
            device: Arc::clone(self),
            raw,
            memory,
            size: size as u64,
            allocated: req.size,
            mapped,
            coherent,
            host_access: Mutex::new(()),
        }))
    }
    pub fn storage(self: &Arc<Self>, bytes: usize) -> Result<Arc<Buffer>> {
        self.buffer(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
        )
    }
    pub fn host(self: &Arc<Self>, bytes: usize) -> Result<Arc<Buffer>> {
        self.buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )
    }
    pub fn upload(self: &Arc<Self>, data: &[u8]) -> Result<Arc<Buffer>> {
        let destination = self.storage(data.len())?;
        let chunk = 16 * 1024 * 1024;
        let staging = self.host(data.len().min(chunk))?;
        let mut commands = Commands::new(Arc::clone(self), 1)?;
        for (i, bytes) in data.chunks(chunk).enumerate() {
            staging.write(0, bytes)?;
            let copied = bytes.len().next_multiple_of(4);
            if copied > bytes.len() {
                staging.write(bytes.len(), &[0u8; 3][..copied - bytes.len()])?;
            }
            commands.begin()?;
            unsafe {
                self.raw.cmd_copy_buffer(
                    commands.raw,
                    staging.raw,
                    destination.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: (i * chunk) as u64,
                        size: copied as u64,
                    }],
                )
            };
            commands.transfer_to_compute();
            commands.submit()?;
        }
        self.uploads.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(destination)
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_pipeline(self.pipeline, None);
            self.raw.destroy_pipeline_layout(self.pipeline_layout, None);
            self.raw
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.raw.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(crate) struct Buffer {
    pub device: Arc<Device>,
    pub raw: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    allocated: u64,
    mapped: Option<NonNull<u8>>,
    coherent: bool,
    host_access: Mutex<()>,
}
// SAFETY: mappings live with their allocations; all CPU accesses are mutex
// protected. Callers submit/fence GPU use before reading or writing a mapping.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("bytes", &self.size)
            .finish()
    }
}
impl Buffer {
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<()> {
        ensure!(
            offset
                .checked_add(data.len())
                .is_some_and(|n| n as u64 <= self.size),
            "mapped buffer write exceeds allocation"
        );
        let ptr = self.mapped.context("buffer is not host mapped")?;
        let _guard = self
            .host_access
            .lock()
            .map_err(|_| anyhow::anyhow!("host mapping lock poisoned"))?;
        // SAFETY: bounds above, persistent valid mapping and exclusive CPU access.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr().add(offset), data.len());
        }
        if !self.coherent {
            unsafe {
                self.device
                    .raw
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(self.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .context("flush mapped Vulkan buffer")?;
        }
        Ok(())
    }
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<()> {
        ensure!(
            offset
                .checked_add(output.len())
                .is_some_and(|n| n as u64 <= self.size),
            "mapped buffer read exceeds allocation"
        );
        let ptr = self.mapped.context("buffer is not host mapped")?;
        let _guard = self
            .host_access
            .lock()
            .map_err(|_| anyhow::anyhow!("host mapping lock poisoned"))?;
        if !self.coherent {
            unsafe {
                self.device
                    .raw
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(self.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
            }
            .context("invalidate mapped Vulkan buffer")?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr().add(offset),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(())
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.device.raw.unmap_memory(self.memory)
            }
            self.device.raw.destroy_buffer(self.raw, None);
            self.device.raw.free_memory(self.memory, None);
        }
        self.device
            .allocated
            .fetch_sub(self.allocated, Ordering::AcqRel);
        self.device.allocations.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Push(pub [u32; 16]);
impl Push {
    pub fn new(op: u32) -> Self {
        let mut p = Self::default();
        p.0[0] = op;
        p
    }
    pub fn n(mut self, index: usize, value: usize) -> Result<Self> {
        ensure!(index < 7, "push dimension index outside layout");
        self.0[index + 1] =
            u32::try_from(value).context("GPU dimension exceeds 32-bit address range")?;
        Ok(self)
    }
    pub fn f(mut self, index: usize, value: f32) -> Self {
        self.0[index + 8] = value.to_bits();
        self
    }
    pub fn flags(mut self, value: u32) -> Self {
        self.0[15] = value;
        self
    }
}
pub(crate) struct Operation {
    pub set: vk::DescriptorSet,
    pub push: Push,
    pub groups: [u32; 3],
    pub buffers: [Arc<Buffer>; 7],
}
impl std::fmt::Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Operation")
            .field("op", &self.push.0[0])
            .field("groups", &self.groups)
            .finish()
    }
}

pub(crate) struct Commands {
    pub device: Arc<Device>,
    pool: vk::CommandPool,
    pub raw: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    inflight: bool,
    sets_left: usize,
    sets_total: usize,
}
impl std::fmt::Debug for Commands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Commands")
            .field("inflight", &self.inflight)
            .finish()
    }
}
impl Commands {
    pub fn new(device: Arc<Device>, sets: usize) -> Result<Self> {
        ensure!(
            sets > 0 && sets <= 100_000,
            "descriptor-set request outside engine bound"
        );
        let pool = unsafe {
            device.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        let initialized = (|| -> Result<_> {
            let raw = unsafe {
                device.raw.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }?[0];
            let fence = unsafe {
                device
                    .raw
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (sets as u32) * BINDINGS,
            }];
            match unsafe {
                device.raw.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(sets as u32)
                        .pool_sizes(&sizes),
                    None,
                )
            } {
                Ok(descriptor_pool) => Ok((raw, fence, descriptor_pool)),
                Err(e) => {
                    unsafe { device.raw.destroy_fence(fence, None) };
                    Err(e.into())
                }
            }
        })();
        let (raw, fence, descriptor_pool) = match initialized {
            Ok(x) => x,
            Err(e) => {
                unsafe { device.raw.destroy_command_pool(pool, None) };
                return Err(e);
            }
        };
        Ok(Self {
            device,
            pool,
            raw,
            fence,
            descriptor_pool,
            inflight: false,
            sets_left: sets,
            sets_total: sets,
        })
    }
    pub fn operation(
        &mut self,
        mut push: Push,
        buffers: [Arc<Buffer>; 7],
        groups: [usize; 3],
    ) -> Result<Operation> {
        ensure!(self.sets_left > 0, "compute descriptor pool exhausted");
        ensure!(
            buffers.iter().all(|b| Arc::ptr_eq(&b.device, &self.device)),
            "cross-device Vulkan buffer binding"
        );
        let max = self.device.properties.limits.max_compute_work_group_count;
        let mut actual = [0u32; 3];
        for i in 0..3 {
            actual[i] = u32::try_from(groups[i]).context("dispatch dimension overflow")?;
            ensure!(
                actual[i] > 0 && actual[i] <= max[i],
                "compute dispatch exceeds adapter dimension {i} limit"
            );
        }
        push.0[14] = actual[0];
        let layout = [self.device.set_layout];
        let set = unsafe {
            self.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layout),
            )
        }?[0];
        self.sets_left -= 1;
        self.update_set(set, &buffers);
        Ok(Operation {
            set,
            push,
            groups: actual,
            buffers,
        })
    }
    pub fn update_set(&self, set: vk::DescriptorSet, buffers: &[Arc<Buffer>; 7]) {
        let infos: Vec<_> = buffers
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.raw)
                    .offset(0)
                    .range(b.size)]
            })
            .collect();
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
    }
    /// For a dynamic graph whose operation leases have been dropped after its
    /// fence. Persistent graphs must not call this while retaining descriptors.
    pub fn reset_descriptors(&mut self) -> Result<()> {
        ensure!(
            !self.inflight,
            "cannot reset descriptors while commands execute"
        );
        unsafe {
            self.device
                .raw
                .reset_command_buffer(self.raw, vk::CommandBufferResetFlags::empty())?;
            self.device.raw.reset_descriptor_pool(
                self.descriptor_pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
        }
        self.sets_left = self.sets_total;
        Ok(())
    }
    pub fn groups(&self, count: usize) -> Result<[usize; 3]> {
        ensure!(count > 0, "empty compute dispatch");
        let x = count.min(self.device.properties.limits.max_compute_work_group_count[0] as usize);
        let y = count.div_ceil(x);
        ensure!(
            y <= self.device.properties.limits.max_compute_work_group_count[1] as usize,
            "dispatch group count exceeds hardware range"
        );
        Ok([x, y, 1])
    }
    pub fn begin(&mut self) -> Result<()> {
        ensure!(!self.inflight, "command buffer is still in flight");
        unsafe {
            self.device
                .raw
                .reset_command_buffer(self.raw, vk::CommandBufferResetFlags::empty())?;
            self.device.raw.begin_command_buffer(
                self.raw,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device.raw.cmd_bind_pipeline(
                self.raw,
                vk::PipelineBindPoint::COMPUTE,
                self.device.pipeline,
            );
        }
        Ok(())
    }
    pub fn record(&self, operation: &Operation, push: &Push, groups: [u32; 3]) {
        // SAFETY: Push contains exactly 16 initialized u32s, the pipeline expects
        // 64 bytes, and Vulkan consumes the values during this synchronous call.
        let bytes = unsafe { std::slice::from_raw_parts(push.0.as_ptr().cast::<u8>(), 64) };
        unsafe {
            self.device.raw.cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::COMPUTE,
                self.device.pipeline_layout,
                0,
                &[operation.set],
                &[],
            );
            self.device.raw.cmd_push_constants(
                self.raw,
                self.device.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            self.device
                .raw
                .cmd_dispatch(self.raw, groups[0], groups[1], groups[2]);
        }
        self.device.dispatches.fetch_add(1, Ordering::Relaxed);
        self.compute_barrier();
    }
    pub fn compute_barrier(&self) {
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.raw,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
        }
    }
    pub fn transfer_to_compute(&self) {
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.raw,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
        }
    }
    pub fn compute_to_transfer(&self) {
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.raw,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
                &[],
            );
        }
    }
    pub fn transfer_to_host(&self) {
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.raw,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
        }
    }
    pub fn submit(&mut self) -> Result<()> {
        self.submit_with_host_work(|| Ok(()))
    }

    /// Submit Vulkan work, then perform CPU-only work while its fence is pending.
    /// The fence is always waited even when host work fails; staging buffers
    /// and packed-weight leases cannot be released while GPU copies are in flight.
    /// The callback must never access in-flight GPU buffer contents.
    pub fn submit_with_host_work(&mut self, host_work: impl FnOnce() -> Result<()>) -> Result<()> {
        unsafe {
            self.device.raw.end_command_buffer(self.raw)?;
            self.device.raw.reset_fences(&[self.fence])?;
        }
        let command_buffers = [self.raw];
        let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        {
            let _queue = self
                .device
                .queue_lock
                .lock()
                .map_err(|_| anyhow::anyhow!("Vulkan queue lock poisoned"))?;
            unsafe {
                self.device
                    .raw
                    .queue_submit(self.device.queue, &submit, self.fence)
            }
            .context("submit independent Vulkan computation")?;
            self.inflight = true;
        }
        self.device.submissions.fetch_add(1, Ordering::Relaxed);
        let host_result = host_work();
        let fence_result = unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX)
        }
        .context("wait for independent Vulkan computation");
        if fence_result.is_ok() {
            self.inflight = false;
        }
        fence_result?;
        host_result
    }
}
impl Drop for Commands {
    fn drop(&mut self) {
        unsafe {
            if self.inflight {
                let _ = self
                    .device
                    .raw
                    .wait_for_fences(&[self.fence], true, u64::MAX);
            }
            self.device
                .raw
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.raw.destroy_fence(self.fence, None);
            self.device.raw.destroy_command_pool(self.pool, None);
        }
    }
}

pub(crate) fn bytes_of_f32(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
pub(crate) fn bytes_of_f32_mut(values: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}
pub(crate) fn float_bytes(elements: usize) -> Result<usize> {
    elements
        .checked_mul(4)
        .context("GPU float buffer size overflow")
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn adapter(vendor: u32, name: &str, kind: vk::PhysicalDeviceType, heap_mib: u64) -> Adapter {
        let mut properties = vk::PhysicalDeviceProperties::default();
        properties.vendor_id = vendor;
        properties.device_type = kind;
        properties.api_version = vk::API_VERSION_1_1;
        for (target, source) in properties.device_name.iter_mut().zip(name.bytes()) {
            *target = source as _;
        }
        properties.limits.max_compute_work_group_invocations = WORKGROUP as u32;
        properties.limits.max_compute_work_group_size = [WORKGROUP as u32, 1, 1];
        properties.limits.max_compute_shared_memory_size = (WORKGROUP * 4) as u32;
        properties.limits.max_per_stage_descriptor_storage_buffers = BINDINGS;
        properties.limits.max_descriptor_set_storage_buffers = BINDINGS;
        properties.limits.max_push_constants_size = 64;
        let mut memory = vk::PhysicalDeviceMemoryProperties::default();
        memory.memory_heap_count = 1;
        memory.memory_heaps[0] = vk::MemoryHeap {
            size: heap_mib * MIB,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        memory.memory_type_count = 1;
        memory.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE,
            heap_index: 0,
        };
        Adapter {
            properties,
            memory,
            compute_queue: Some(0),
        }
    }

    #[test]
    fn incompatible_higher_ranked_adapter_is_skipped_before_selection() {
        let mut discrete = adapter(
            0x1002,
            "AMD Radeon RX 9070 XT",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            16_384,
        );
        discrete
            .properties
            .limits
            .max_descriptor_set_storage_buffers = BINDINGS - 1;
        let integrated = adapter(
            0x8086,
            "Intel integrated",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            8192,
        );
        let adapters = [discrete, integrated];
        assert_eq!(
            choose_adapter(&adapters, &DeviceOptions::default()).unwrap(),
            Some(1)
        );
        let report = describe(&adapters[0], 0, &DeviceOptions::default());
        assert_eq!(report["compatible"], false);
        assert!(
            report["rejection_reasons"]
                .as_array()
                .unwrap()
                .iter()
                .any(|reason| reason.as_str().unwrap().contains("seven"))
        );
    }

    #[test]
    fn vendor_and_model_name_do_not_affect_hardware_rank() {
        for (vendor, name) in [
            (0x10de, "NVIDIA GPU"),
            (0x8086, "Intel Arc"),
            (0x1234, "Future hardware"),
        ] {
            let other = adapter(vendor, name, vk::PhysicalDeviceType::DISCRETE_GPU, 16_384);
            let radeon = adapter(
                0x1002,
                "AMD Radeon RX 9070 XT",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                16_384,
            );
            assert_eq!(
                choose_adapter(&[other, radeon], &DeviceOptions::default()).unwrap(),
                Some(0)
            );
        }
    }

    #[test]
    fn device_class_then_heap_capacity_then_index_break_ties() {
        let adapters = [
            adapter(
                0x8086,
                "Integrated GPU",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                32_768,
            ),
            adapter(
                0x1002,
                "Discrete A",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                8192,
            ),
            adapter(
                0x10de,
                "Discrete B",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                16_384,
            ),
            adapter(
                0x8086,
                "Discrete C",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                16_384,
            ),
        ];
        assert_eq!(
            choose_adapter(&adapters, &DeviceOptions::default()).unwrap(),
            Some(2)
        );
    }

    #[test]
    fn software_is_an_explicit_opt_in_and_ranks_after_hardware() {
        let software = adapter(0, "llvmpipe", vk::PhysicalDeviceType::CPU, 16_384);
        let options = DeviceOptions::default();
        assert_eq!(
            choose_adapter(std::slice::from_ref(&software), &options).unwrap(),
            None
        );
        let allowed = DeviceOptions {
            allow_software: true,
            ..options
        };
        assert_eq!(
            choose_adapter(std::slice::from_ref(&software), &allowed).unwrap(),
            Some(0)
        );
        let hardware = adapter(0x1234, "Hardware", vk::PhysicalDeviceType::OTHER, 256);
        assert_eq!(
            choose_adapter(&[software, hardware], &allowed).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn explicit_device_selection_is_strict() {
        let mut bad = adapter(
            0x1002,
            "Unsupported",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8192,
        );
        bad.compute_queue = None;
        let good = adapter(
            0x10de,
            "Supported",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8192,
        );
        let adapters = [bad, good];
        let mut options = DeviceOptions {
            device_index: Some(0),
            ..Default::default()
        };
        assert!(
            choose_adapter(&adapters, &options)
                .unwrap_err()
                .to_string()
                .contains("no compute queue")
        );
        options.device_index = Some(2);
        assert!(
            choose_adapter(&adapters, &options)
                .unwrap_err()
                .to_string()
                .contains("outside 2 adapters")
        );
        options.device_index = Some(1);
        assert_eq!(choose_adapter(&adapters, &options).unwrap(), Some(1));
    }

    #[test]
    fn explicit_budget_is_validated_before_ranking_and_never_silently_dropped() {
        let adapters = [
            adapter(
                0x1002,
                "Small discrete",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                256,
            ),
            adapter(
                0x8086,
                "Larger integrated",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                1024,
            ),
        ];
        let mut options = DeviceOptions {
            memory_budget_bytes: 512 * MIB as usize,
            ..Default::default()
        };
        assert_eq!(choose_adapter(&adapters, &options).unwrap(), Some(1));
        options.device_index = Some(0);
        assert!(
            choose_adapter(&adapters, &options)
                .unwrap_err()
                .to_string()
                .contains("allocation budget")
        );
        options.device_index = None;
        options.memory_budget_bytes = 2048 * MIB as usize;
        assert!(choose_adapter(&adapters, &options).is_err());
        options.memory_budget_bytes = 1;
        assert!(
            choose_adapter(&adapters, &options)
                .unwrap_err()
                .to_string()
                .contains("at least 1 MiB")
        );
    }

    #[test]
    fn discovery_and_selection_share_all_baseline_requirements() {
        let mut bad = adapter(
            0x1234,
            "Unsupported capabilities",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            256,
        );
        bad.properties.api_version = vk::API_VERSION_1_0;
        bad.properties.limits.max_compute_work_group_invocations = 64;
        bad.properties.limits.max_compute_shared_memory_size = 256;
        bad.properties.limits.max_push_constants_size = 32;
        bad.memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let options = DeviceOptions::default();
        let reasons = bad.rejection_reasons(&options);
        assert_eq!(reasons.len(), 5);
        let report = describe(&bad, 0, &options);
        assert_eq!(report["rejection_reasons"], json!(reasons));
        assert_eq!(report["compatible"], false);
        assert_eq!(report["host_visible_staging_memory"], false);
        assert_eq!(report["compute_verified"], false);
        assert_eq!(choose_adapter(&[bad], &options).unwrap(), None);
    }

    #[test]
    fn device_local_heap_needs_an_usable_memory_type_and_is_not_free_memory() {
        let mut device = adapter(
            0x1234,
            "Missing local memory",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            256,
        );
        device.memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        let options = DeviceOptions::default();
        assert_eq!(device.largest_usable_heap(), 0);
        assert_eq!(
            choose_adapter(std::slice::from_ref(&device), &options).unwrap(),
            None
        );
        device.memory.memory_types[0].property_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let report = describe(&device, 0, &options);
        assert_eq!(report["compatible"], true);
        assert_eq!(
            report["allocation_budget_bytes"],
            json!((256 * MIB) * 7 / 10)
        );
        assert_eq!(report["allocation_budget_is_free_memory"], false);
        assert_eq!(report["compute_verified"], false);
    }
}
