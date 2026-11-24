//! Vulkan ownership, bounded allocation and reusable compute submissions.
use anyhow::{Context as _, Result, bail, ensure};
use ash::{Entry, Instance, vk};
use serde_json::{Value, json};
use std::{
    cell::Cell,
    ffi::{CStr, CString},
    ptr::NonNull,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

const BINDINGS: u32 = 7;
pub const WORKGROUP: usize = 128;

/// Physical storage used for ordinary Qwen3 attention history.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
#[allow(non_camel_case_types)]
pub enum KvPrecision {
    #[default]
    F32,
    #[value(alias = "q4")]
    Q4_64,
}
impl KvPrecision {
    pub fn is_compact(self) -> bool {
        self == Self::Q4_64
    }
    pub fn is_standard(&self) -> bool {
        *self == Self::F32
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q4_64 => "Q4_64",
        }
    }
    pub fn row_bytes(self, width: usize) -> Result<usize> {
        ensure!(width > 0, "attention-cache width must be positive");
        match self {
            Self::F32 => width.checked_mul(4),
            Self::Q4_64 => width
                .checked_add(127)
                .map(|n| n / 128)
                .and_then(|n| n.checked_mul(68)),
        }
        .context("attention-cache row size overflow")
    }
}

/// KV precision and placement are independent: moving history does not alter it.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lowercase")]
pub enum KvPlacement {
    #[default]
    Gpu,
    Ram,
}
impl KvPlacement {
    pub fn is_gpu(&self) -> bool {
        *self == Self::Gpu
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Ram => "ram",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllocationClass {
    Gpu,
    Host,
}

#[derive(Clone, Debug)]
pub struct DeviceOptions {
    pub device_index: Option<usize>,
    /// Explicitly enable a CPU/software Vulkan implementation for shader tests.
    /// Production discovery never accepts one by default.
    pub allow_software: bool,
    /// Opt into Q8 activation integer-dot kernels on supporting adapters.
    pub fast_math: bool,
    /// None uses engine policy; Some(false) always keeps legacy packed copies.
    pub direct_experts: Option<bool>,
    pub kv_precision: KvPrecision,
    pub kv_placement: KvPlacement,
    /// Independent cap for Vulkan allocations backed by system RAM; zero plans from available RAM.
    pub host_memory_budget_bytes: usize,
    /// Maximum total live Vulkan allocation bytes owned by this engine. Zero
    /// selects 70% of the largest usable device-local heap; this is not free VRAM.
    pub memory_budget_bytes: usize,
}
impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            device_index: None,
            allow_software: false,
            fast_math: false,
            direct_experts: None,
            kv_precision: KvPrecision::F32,
            kv_placement: KvPlacement::Gpu,
            host_memory_budget_bytes: 0,
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
        .api_version(supported.min(vk::API_VERSION_1_3));
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
    integer_dot_product: bool,
    direct_experts_supported: bool,
    integer_dot_product_accelerated: bool,
    integer_dot_extension: bool,
    heap_budgets: Option<Vec<(u64, u64)>>,
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
        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let memory_budget_available = extensions.iter().any(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }
            == ash::ext::memory_budget::NAME);
        let heap_budgets = if memory_budget_available {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut properties =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                instance.get_physical_device_memory_properties2(physical, &mut properties);
            }
            Some(
                (0..memory.memory_heap_count as usize)
                    .map(|i| (budget.heap_budget[i], budget.heap_usage[i]))
                    .collect(),
            )
        } else {
            None
        };
        let integer_dot_extension = extensions.iter()
            .any(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }
                == ash::khr::shader_integer_dot_product::NAME);
        let query_integer_dot =
            properties.api_version >= vk::API_VERSION_1_3 || integer_dot_extension;
        let mut integer_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut integer_dot_properties =
            vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        if query_integer_dot {
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut integer_dot);
            let mut details =
                vk::PhysicalDeviceProperties2::default().push_next(&mut integer_dot_properties);
            // These structures are core in Vulkan 1.3 and defined by the advertised
            // KHR extension on earlier implementations.
            unsafe {
                instance.get_physical_device_features2(physical, &mut features);
                instance.get_physical_device_properties2(physical, &mut details);
            }
        }
        Self {
            properties,
            memory,
            compute_queue,
            integer_dot_product: integer_dot.shader_integer_dot_product != 0,
            direct_experts_supported: unsafe { instance.get_physical_device_features(physical) }.shader_storage_buffer_array_dynamic_indexing != 0
                && properties.limits.max_per_stage_descriptor_storage_buffers >= BINDINGS + 7
                && properties.limits.max_descriptor_set_storage_buffers >= BINDINGS + 7,
            integer_dot_product_accelerated: integer_dot_properties
                .integer_dot_product4x8_bit_packed_signed_accelerated
                != 0,
            integer_dot_extension,
            heap_budgets,
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

    fn has_ram_storage_memory(&self) -> bool {
        self.memory.memory_types[..self.memory.memory_type_count as usize]
            .iter()
            .any(|ty| {
                ty.property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && ty.heap_index < self.memory.memory_heap_count
                    && (!self.memory.memory_heaps[ty.heap_index as usize]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                        || software(&self.properties))
            })
    }
    fn fit_budget(&self, options: &DeviceOptions) -> u64 {
        let Some(budgets) = &self.heap_budgets else {
            return self.budget(options);
        };
        let available = self.memory.memory_heaps[..self.memory.memory_heap_count as usize]
            .iter()
            .enumerate()
            .filter(|(_, heap)| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|(index, heap)| {
                budgets[index]
                    .0
                    .min(heap.size)
                    .saturating_sub(budgets[index].1)
            })
            .max()
            .unwrap_or(0);
        self.budget(options)
            .min(available.saturating_sub(256 * 1024 * 1024))
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
        if options.kv_placement == KvPlacement::Ram && !self.has_ram_storage_memory() {
            reasons.push("RAM context placement requires host-visible system-memory storage; host-visible VRAM is insufficient".into());
        }
        if options.direct_experts == Some(true) && !self.direct_experts_supported {
            reasons.push("direct expert access requires storage-buffer-array dynamic indexing and at least 14 storage-buffer descriptors".into());
        }
        if options.fast_math && !self.integer_dot_product {
            reasons.push("fast math requires the shaderIntegerDotProduct feature (Vulkan 1.3 or VK_KHR_shader_integer_dot_product)".into());
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
    ensure!(selected.is_some() || options.direct_experts != Some(true), "no eligible Vulkan adapter supports forced direct expert access");
    ensure!(
        selected.is_some() || !options.fast_math,
        "no eligible Vulkan adapter supports the requested fast math integer-dot kernels"
    );
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

/// Fresh discovery/budget snapshot without creating a logical device or compute work.
pub fn inspect_device(options: &DeviceOptions) -> Result<Value> {
    let (_entry, instance) = open_instance()?;
    let result = (|| {
        let adapters = unsafe { instance.enumerate_physical_devices() }?
            .into_iter()
            .map(|physical| Adapter::query(&instance, physical))
            .collect::<Vec<_>>();
        let selected =
            choose_adapter(&adapters, options)?.context("no compatible Vulkan adapter")?;
        Ok(describe(&adapters[selected], selected, options))
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
        "ram_context_storage_available":adapter.has_ram_storage_memory(),
        "memory_budget_extension_available":adapter.heap_budgets.is_some(),
        "physical_heap_budget_snapshot":adapter.heap_budgets.as_ref().map(|budgets| budgets.iter().enumerate().map(|(i,(budget,usage))|json!({
            "heap":i,"device_local":memory.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            "budget_bytes":(*budget).min(memory.memory_heaps[i].size),"reported_budget_bytes":budget,
            "usage_bytes":usage,"available_bytes":(*budget).min(memory.memory_heaps[i].size).saturating_sub(*usage)})).collect::<Vec<_>>()),
        "fit_allocation_budget_bytes":adapter.fit_budget(options),
        "fit_headroom_bytes":256u64*1024*1024,
        "physical_budget_is_snapshot_not_owned_allocation_limit":true,
        "allocation_budget_bytes":adapter.budget(options),"allocation_budget_is_free_memory":false,
        "max_storage_buffer_bytes":p.limits.max_storage_buffer_range,
        "max_workgroup_invocations":p.limits.max_compute_work_group_invocations,
        "direct_expert_weights_supported":adapter.direct_experts_supported,
        "direct_experts":options.direct_experts,
        "fast_math_available":adapter.integer_dot_product,
        "integer_dot_product_packed_signed_accelerated":adapter.integer_dot_product_accelerated,
        "fast_math":options.fast_math,"kv_precision":options.kv_precision.as_str(),"kv_placement":options.kv_placement.as_str(),
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

#[derive(Clone, Copy)]
struct ComputePipeline {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}
impl ComputePipeline {
    fn destroy(self, raw: &ash::Device) {
        unsafe {
            raw.destroy_pipeline(self.pipeline, None);
            raw.destroy_pipeline_layout(self.pipeline_layout, None);
            raw.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

// Keep the ordinary pipeline's exact original module AND seven-single-buffer
// layout. The additional array module is used only by expert GEMV operations.
fn create_compute_pipeline(raw: &ash::Device, array_weights: bool, fast_math: bool) -> Result<ComputePipeline> {
    let bindings: Vec<_> = (0..BINDINGS)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(if binding == 0 && array_weights { 8 } else { 1 })
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
    let binary: &[u8] = match (array_weights, fast_math) {
        (true, true) => include_bytes!("ops_fast_direct.spv"),
        (true, false) => include_bytes!("ops_direct.spv"),
        (false, true) => include_bytes!("ops_fast.spv"),
        (false, false) => include_bytes!("ops.spv"),
    };
    match compile_compute_shader(raw, pipeline_layout, binary) {
        Ok(pipeline) => Ok(ComputePipeline { set_layout, pipeline_layout, pipeline }),
        Err(error) => {
            unsafe {
                raw.destroy_pipeline_layout(pipeline_layout, None);
                raw.destroy_descriptor_set_layout(set_layout, None);
            }
            Err(error)
        }
    }
}

fn compile_compute_shader(
    raw: &ash::Device,
    layout: vk::PipelineLayout,
    binary: &[u8],
) -> Result<vk::Pipeline> {
    ensure!(binary.len() % 4 == 0, "invalid embedded SPIR-V length");
    let words: Vec<u32> = binary.chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    let module = unsafe {
        raw.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
    }?;
    let main = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&main);
    let pipelines = unsafe {
        raw.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
            None,
        )
    };
    unsafe { raw.destroy_shader_module(module, None); }
    match pipelines {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((partial, error)) => {
            for pipeline in partial {
                unsafe { raw.destroy_pipeline(pipeline, None); }
            }
            Err(error.into())
        }
    }
}

// The ordinary F32 shader permits contraction. Match its arithmetic only on a
// driver where the pointwise and full-model comparisons confirmed fused MACs.
fn verified_fast_decode_driver(p: &vk::PhysicalDeviceProperties) -> bool {
    cfg!(windows) && (p.vendor_id, p.device_id, p.driver_version, p.api_version)
        == (0x1002, 0x7550, 8_389_024, vk::make_api_version(0, 1, 4, 359))
}

struct FastDecodeShaders {
    words: vk::Pipeline,
    router: Option<vk::Pipeline>,
}
impl FastDecodeShaders {
    fn new(raw: &ash::Device, ordinary: ComputePipeline, direct: ComputePipeline,
           verified_driver: bool) -> Result<Self> {
        let words = compile_compute_shader(raw, direct.pipeline_layout, include_bytes!("expert_words.spv"))?;
        let router = if verified_driver {
            match compile_compute_shader(raw, ordinary.pipeline_layout, include_bytes!("router_f32.spv")) {
                Ok(pipeline) => Some(pipeline),
                Err(error) => {
                    unsafe { raw.destroy_pipeline(words, None); }
                    return Err(error);
                }
            }
        } else {
            None
        };
        Ok(Self { words, router })
    }
    fn destroy(&self, raw: &ash::Device) {
        unsafe {
            raw.destroy_pipeline(self.words, None);
            if let Some(router) = self.router {
                raw.destroy_pipeline(router, None);
            }
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DiagnosticPipelineMode {
    #[default]
    Production,
    AllArrayPacked,
    AllArrayDirect,
}
#[cfg(test)]
thread_local! {
    static DIAGNOSTIC_PIPELINE_MODE: Cell<DiagnosticPipelineMode> = const { Cell::new(DiagnosticPipelineMode::Production) };
}
#[cfg(test)]
pub(crate) fn with_diagnostic_pipeline<T>(mode: DiagnosticPipelineMode, action: impl FnOnce() -> T) -> T {
    struct Restore(DiagnosticPipelineMode);
    impl Drop for Restore {
        fn drop(&mut self) { DIAGNOSTIC_PIPELINE_MODE.set(self.0); }
    }
    let previous=DIAGNOSTIC_PIPELINE_MODE.replace(mode);
    let _restore=Restore(previous);
    action()
}

// The two kernels share the ordinary seven-buffer layout. A graph can survive
// multiple cache sizes, so dispatch selection uses the current push constants.
#[derive(Default)]
struct AttentionShaders {
    scores: Option<vk::Pipeline>,
    values: Option<vk::Pipeline>,
    score_dispatches: AtomicU64,
    value_dispatches: AtomicU64,
}

impl AttentionShaders {
    fn new(raw: &ash::Device, layout: vk::PipelineLayout) -> Self {
        Self {
            scores: compile_compute_shader(raw, layout, include_bytes!("tiled_scores.spv")).ok(),
            values: compile_compute_shader(raw, layout, include_bytes!("long_attention.spv")).ok(),
            ..Self::default()
        }
    }

    fn select(&self, push: &Push, mut groups: [u32; 3]) -> Option<(vk::Pipeline, [u32; 3])> {
        let [op, tokens, heads, kv_heads, width, stride, ..] = push.0;
        if (heads, kv_heads, width) != (32, 4, 128)
            || tokens < 128 || tokens > stride || push.0[15] != 0
        {
            return None;
        }
        let (pipeline, count) = match op {
            4 if groups == [tokens, heads, 1] => {
                groups[0] = tokens.div_ceil(4);
                (self.scores?, &self.score_dispatches)
            }
            5 if groups == [heads, 1, 1] => (self.values?, &self.value_dispatches),
            _ => return None,
        };
        count.fetch_add(1, Ordering::Relaxed);
        Some((pipeline, groups))
    }
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
    direct_pipeline: Option<ComputePipeline>,
    fast_decode: Option<FastDecodeShaders>,
    expert_word_dispatches: AtomicU64,
    router_dispatches: AtomicU64,
    #[cfg(test)]
    pub(crate) diagnostic_pipeline_mode: DiagnosticPipelineMode,
    info: Value,
    pub budget: u64,
    pub(crate) fast_math: bool,
    pub(crate) direct_experts: bool,
    pub(crate) kv_precision: KvPrecision,
    pub(crate) kv_placement: KvPlacement,
    pub host_budget: u64,
    attention: AttentionShaders,
    host_allocated: AtomicU64,
    host_high_water: AtomicU64,
    allocated: AtomicU64,
    high_water: AtomicU64,
    allocations: AtomicUsize,
    pub uploads: AtomicU64,
    pub readbacks: AtomicU64,
    pub submissions: AtomicU64,
    pub dispatches: AtomicU64,
    pub(crate) group_compute_barriers: bool,
    grouped_dispatches: AtomicU64,
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
            let direct_supported = available.shader_storage_buffer_array_dynamic_indexing != 0
                && properties.limits.max_per_stage_descriptor_storage_buffers >= BINDINGS + 7
                && properties.limits.max_descriptor_set_storage_buffers >= BINDINGS + 7;
            // Automatic stays Packed; production Direct selection is explicit.
            let direct_requested = options.direct_experts.unwrap_or(false);
            #[cfg(test)]
            let direct_requested = direct_requested || (options.direct_experts.is_none()
                && std::env::var_os("NEXTORA_DIRECT_EXPERTS").is_some_and(|v| v == "1"))
                || DIAGNOSTIC_PIPELINE_MODE.get() != DiagnosticPipelineMode::Production;
            ensure!(options.direct_experts != Some(true) || direct_supported, "forced direct expert access is unsupported by this adapter");
            #[cfg(test)]
            ensure!(DIAGNOSTIC_PIPELINE_MODE.get() == DiagnosticPipelineMode::Production || direct_supported, "array diagnostic requires a compatible adapter");
            let direct_experts = direct_requested && direct_supported;
            let features = vk::PhysicalDeviceFeatures::default()
                .robust_buffer_access(available.robust_buffer_access != 0)
                .shader_storage_buffer_array_dynamic_indexing(direct_experts);
            let mut integer_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
                .shader_integer_dot_product(options.fast_math);
            let extensions = if options.fast_math && adapter.integer_dot_extension {
                vec![ash::khr::shader_integer_dot_product::NAME.as_ptr()]
            } else {
                Vec::new()
            };
            if options.fast_math && extensions.is_empty() {
                let loader_api = unsafe { entry.try_enumerate_instance_version() }?
                    .unwrap_or(vk::API_VERSION_1_0);
                ensure!(
                    loader_api >= vk::API_VERSION_1_3,
                    "fast math requires a Vulkan 1.3 loader or the VK_KHR_shader_integer_dot_product extension"
                );
            }
            let mut create = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_features(&features)
                .enabled_extension_names(&extensions);
            if options.fast_math {
                create = create.push_next(&mut integer_dot);
            }
            let raw = unsafe { instance.create_device(physical, &create, None) }
                .context("create Vulkan device")?;
            let created = (|| -> Result<_> {
                let legacy = create_compute_pipeline(&raw, false, options.fast_math)?;
                let direct = if direct_experts {
                    match create_compute_pipeline(&raw, true, options.fast_math) {
                        Ok(pipeline) => Some(pipeline),
                        Err(error) => { legacy.destroy(&raw); return Err(error); }
                    }
                } else { None };
                let enable_fast_decode = options.fast_math;
                #[cfg(test)]
                let enable_fast_decode = enable_fast_decode
                    && DIAGNOSTIC_PIPELINE_MODE.get() == DiagnosticPipelineMode::Production;
                let (fast_decode, fallback) = match direct.filter(|_| enable_fast_decode) {
                    Some(array) => match FastDecodeShaders::new(&raw, legacy, array, verified_fast_decode_driver(&properties)) {
                        Ok(shaders) => (Some(shaders), None),
                        Err(error) => (None, Some(error.to_string())),
                    },
                    None => (None, None),
                };
                Ok((legacy, direct, fast_decode, fallback))
            })();
            match created {
                Ok((legacy, direct_pipeline, fast_decode, fallback)) => {
                    let specialized = fast_decode.is_some();
                    let f32_router = fast_decode.as_ref().is_some_and(|shaders| shaders.router.is_some());
                    Ok((
                    raw,
                    queue_family,
                    properties,
                    memory,
                    legacy.set_layout,
                    legacy.pipeline_layout,
                    legacy.pipeline,
                    direct_pipeline,
                    fast_decode,
                    {
                        let mut info = describe(adapter, selected, options);
                        info["direct_expert_weights_supported"] = json!(direct_supported);
                        info["direct_expert_weights_requested"] = json!(direct_requested);
                        info["direct_expert_weights_enabled"] = json!(direct_experts);
                        info["specialized_fast_decode_enabled"] = json!(specialized);
                        info["specialized_fast_decode_fallback_reason"] = json!(fallback);
                        info["specialized_f32_router_enabled"] = json!(f32_router);
                        info["direct_expert_pipeline_scope"] = json!("expert_gemv_only");
                        info["ordinary_shader_module"] = json!(if options.fast_math { "ops_fast.spv" } else { "ops.spv" });
                        #[cfg(test)]
                        { info["diagnostic_pipeline_mode"] = json!(format!("{:?}",DIAGNOSTIC_PIPELINE_MODE.get())); }
                        info
                    },
                    budget,
                ))
                }
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
            direct_pipeline,
            fast_decode,
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
        // The legacy compiler contracts the weighted sum in a driver-specific
        // order. Only select these binaries on the bit-validated configuration.
        let verified_attention = direct_pipeline.is_some() && options.fast_math
            && options.kv_precision == KvPrecision::F32 && verified_fast_decode_driver(&properties);
        #[cfg(test)]
        let verified_attention = verified_attention && DIAGNOSTIC_PIPELINE_MODE.get() == DiagnosticPipelineMode::Production;
        let attention = if verified_attention {
            AttentionShaders::new(&raw, pipeline_layout)
        } else {
            AttentionShaders::default()
        };
        info["tiled_scores_available"] = json!(attention.scores.is_some());
        info["long_attention_available"] = json!(attention.values.is_some());
        let direct_experts = info["direct_expert_weights_enabled"]
            .as_bool()
            .unwrap_or(false);
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
            direct_pipeline,
            fast_decode,
            expert_word_dispatches: AtomicU64::new(0),
            router_dispatches: AtomicU64::new(0),
            #[cfg(test)]
            diagnostic_pipeline_mode: DIAGNOSTIC_PIPELINE_MODE.get(),
            info,
            budget,
            fast_math: options.fast_math,
            direct_experts,
            kv_precision: options.kv_precision,
            kv_placement: options.kv_placement,
            host_budget: if options.host_memory_budget_bytes > 0 {
                options.host_memory_budget_bytes as u64
            } else {
                crate::memory::available_memory()
                    .unwrap_or(budget)
                    .saturating_mul(7)
                    / 10
            },
            attention,
            host_allocated: AtomicU64::new(0),
            host_high_water: AtomicU64::new(0),
            allocated: AtomicU64::new(0),
            high_water: AtomicU64::new(0),
            allocations: AtomicUsize::new(0),
            uploads: AtomicU64::new(0),
            readbacks: AtomicU64::new(0),
            submissions: AtomicU64::new(0),
            dispatches: AtomicU64::new(0),
            group_compute_barriers: direct_experts && options.fast_math,
            grouped_dispatches: AtomicU64::new(0),
        }))
    }
    pub fn info(&self) -> Value {
        self.info.clone()
    }
    pub fn reserved(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }
    pub fn host_reserved(&self) -> u64 {
        self.host_allocated.load(Ordering::Relaxed)
    }
    pub fn stats(&self) -> Value {
        json!({"allocation_budget_bytes":self.budget,"live_allocation_bytes":self.reserved(),"high_water_allocation_bytes":self.high_water.load(Ordering::Relaxed),
        "group_compute_barriers":self.group_compute_barriers,"grouped_dispatches":self.grouped_dispatches.load(Ordering::Relaxed),
        "live_allocations":self.allocations.load(Ordering::Relaxed),"uploaded_bytes":self.uploads.load(Ordering::Relaxed),"readback_bytes":self.readbacks.load(Ordering::Relaxed),
        "host_allocation_budget_bytes":self.host_budget,"host_live_allocation_bytes":self.host_reserved(),
        "host_high_water_allocation_bytes":self.host_high_water.load(Ordering::Relaxed),
        "total_live_allocation_bytes":self.reserved().saturating_add(self.host_reserved()),
        "expert_word_dispatches":self.expert_word_dispatches.load(Ordering::Relaxed),
        "f32_router_dispatches":self.router_dispatches.load(Ordering::Relaxed),
        "direct_expert_weights_enabled":self.direct_experts,
        "direct_expert_weights_supported":self.info["direct_expert_weights_supported"],
        "kv_placement":self.kv_placement.as_str(),
        "tiled_score_dispatches":self.attention.score_dispatches.load(Ordering::Relaxed),
        "long_attention_dispatches":self.attention.value_dispatches.load(Ordering::Relaxed),
        "queue_submissions":self.submissions.load(Ordering::Relaxed),"compute_dispatches":self.dispatches.load(Ordering::Relaxed),
        "budget_is_owned_allocation_limit_not_free_vram":true,"software":self.info["software"],"fast_math":self.fast_math})
    }
    fn reserve_class(&self, class: AllocationClass, bytes: u64) -> Result<()> {
        let (live, high, budget) = match class {
            AllocationClass::Gpu => (&self.allocated, &self.high_water, self.budget),
            AllocationClass::Host => (
                &self.host_allocated,
                &self.host_high_water,
                self.host_budget,
            ),
        };
        let old = live
            .try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(bytes).filter(|n| *n <= budget)
            })
            .map_err(|old| {
                BudgetExceeded(format!(
                    "Vulkan {} budget exceeded: {old} live + {bytes} requested > {budget} bytes",
                    if class == AllocationClass::Gpu {
                        "owned-allocation"
                    } else {
                        "RAM allocation"
                    }
                ))
            })?;
        high.fetch_max(old + bytes, Ordering::Relaxed);
        Ok(())
    }
    fn release_class(&self, class: AllocationClass, bytes: u64) {
        match class {
            AllocationClass::Gpu => self.allocated.fetch_sub(bytes, Ordering::AcqRel),
            AllocationClass::Host => self.host_allocated.fetch_sub(bytes, Ordering::AcqRel),
        };
    }
    pub fn buffer(
        self: &Arc<Self>,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Arc<Buffer>> {
        self.buffer_placed(bytes, usage, host, false)
    }
    fn buffer_placed(
        self: &Arc<Self>,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        host: bool,
        ram_storage: bool,
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
        let is_software = self.info["software"].as_bool().unwrap_or(false);
        let memory_type = (0..self.memory.memory_type_count)
            .filter(|i| {
                let ty = self.memory.memory_types[*i as usize];
                let local = self.memory.memory_heaps[ty.heap_index as usize]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                req.memory_type_bits & (1u32 << i) != 0
                    && ty.property_flags.contains(required)
                    && (!ram_storage || !local || is_software)
                    && (host
                        || ty
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
            })
            .max_by_key(|i| {
                let ty = self.memory.memory_types[*i as usize];
                let local = self.memory.memory_heaps[ty.heap_index as usize]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                (
                    u32::from(host && !local),
                    (ty.property_flags & preferred).as_raw().count_ones(),
                )
            });
        let Some(memory_type) = memory_type else {
            unsafe { self.raw.destroy_buffer(raw, None) };
            bail!("no compatible Vulkan memory type (host visible={host})");
        };
        let heap = self.memory.memory_types[memory_type as usize].heap_index as usize;
        let system_ram = !self.memory.memory_heaps[heap]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            || (host && is_software);
        let class = if system_ram {
            AllocationClass::Host
        } else {
            AllocationClass::Gpu
        };
        if let Err(e) = self.reserve_class(class, req.size) {
            unsafe { self.raw.destroy_buffer(raw, None) };
            return Err(e);
        }
        let count = self.allocations.fetch_add(1, Ordering::AcqRel);
        if count >= self.properties.limits.max_memory_allocation_count as usize {
            self.allocations.fetch_sub(1, Ordering::AcqRel);
            self.release_class(class, req.size);
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
                self.release_class(class, req.size);
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
                self.release_class(class, req.size);
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
            class,
            system_ram,
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
    /// Host-visible storage on a genuine system-memory heap. Software Vulkan
    /// uses logical accounting because its single heap is already system RAM.
    pub fn host_storage(self: &Arc<Self>, bytes: usize) -> Result<Arc<Buffer>> {
        self.buffer_placed(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            true,
            true,
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
            if let Some(shaders) = &self.fast_decode { shaders.destroy(&self.raw); }
            if let Some(direct) = self.direct_pipeline { direct.destroy(&self.raw); }
            for pipeline in [self.attention.scores, self.attention.values].into_iter().flatten() {
                self.raw.destroy_pipeline(pipeline, None);
            }
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
    class: AllocationClass,
    system_ram: bool,
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
    pub fn is_system_ram(&self) -> bool {
        self.system_ram
    }
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
        self.device.release_class(self.class, self.allocated);
        self.device.allocations.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Push(pub [u32; 16]);
impl Push {
    fn is_word_expert(&self) -> bool {
        // Q8 activations, Direct expert weights, whole K blocks and aligned scales.
        self.0[0] == 0 && self.0[15] & (512 | 32) == (512 | 32)
            && matches!(self.0[3], 12 | 14) && self.0[1] % 256 == 0
            && self.0[11] % 32 == 0
    }

    fn is_f32_router(&self, groups: [u32; 3]) -> bool {
        // The shader reads exactly 128 contiguous rows of 2,048 F32 weights.
        self.0[..6] == [0, 2048, 128, 0, 8192, 1] && self.0[10..14] == [0; 4]
            && self.0[15] == 0 && groups == [128, 1, 1]
    }

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
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub array_weights: bool,
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
    bound_pipeline: Cell<vk::Pipeline>,
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
                descriptor_count: (sets as u32)
                    * (BINDINGS + if device.direct_experts { 7 } else { 0 }),
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
            bound_pipeline: Cell::new(vk::Pipeline::null()),
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
        let array_weights = push.0[0] == 0 && push.0[15] & 512 != 0;
        #[cfg(test)]
        let array_weights = array_weights || self.device.diagnostic_pipeline_mode != DiagnosticPipelineMode::Production;
        let mut compute = if array_weights {
            self.device.direct_pipeline.context("direct expert operation requires the array pipeline")?
        } else {
            ComputePipeline { set_layout: self.device.set_layout, pipeline_layout: self.device.pipeline_layout, pipeline: self.device.pipeline }
        };
        if let Some(shaders) = &self.device.fast_decode {
            if push.is_word_expert() {
                compute.pipeline = shaders.words;
            } else if push.is_f32_router(actual) {
                if let Some(router) = shaders.router {
                    compute.pipeline = router;
                }
            }
        }
        let layout = [compute.set_layout];
        let set = unsafe {
            self.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layout),
            )
        }?[0];
        self.sets_left -= 1;
        self.update_buffers(set, &buffers, array_weights);
        Ok(Operation {
            set,
            push,
            groups: actual,
            buffers,
            pipeline: compute.pipeline,
            pipeline_layout: compute.pipeline_layout,
            array_weights,
        })
    }
    pub fn update_set(&self, set: vk::DescriptorSet, buffers: &[Arc<Buffer>; 7]) {
        #[cfg(not(test))]
        let array_weights = false;
        #[cfg(test)]
        let array_weights = self.device.diagnostic_pipeline_mode != DiagnosticPipelineMode::Production;
        self.update_buffers(set, buffers, array_weights);
    }
    fn update_buffers(&self, set: vk::DescriptorSet, buffers: &[Arc<Buffer>; 7], array_weights: bool) {
        let infos: Vec<Vec<_>> = buffers
            .iter()
            .enumerate()
            .map(|(index, b)| {
                vec![
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.raw)
                        .offset(0)
                        .range(b.size);
                    if index == 0 && array_weights {
                        8
                    } else {
                        1
                    }
                ]
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
    /// Update a completed graph's dynamically-uniform expert array. Callers
    /// retain all source Arc leases until the consuming submission fence.
    pub(crate) fn update_weight_slots(
        &self,
        set: vk::DescriptorSet,
        weights: &[Arc<Buffer>],
    ) -> Result<()> {
        ensure!(
            self.device.direct_experts && !self.inflight,
            "direct weight descriptors require an idle compatible command owner"
        );
        ensure!(
            !weights.is_empty() && weights.len() <= 8,
            "direct expert array requires 1..=8 buffers"
        );
        ensure!(
            weights.iter().all(|b| Arc::ptr_eq(&b.device, &self.device)),
            "cross-device expert descriptor"
        );
        let infos: Vec<_> = (0..8)
            .map(|i| {
                let buffer = &weights[i.min(weights.len() - 1)];
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.raw)
                    .offset(0)
                    .range(buffer.size)
            })
            .collect();
        unsafe {
            self.device.raw.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos)],
                &[],
            )
        };
        Ok(())
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
        self.bound_pipeline.set(self.device.pipeline);
        Ok(())
    }
    pub fn record(&self, operation: &Operation, push: &Push, groups: [u32; 3]) {
        self.record_dispatch(operation, push, groups, true);
    }

    pub(crate) fn record_dispatch(
        &self, operation: &Operation, push: &Push, groups: [u32; 3], barrier_after: bool,
    ) {
        // SAFETY: Push contains exactly 16 initialized u32s, the pipeline expects
        // 64 bytes, and Vulkan consumes the values during this synchronous call.
        debug_assert!(!operation.array_weights || self.device.direct_experts);
        let bytes = unsafe { std::slice::from_raw_parts(push.0.as_ptr().cast::<u8>(), 64) };
        let (pipeline, groups) = if matches!(push.0[0], 4 | 5) && !operation.array_weights {
            self.device.attention.select(push, groups).unwrap_or((operation.pipeline, groups))
        } else {
            (operation.pipeline, groups)
        };
        unsafe {
            if self.bound_pipeline.get() != pipeline {
                self.device.raw.cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.bound_pipeline.set(pipeline);
            }
            self.device.raw.cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::COMPUTE,
                operation.pipeline_layout,
                0,
                &[operation.set],
                &[],
            );
            self.device.raw.cmd_push_constants(
                self.raw,
                operation.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            self.device
                .raw
                .cmd_dispatch(self.raw, groups[0], groups[1], groups[2]);
        }
        if let Some(shaders) = &self.device.fast_decode {
            if operation.pipeline == shaders.words {
                self.device.expert_word_dispatches.fetch_add(1, Ordering::Relaxed);
            } else if Some(operation.pipeline) == shaders.router {
                self.device.router_dispatches.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.device.dispatches.fetch_add(1, Ordering::Relaxed);
        if barrier_after {
            self.compute_barrier();
        } else {
            self.device.grouped_dispatches.fetch_add(1, Ordering::Relaxed);
        }
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

    /// CPU-only work can use the time between queue submission and fence completion.
    /// Its result is returned after the fence joins, including when host work fails.
    pub fn submit_with_host_work(&mut self, host_work: impl FnOnce() -> Result<()>) -> Result<()> {
        self.submit_and_wait(std::time::Duration::ZERO, host_work).map(|_| ())
    }

    pub(crate) fn submit_routing(
        &mut self, poll_budget: std::time::Duration,
    ) -> Result<(bool, std::time::Duration)> {
        self.submit_and_wait(poll_budget, || Ok(()))
    }

    fn submit_and_wait(
        &mut self, poll_budget: std::time::Duration, host_work: impl FnOnce() -> Result<()>,
    ) -> Result<(bool, std::time::Duration)> {
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
        let mut poll_time = std::time::Duration::ZERO;
        if !poll_budget.is_zero() {
            let start = std::time::Instant::now();
            loop {
                let signaled = unsafe { self.device.raw.get_fence_status(self.fence) }
                    .context("poll sparse routing fence")?;
                poll_time = start.elapsed();
                if signaled {
                    self.inflight = false;
                    return host_result.map(|()| (true, poll_time));
                }
                if poll_time >= poll_budget {
                    break;
                }
                std::hint::spin_loop();
            }
        }
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX)
        }
        .context("wait for independent Vulkan computation")?;
        self.inflight = false;
        host_result.map(|()| (false, poll_time))
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
            integer_dot_product: false,
            direct_experts_supported: false,
            integer_dot_product_accelerated: false,
            integer_dot_extension: false,
            heap_budgets: None,
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
    #[test]
    fn context_cache_layout_matches_exact_packed_storage() {
        assert_eq!(KvPrecision::F32.row_bytes(512).unwrap(), 2048);
        assert_eq!(KvPrecision::Q4_64.row_bytes(512).unwrap(), 272);
        assert_eq!(KvPrecision::Q4_64.row_bytes(1).unwrap(), 68);
        assert_eq!(KvPrecision::Q4_64.row_bytes(129).unwrap(), 136);
        assert_eq!(
            KvPrecision::Q4_64.row_bytes(512).unwrap() as u64 * 524288 * 48 * 2,
            13_690_208_256
        );
        assert!(KvPrecision::Q4_64.row_bytes(0).is_err());
        assert!(KvPrecision::Q4_64.row_bytes(usize::MAX).is_err());
        assert!(KvPrecision::F32.row_bytes(usize::MAX).is_err());
        assert_eq!(serde_json::to_value(KvPrecision::Q4_64).unwrap(), "q4_64");
        assert!(serde_json::from_value::<KvPrecision>(json!("unknown")).is_err());
    }

    #[test]
    fn diagnostic_pipeline_mode_is_thread_local_and_restores_after_unwind() {
        assert_eq!(DIAGNOSTIC_PIPELINE_MODE.get(),DiagnosticPipelineMode::Production);
        with_diagnostic_pipeline(DiagnosticPipelineMode::AllArrayDirect, || {
            assert_eq!(DIAGNOSTIC_PIPELINE_MODE.get(),DiagnosticPipelineMode::AllArrayDirect);
            let caught=std::panic::catch_unwind(|| with_diagnostic_pipeline(DiagnosticPipelineMode::AllArrayPacked, || panic!("restore test")));
            assert!(caught.is_err());
            assert_eq!(DIAGNOSTIC_PIPELINE_MODE.get(),DiagnosticPipelineMode::AllArrayDirect);
        });
        assert_eq!(DIAGNOSTIC_PIPELINE_MODE.get(),DiagnosticPipelineMode::Production);
    }

    #[test]
    fn forced_direct_access_filters_unsupported_adapters() {
        let mut adapter=adapter(0x1002,"Test hardware",vk::PhysicalDeviceType::DISCRETE_GPU,8192);
        let mut options=DeviceOptions::default();assert_eq!(options.direct_experts,None);
        options.direct_experts=Some(true);
        assert!(choose_adapter(std::slice::from_ref(&adapter),&options).unwrap_err().to_string().contains("direct expert"));
        adapter.direct_experts_supported=true;
        assert_eq!(choose_adapter(std::slice::from_ref(&adapter),&options).unwrap(),Some(0));
        assert_eq!(describe(&adapter,0,&options)["direct_expert_weights_supported"],true);
        adapter.direct_experts_supported=false;options.direct_experts=Some(false);
        assert_eq!(choose_adapter(std::slice::from_ref(&adapter),&options).unwrap(),Some(0));
    }

    #[test]
    fn fast_math_is_opt_in_and_filters_incompatible_adapters() {
        let unsupported = adapter(
            0x1234,
            "Baseline adapter",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8192,
        );
        let mut supported = adapter(
            0x1234,
            "Integer dot adapter",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            8192,
        );
        supported.integer_dot_product = true;
        supported.integer_dot_product_accelerated = true;
        supported.integer_dot_extension = true;
        let mut options = DeviceOptions::default();
        assert!(!options.fast_math);
        assert_eq!(
            choose_adapter(&[unsupported, supported], &options).unwrap(),
            Some(0)
        );
        let unsupported = adapter(
            0x1234,
            "Baseline adapter",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8192,
        );
        options.fast_math = true;
        assert!(
            choose_adapter(&[unsupported], &options)
                .unwrap_err()
                .to_string()
                .contains("fast math")
        );
        let mut supported = adapter(
            0x1234,
            "Integer dot adapter",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            8192,
        );
        supported.integer_dot_product = true;
        let report = describe(&supported, 0, &options);
        assert_eq!(report["fast_math_available"], true);
        assert_eq!(report["fast_math"], true);
        assert_eq!(report["compatible"], true);
    }

    #[test]
    fn ram_placement_requires_system_memory_and_physical_budget_is_clamped() {
        let mut candidate = adapter(
            0x1002,
            "Discrete",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8192,
        );
        let mut options = DeviceOptions::default();
        options.kv_placement = KvPlacement::Ram;
        assert!(!candidate.has_ram_storage_memory());
        assert!(
            candidate
                .rejection_reasons(&options)
                .iter()
                .any(|reason| reason.contains("system-memory"))
        );
        candidate.memory.memory_heap_count = 2;
        candidate.memory.memory_heaps[1] = vk::MemoryHeap {
            size: 32 * 1024 * MIB,
            flags: vk::MemoryHeapFlags::empty(),
        };
        candidate.memory.memory_type_count = 2;
        candidate.memory.memory_types[1] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_CACHED,
            heap_index: 1,
        };
        assert!(candidate.has_ram_storage_memory());
        candidate.heap_budgets = Some(vec![(16 * 1024 * MIB, 1024 * MIB), (32 * 1024 * MIB, 0)]);
        options.memory_budget_bytes = 8 * 1024 * MIB as usize;
        assert_eq!(candidate.fit_budget(&options), (8192 - 1024 - 256) * MIB);
        let info = describe(&candidate, 0, &options);
        assert_eq!(
            info["physical_heap_budget_snapshot"][0]["budget_bytes"],
            8192 * MIB
        );
        assert_eq!(
            info["physical_heap_budget_snapshot"][0]["available_bytes"],
            7168 * MIB
        );
        assert_eq!(info["allocation_budget_bytes"], 8192 * MIB);
    }

    #[test]
    #[ignore = "requires explicit Vulkan RAM storage execution"]
    fn software_vulkan_ram_and_gpu_allocation_ledgers_are_independent_and_rollback() {
        assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
        let device = Device::new(&DeviceOptions {
            allow_software: true,
            memory_budget_bytes: MIB as usize,
            host_memory_budget_bytes: (4 * MIB) as usize,
            kv_placement: KvPlacement::Ram,
            ..Default::default()
        })
        .unwrap();
        let ram = device.host_storage((2 * MIB) as usize).unwrap();
        assert!(ram.is_system_ram());
        assert_eq!(device.reserved(), 0);
        let host_live = device.host_reserved();
        assert!(host_live >= 2 * MIB);
        assert!(device.storage((2 * MIB) as usize).is_err());
        assert_eq!(device.host_reserved(), host_live);
        assert_eq!(device.reserved(), 0);
        let gpu = device.storage((MIB / 2) as usize).unwrap();
        let gpu_live = device.reserved();
        assert!(gpu_live >= MIB / 2);
        assert!(device.host_storage((3 * MIB) as usize).is_err());
        assert_eq!(device.reserved(), gpu_live);
        assert_eq!(device.host_reserved(), host_live);
        drop(ram);
        drop(gpu);
        assert_eq!(device.reserved(), 0);
        assert_eq!(device.host_reserved(), 0);
        assert_eq!(device.stats()["live_allocations"], 0);
    }
}
