//! Read-only Vulkan discovery. No logical device, GPU allocations, kernels, HIP
//! runtime, package-manager commands, or persistent configuration are touched.
//!
//! `backend_index` mirrors `rusting_strata::vulkan::Vk::new`: filter by its
//! `unusable` predicate, then stable-sort discrete, integrated, and other GPUs.
//! `enumeration_index` preserves the Vulkan loader's original device order.

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const AMD_VENDOR: u32 = 0x1002;
const NOT_TESTED: &str = "not_tested: discovery only; no compute was submitted";

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub schema_version: u32,
    pub operating_system: String,
    pub architecture: String,
    pub native_windows: bool,
    pub wsl: bool,
    pub kernel_release: Option<String>,
    pub dxg_present: bool,
    pub radv_device_coherent_memory: Option<String>,
    pub rocm_metadata: Vec<RocmMetadata>,
    pub vulkan_loader_version: Option<String>,
    pub vulkan_error: Option<String>,
    pub adapters: Vec<AdapterReport>,
    pub compute_status: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RocmMetadata {
    pub metadata_path: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCheck {
    pub name: String,
    pub available: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SubgroupReport {
    pub default_size: u32,
    pub minimum_size: u32,
    pub maximum_size: u32,
    pub can_require_size_in_compute: bool,
    pub compute_supported: bool,
    pub arithmetic: bool,
    pub shuffle: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryHeapReport {
    pub index: usize,
    pub device_local: bool,
    pub size_bytes: u64,
    /// Driver estimates for this process, not reserved or guaranteed capacity.
    pub budget_bytes: Option<u64>,
    pub process_usage_bytes: Option<u64>,
    pub estimated_headroom_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdapterReport {
    pub enumeration_index: usize,
    pub backend_index: Option<usize>,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub api_version: String,
    pub driver_name: String,
    pub driver_info: String,
    pub is_rx_9070_xt: bool,
    pub software_adapter: bool,
    pub vulkan_1_3: bool,
    pub required_features: Vec<FeatureCheck>,
    pub subgroup: SubgroupReport,
    pub compute_queue_family: Option<u32>,
    pub max_compute_shared_memory_bytes: u32,
    pub max_compute_workgroup_invocations: u32,
    pub memory_budget_extension: bool,
    pub amd_coherent_memory_extension: bool,
    pub amd_coherent_memory_feature: bool,
    pub coherent_host_visible_device_memory: bool,
    pub memory_heaps: Vec<MemoryHeapReport>,
    pub upstream_filter_passes: bool,
    pub missing_requirements: Vec<String>,
    pub warnings: Vec<String>,
    pub compute_status: String,
    #[serde(skip)]
    device_sort_key: u8,
}

impl DoctorReport {
    /// Select the exact RX 9070 XT by the engine's filtered/sorted device index.
    /// This confirms discovery prerequisites, not successful GPU computation.
    pub fn select_9070_xt(&self, requested: Option<usize>) -> Result<usize> {
        if let Some(error) = &self.vulkan_error {
            bail!("Vulkan discovery failed: {error}");
        }
        if let Some(index) = requested {
            let adapter = self
                .adapters
                .iter()
                .find(|a| a.backend_index == Some(index))
                .ok_or_else(|| anyhow!("Vulkan backend device {index} is unavailable; run doctor to see backend_index values"))?;
            validate_selection(adapter)?;
            return Ok(index);
        }

        let matches: Vec<_> = self
            .adapters
            .iter()
            .filter(|a| a.is_rx_9070_xt && validate_selection(a).is_ok())
            .collect();
        match matches.as_slice() {
            [adapter] => adapter.backend_index.ok_or_else(|| anyhow!("selected adapter has no backend index")),
            [] => {
                let details: Vec<_> = self.adapters.iter().filter(|a| a.is_rx_9070_xt)
                    .map(|a| format!("{}: {}", a.name, validate_selection(a).err().map_or_else(|| "unavailable".to_owned(), |e| e.to_string())))
                    .collect();
                if details.is_empty() {
                    bail!("No hardware AMD Radeon RX 9070 XT was identified by Vulkan. Run doctor; a CPU/software adapter or a gfx1201/Navi48 name alone does not identify this exact model.");
                }
                bail!("RX 9070 XT discovery prerequisites failed: {}", details.join("; "));
            }
            _ => bail!("Multiple eligible RX 9070 XT adapters were found; select one with --device and its doctor backend_index"),
        }
    }

    pub fn print_human(&self) {
        println!("Forge 9070 hardware doctor (read-only)");
        println!("Platform: {} / {}{}", self.operating_system, self.architecture,
            if self.wsl { " / WSL" } else if self.native_windows { " / native Windows" } else { "" });
        if let Some(kernel) = &self.kernel_release { println!("Kernel: {kernel}"); }
        if self.wsl { println!("WSL /dev/dxg: {}", yes_no(self.dxg_present)); }
        for metadata in &self.rocm_metadata {
            println!("ROCm metadata: {} ({})", metadata.version, metadata.metadata_path);
        }
        if self.rocm_metadata.is_empty() { println!("ROCm metadata: not found (not required by the Vulkan backend)"); }
        println!("Vulkan loader: {}", self.vulkan_loader_version.as_deref().unwrap_or("unavailable"));
        if let Some(error) = &self.vulkan_error { println!("Vulkan discovery error: {error}"); }
        for adapter in &self.adapters {
            println!("\nAdapter {}: {}", adapter.enumeration_index, adapter.name);
            println!("  Backend index: {} | {} | vendor 0x{:04x}, device 0x{:04x}",
                adapter.backend_index.map_or_else(|| "unavailable".to_owned(), |i| i.to_string()),
                adapter.device_type, adapter.vendor_id, adapter.device_id);
            println!("  Driver: {} {} | Vulkan {}", adapter.driver_name, adapter.driver_info, adapter.api_version);
            println!("  Exact RX 9070 XT: {} | software adapter: {}", yes_no(adapter.is_rx_9070_xt), yes_no(adapter.software_adapter));
            println!("  Compute queue: {} | subgroup {} ({}..={}), size control in compute: {}",
                adapter.compute_queue_family.map_or_else(|| "none".to_owned(), |i| i.to_string()),
                adapter.subgroup.default_size, adapter.subgroup.minimum_size, adapter.subgroup.maximum_size,
                yes_no(adapter.subgroup.can_require_size_in_compute));
            println!("  Subgroup compute/arithmetic/shuffle: {}/{}/{}",
                yes_no(adapter.subgroup.compute_supported), yes_no(adapter.subgroup.arithmetic), yes_no(adapter.subgroup.shuffle));
            println!("  Required features: {}", adapter.required_features.iter()
                .map(|f| format!("{}={}", f.name, yes_no(f.available))).collect::<Vec<_>>().join(", "));
            println!("  Shared memory/workgroup: {} bytes | max workgroup invocations: {}",
                adapter.max_compute_shared_memory_bytes, adapter.max_compute_workgroup_invocations);
            println!("  AMD coherent memory extension/feature/host-visible device memory: {}/{}/{}",
                yes_no(adapter.amd_coherent_memory_extension), yes_no(adapter.amd_coherent_memory_feature),
                yes_no(adapter.coherent_host_visible_device_memory));
            for heap in &adapter.memory_heaps {
                print!("  Heap {}{}: {:.2} GiB", heap.index, if heap.device_local { " device-local" } else { " host/shared" }, gib(heap.size_bytes));
                if let (Some(budget), Some(usage), Some(headroom)) = (heap.budget_bytes, heap.process_usage_bytes, heap.estimated_headroom_bytes) {
                    print!("; driver budget {:.2}, process use {:.2}, estimated headroom {:.2} GiB", gib(budget), gib(usage), gib(headroom));
                }
                println!();
            }
            if adapter.missing_requirements.is_empty() {
                println!("  Discovery prerequisites: present; compute remains untested");
            } else {
                println!("  Missing/blocked: {}", adapter.missing_requirements.join("; "));
            }
            for warning in &adapter.warnings { println!("  Note: {warning}"); }
        }
        println!("\nCompute status: {}", self.compute_status);
        for note in &self.notes { println!("Note: {note}"); }
    }
}

/// Collect metadata and query physical devices. This deliberately does not call
/// `Vk::new`, which allocates GPU buffers and mutates the RADV environment.
pub fn collect() -> DoctorReport {
    let kernel_release = read_small_file(Path::new("/proc/sys/kernel/osrelease"));
    // A native Windows .exe launched from WSL can inherit WSL_* variables;
    // its Vulkan loader is still the native Windows one.
    let wsl = cfg!(target_os = "linux") && (
        kernel_release.as_deref().is_some_and(|k| k.to_ascii_lowercase().contains("microsoft"))
        || std::env::var_os("WSL_INTEROP").is_some()
        || std::env::var_os("WSL_DISTRO_NAME").is_some());
    let mut report = DoctorReport {
        schema_version: 1,
        operating_system: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        native_windows: cfg!(target_os = "windows"),
        wsl,
        kernel_release,
        dxg_present: Path::new("/dev/dxg").exists(),
        radv_device_coherent_memory: std::env::var("radv_device_coherent_memory").ok(),
        rocm_metadata: rocm_metadata(),
        vulkan_loader_version: None,
        vulkan_error: None,
        adapters: Vec::new(),
        compute_status: NOT_TESTED.to_owned(),
        notes: vec![
            "ROCm/HIP was not initialized. Metadata does not establish GPU support; the run path uses Vulkan.".to_owned(),
            "Discovery does not verify kernel execution, model compatibility, or inference speed.".to_owned(),
            "Memory budgets are changing driver estimates, not reserved memory or a safe cache-allocation promise.".to_owned(),
        ],
    };
    if wsl {
        report.notes.push("WSL /dev/dxg presence alone does not establish Vulkan support. A real Vulkan adapter with the required features must be exposed.".to_owned());
        if report.rocm_metadata.iter().any(|m| m.version.starts_with("6.3.4")) {
            report.notes.push("ROCm 6.3.4 does not provide an officially supported RX 9070 XT WSL configuration. This Vulkan doctor does not change that ROCm installation.".to_owned());
        }
    }
    if report.radv_device_coherent_memory.is_none() {
        report.notes.push("radv_device_coherent_memory is unset. This doctor leaves it unchanged; the upstream backend sets it to true before initialization, so its coherent-memory discovery may differ.".to_owned());
    }
    if let Err(error) = collect_vulkan(&mut report) {
        report.vulkan_error = Some(format!("{error:#}"));
    }
    report
}

struct InstanceGuard {
    instance: ash::Instance,
    // Keep the dynamic loader alive until after the instance is destroyed.
    _entry: ash::Entry,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the instance, no logical device is
        // created, and all physical-device queries have returned before drop.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn collect_vulkan(report: &mut DoctorReport) -> Result<()> {
    // SAFETY: ash loads the platform Vulkan library; its lifetime is retained by
    // InstanceGuard, and no driver entry point outlives the guard.
    let entry = unsafe { ash::Entry::load() }.context("load the Vulkan loader")?;
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .context("query Vulkan loader version")?.unwrap_or(vk::API_VERSION_1_0);
    report.vulkan_loader_version = Some(version_string(loader_version));
    let app = vk::ApplicationInfo::default().api_version(loader_version.min(vk::API_VERSION_1_3));
    let instance = unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
    }.context("create a Vulkan discovery instance")?;
    let guard = InstanceGuard { instance, _entry: entry };
    let physical_devices = unsafe { guard.instance.enumerate_physical_devices() }
        .context("enumerate Vulkan physical devices")?;
    for (enumeration_index, device) in physical_devices.into_iter().enumerate() {
        report.adapters.push(inspect_adapter(&guard.instance, device, enumeration_index, loader_version));
    }
    assign_backend_indices(&mut report.adapters);
    if report.adapters.is_empty() {
        report.notes.push("The Vulkan loader found no physical devices.".to_owned());
    }
    Ok(())
}

fn inspect_adapter(instance: &ash::Instance, device: vk::PhysicalDevice, index: usize, loader_version: u32) -> AdapterReport {
    // SAFETY: device was returned by this live instance; all output structures
    // are correctly initialized and their pNext storage remains alive.
    let p = unsafe { instance.get_physical_device_properties(device) };
    let name = p.device_name_as_c_str().map_or_else(|_| "<invalid device name>".to_owned(), |s| s.to_string_lossy().into_owned());
    let modern = loader_version >= vk::API_VERSION_1_3 && p.api_version >= vk::API_VERSION_1_3;
    let queue = unsafe { instance.get_physical_device_queue_family_properties(device) }
        .iter().position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE)).map(|i| i as u32);
    let (extensions, extension_error) = match unsafe { instance.enumerate_device_extension_properties(device) } {
        Ok(v) => (v, None),
        Err(e) => (Vec::new(), Some(format!("Could not enumerate device extensions: {e:?}"))),
    };
    let has = |n: &std::ffi::CStr| extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(n));
    let budget_ext = has(ash::ext::memory_budget::NAME);
    let coherent_ext = has(ash::amd::device_coherent_memory::NAME);
    let mut subgroup = SubgroupReport::default();
    let mut driver_name = String::new();
    let mut driver_info = String::new();
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut basic = vk::PhysicalDeviceFeatures::default();
    let mut coherent_feature = false;
    if modern {
        let mut sg = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut sg).push_next(&mut p11).push_next(&mut driver);
        unsafe { instance.get_physical_device_properties2(device, &mut properties) };
        subgroup = SubgroupReport {
            default_size: p11.subgroup_size,
            minimum_size: sg.min_subgroup_size,
            maximum_size: sg.max_subgroup_size,
            can_require_size_in_compute: sg.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE),
            compute_supported: p11.subgroup_supported_stages.contains(vk::ShaderStageFlags::COMPUTE),
            arithmetic: p11.subgroup_supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC),
            shuffle: p11.subgroup_supported_operations.contains(vk::SubgroupFeatureFlags::SHUFFLE),
        };
        driver_name = driver.driver_name_as_c_str().map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned());
        driver_info = driver.driver_info_as_c_str().map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned());
        let mut f = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
        unsafe { instance.get_physical_device_features2(device, &mut f) };
        basic = f.features;
        if coherent_ext {
            let mut coherent = vk::PhysicalDeviceCoherentMemoryFeaturesAMD::default();
            unsafe { instance.get_physical_device_features2(device, &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut coherent)) };
            coherent_feature = coherent.device_coherent_memory == vk::TRUE;
        }
    }
    let feature_values = [
        ("shaderInt16", basic.shader_int16), ("shaderInt64", basic.shader_int64),
        ("shaderFloat64", basic.shader_float64), ("storageBuffer16BitAccess", f11.storage_buffer16_bit_access),
        ("bufferDeviceAddress", f12.buffer_device_address), ("storageBuffer8BitAccess", f12.storage_buffer8_bit_access),
        ("shaderInt8", f12.shader_int8), ("shaderFloat16", f12.shader_float16),
        ("vulkanMemoryModel", f12.vulkan_memory_model), ("vulkanMemoryModelDeviceScope", f12.vulkan_memory_model_device_scope),
        ("subgroupSizeControl", f13.subgroup_size_control), ("computeFullSubgroups", f13.compute_full_subgroups),
        ("shaderIntegerDotProduct", f13.shader_integer_dot_product),
    ];
    let mut required_features: Vec<_> = feature_values.into_iter().map(|(name, available)| FeatureCheck {
        name: name.to_owned(), available: available == vk::TRUE,
    }).collect();
    // Vk::new enables timelineSemaphore, although upstream unusable() does not
    // check it. Preserve its indexing but reject it in our stronger selection.
    let filter_features = required_features.iter().all(|f| f.available);
    required_features.push(FeatureCheck { name: "timelineSemaphore".to_owned(), available: f12.timeline_semaphore == vk::TRUE });
    let subgroup_size_ok = subgroup_size_supported(&subgroup);
    let subgroup_ops_ok = subgroup.compute_supported && subgroup.arithmetic && subgroup.shuffle;
    let upstream_filter_passes = p.device_type != vk::PhysicalDeviceType::CPU && modern
        && subgroup_size_ok && subgroup_ops_ok && filter_features;
    let software = is_software(p.device_type == vk::PhysicalDeviceType::CPU, &name, &driver_name);
    let mut missing = Vec::new();
    if software { missing.push("CPU/software adapters are not accepted".to_owned()); }
    if !modern { missing.push("Vulkan 1.3 in both loader and physical device is required; newer feature queries were skipped".to_owned()); }
    if !subgroup_size_ok { missing.push("compute subgroup size 32, or a fixed 64-lane subgroup, is required".to_owned()); }
    if !subgroup_ops_ok { missing.push("subgroup arithmetic and shuffle in compute shaders are required".to_owned()); }
    for f in &required_features { if !f.available { missing.push(format!("missing {}", f.name)); } }
    if queue.is_none() { missing.push("no compute queue family".to_owned()); }
    if let Some(error) = extension_error { missing.push(error); }
    let mut memory = unsafe { instance.get_physical_device_memory_properties(device) };
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let can_query_budget = budget_ext && loader_version >= vk::API_VERSION_1_1;
    if can_query_budget {
        let mut m = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { instance.get_physical_device_memory_properties2(device, &mut m) };
        memory = m.memory_properties;
    }
    let coherent_host_device = memory.memory_types[..memory.memory_type_count as usize].iter().any(|m| {
        m.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD)
    });
    let memory_heaps = memory.memory_heaps[..memory.memory_heap_count as usize].iter().enumerate().map(|(i, heap)| MemoryHeapReport {
        index: i,
        device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
        size_bytes: heap.size,
        budget_bytes: can_query_budget.then_some(budget.heap_budget[i]),
        process_usage_bytes: can_query_budget.then_some(budget.heap_usage[i]),
        estimated_headroom_bytes: can_query_budget.then_some(budget.heap_budget[i].saturating_sub(budget.heap_usage[i])),
    }).collect();
    let mut warnings = Vec::new();
    if p.vendor_id == AMD_VENDOR && !(coherent_ext && coherent_feature) {
        warnings.push("AMD device-coherent memory is unavailable in this snapshot. Upstream warns that CPU-expert mailbox work can stall or be slow; discovery alone cannot validate that path.".to_owned());
    }
    if coherent_ext && coherent_feature && !coherent_host_device {
        warnings.push("The AMD coherent-memory feature is exposed, but no host-visible coherent device-local memory type was identified.".to_owned());
    }
    if !can_query_budget { warnings.push("No usable VK_EXT_memory_budget query; heap size is capacity, not free VRAM.".to_owned()); }
    AdapterReport {
        enumeration_index: index, backend_index: None,
        is_rx_9070_xt: is_rx_9070_xt(p.vendor_id, &name),
        software_adapter: software, name, vendor_id: p.vendor_id, device_id: p.device_id,
        device_type: format!("{:?}", p.device_type), api_version: version_string(p.api_version),
        driver_name, driver_info, vulkan_1_3: modern, required_features, subgroup,
        compute_queue_family: queue, max_compute_shared_memory_bytes: p.limits.max_compute_shared_memory_size,
        max_compute_workgroup_invocations: p.limits.max_compute_work_group_invocations,
        memory_budget_extension: budget_ext, amd_coherent_memory_extension: coherent_ext,
        amd_coherent_memory_feature: coherent_feature, coherent_host_visible_device_memory: coherent_host_device,
        memory_heaps, upstream_filter_passes, missing_requirements: missing, warnings,
        compute_status: NOT_TESTED.to_owned(),
        device_sort_key: match p.device_type { vk::PhysicalDeviceType::DISCRETE_GPU => 0, vk::PhysicalDeviceType::INTEGRATED_GPU => 1, _ => 2 },
    }
}

fn assign_backend_indices(adapters: &mut [AdapterReport]) {
    for a in adapters.iter_mut() { a.backend_index = None; }
    let mut eligible: Vec<_> = adapters.iter().enumerate().filter(|(_, a)| a.upstream_filter_passes)
        .map(|(position, a)| (position, a.device_sort_key)).collect();
    eligible.sort_by_key(|(_, key)| *key); // stable, exactly as upstream
    for (backend_index, (position, _)) in eligible.into_iter().enumerate() {
        adapters[position].backend_index = Some(backend_index);
    }
}

fn validate_selection(adapter: &AdapterReport) -> Result<()> {
    if adapter.software_adapter { bail!("{} is a CPU/software adapter", adapter.name); }
    if !adapter.is_rx_9070_xt { bail!("{} is not an identified AMD Radeon RX 9070 XT", adapter.name); }
    if !adapter.missing_requirements.is_empty() { bail!("{}", adapter.missing_requirements.join("; ")); }
    if adapter.backend_index.is_none() { bail!("{} did not pass the upstream device filter", adapter.name); }
    Ok(())
}

fn subgroup_size_supported(s: &SubgroupReport) -> bool {
    (s.minimum_size <= 32 && s.maximum_size >= 32 && s.can_require_size_in_compute)
        || (s.minimum_size == 64 && s.maximum_size == 64 && s.default_size == 64)
}

fn is_rx_9070_xt(vendor_id: u32, name: &str) -> bool {
    if vendor_id != AMD_VENDOR { return false; }
    let upper = name.to_ascii_uppercase();
    let words: Vec<_> = upper.split(|c: char| !c.is_ascii_alphanumeric()).filter(|s| !s.is_empty()).collect();
    words.contains(&"RX9070XT")
        || words.windows(2).any(|w| w == ["RX9070", "XT"] || w == ["RX", "9070XT"])
        || words.windows(3).any(|w| w == ["RX", "9070", "XT"])
}

fn is_software(cpu: bool, name: &str, driver: &str) -> bool {
    let text = format!("{name} {driver}").to_ascii_lowercase();
    cpu || ["llvmpipe", "lavapipe", "swiftshader", "microsoft basic render"].iter().any(|s| text.contains(s))
}

fn read_small_file(path: &Path) -> Option<String> {
    let mut text = String::new();
    File::open(path).ok()?.take(4096).read_to_string(&mut text).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn rocm_metadata() -> Vec<RocmMetadata> {
    let mut roots = BTreeSet::from([PathBuf::from("/opt/rocm")]);
    for variable in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(path) = std::env::var_os(variable) { roots.insert(PathBuf::from(path)); }
    }
    if let Ok(entries) = std::fs::read_dir("/opt") {
        for entry in entries.flatten().take(256) {
            if entry.file_name().to_string_lossy().starts_with("rocm-") { roots.insert(entry.path()); }
        }
    }
    let mut seen = BTreeSet::new();
    roots.into_iter().filter_map(|root| {
        let path = root.join(".info/version");
        let resolved = path.canonicalize().unwrap_or(path);
        if !seen.insert(resolved.clone()) { return None; }
        read_small_file(&resolved).map(|version| RocmMetadata { metadata_path: resolved.display().to_string(), version })
    }).collect()
}

fn version_string(version: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(version), vk::api_version_minor(version), vk::api_version_patch(version))
}
fn yes_no(value: bool) -> &'static str { if value { "yes" } else { "no" } }
fn gib(bytes: u64) -> f64 { bytes as f64 / (1024.0 * 1024.0 * 1024.0) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_card_matching_accepts_driver_spellings_but_not_sibling_cards() {
        for name in ["AMD Radeon RX 9070 XT", "AMD Radeon RX 9070 XT (RADV NAVI48)", "Radeon RX9070XT", "AMD Radeon RX9070 XT", "radeon rx 9070xt"] {
            assert!(is_rx_9070_xt(AMD_VENDOR, name), "{name}");
        }
        for name in ["AMD Radeon RX 9070", "AMD Radeon RX 9070 XTX", "AMD Radeon RX 9070 XTX (RADV)", "AMD Radeon RX 9070 GRE", "gfx1201", "NAVI48", "AMD Radeon AI PRO R9700"] {
            assert!(!is_rx_9070_xt(AMD_VENDOR, name), "{name}");
        }
        assert!(!is_rx_9070_xt(0x10de, "AMD Radeon RX 9070 XT"));
    }

    #[test]
    fn software_detection_does_not_reject_amd_software_branding() {
        assert!(is_software(false, "llvmpipe (LLVM)", "Mesa"));
        assert!(is_software(false, "AMD Radeon RX 9070 XT", "SwiftShader"));
        assert!(is_software(true, "some device", "driver"));
        assert!(!is_software(false, "AMD Radeon RX 9070 XT", "AMD Software Adrenalin"));
    }

    #[test]
    fn subgroup_support_matches_wave32_and_fixed_wave64_paths() {
        let mut s = SubgroupReport { default_size: 32, minimum_size: 32, maximum_size: 64, can_require_size_in_compute: true, ..Default::default() };
        assert!(subgroup_size_supported(&s));
        s.can_require_size_in_compute = false;
        assert!(!subgroup_size_supported(&s));
        s.minimum_size = 64; s.maximum_size = 64; s.default_size = 64;
        assert!(subgroup_size_supported(&s));
        s.default_size = 32;
        assert!(!subgroup_size_supported(&s));
    }

    fn adapter(index: usize, sort: u8, usable: bool, name: &str) -> AdapterReport {
        AdapterReport {
            enumeration_index: index, backend_index: None, name: name.to_owned(), vendor_id: AMD_VENDOR, device_id: 0,
            device_type: "TEST".to_owned(), api_version: "1.3.0".to_owned(), driver_name: String::new(), driver_info: String::new(),
            is_rx_9070_xt: is_rx_9070_xt(AMD_VENDOR, name), software_adapter: false, vulkan_1_3: true,
            required_features: Vec::new(), subgroup: SubgroupReport::default(), compute_queue_family: Some(0),
            max_compute_shared_memory_bytes: 65536, max_compute_workgroup_invocations: 1024,
            memory_budget_extension: false, amd_coherent_memory_extension: false, amd_coherent_memory_feature: false,
            coherent_host_visible_device_memory: false, memory_heaps: Vec::new(), upstream_filter_passes: usable,
            missing_requirements: Vec::new(), warnings: Vec::new(), compute_status: NOT_TESTED.to_owned(), device_sort_key: sort,
        }
    }

    #[test]
    fn backend_indices_filter_then_stably_prioritize_discrete_adapters() {
        let mut adapters = vec![adapter(0, 1, true, "iGPU"), adapter(1, 2, false, "software"),
            adapter(2, 0, true, "AMD Radeon RX 9070 XT"), adapter(3, 0, true, "second discrete")];
        assign_backend_indices(&mut adapters);
        assert_eq!(adapters.iter().map(|a| a.backend_index).collect::<Vec<_>>(), vec![Some(2), None, Some(0), Some(1)]);
        assert_eq!(adapters.iter().map(|a| a.enumeration_index).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn selection_refuses_missing_features_and_software_even_with_matching_name() {
        let mut a = adapter(0, 0, true, "AMD Radeon RX 9070 XT");
        a.backend_index = Some(0);
        assert!(validate_selection(&a).is_ok());
        a.missing_requirements.push("missing timelineSemaphore".to_owned());
        assert!(validate_selection(&a).is_err());
        a.missing_requirements.clear(); a.software_adapter = true;
        assert!(validate_selection(&a).is_err());
        a.software_adapter = false; a.is_rx_9070_xt = false;
        assert!(validate_selection(&a).is_err());
    }

    #[test]
    fn public_selection_uses_backend_indices_and_requires_unambiguous_target() {
        let mut adapters = vec![adapter(0, 1, true, "AMD integrated GPU"),
            adapter(1, 0, true, "AMD Radeon RX 9070 XT")];
        assign_backend_indices(&mut adapters);
        let mut report = DoctorReport {
            schema_version: 1, operating_system: "test".to_owned(), architecture: "test".to_owned(),
            native_windows: false, wsl: false, kernel_release: None, dxg_present: false,
            radv_device_coherent_memory: None, rocm_metadata: Vec::new(), vulkan_loader_version: Some("1.3.0".to_owned()),
            vulkan_error: None, adapters, compute_status: NOT_TESTED.to_owned(), notes: Vec::new(),
        };
        assert_eq!(report.select_9070_xt(None).unwrap(), 0);
        assert_eq!(report.select_9070_xt(Some(0)).unwrap(), 0);
        assert!(report.select_9070_xt(Some(1)).is_err());
        report.adapters.push(adapter(2, 0, true, "AMD Radeon RX 9070 XT"));
        assign_backend_indices(&mut report.adapters);
        assert!(report.select_9070_xt(None).is_err());
        assert_eq!(report.select_9070_xt(Some(1)).unwrap(), 1);
        report.vulkan_error = Some("incomplete discovery".to_owned());
        assert!(report.select_9070_xt(Some(0)).is_err());
    }
}
