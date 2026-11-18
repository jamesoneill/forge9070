//! Host capacity and available-memory budgets for the owned expert tier.
//!
//! Avalon separates expert residency from its large sparse embedding table.
//! These budgets apply that policy to pageable Rust storage. They are startup
//! ceilings, not OS reservations or HIP/KFD driver registration allowances.

use anyhow::Context;
use serde::Serialize;

pub const MIB: u64 = 1 << 20;
/// Additional room for staging, tokenizer/work buffers and cache bookkeeping.
/// The user's OS margin and prompt-state cache reserve are separate.
pub const EXPERT_RUNTIME_RESERVE: u64 = 1024 * MIB;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MemorySnapshot {
    pub physical_total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    pub cgroup_headroom_bytes: Option<u64>,
}

fn min_known(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) { (Some(a), Some(b)) => Some(a.min(b)), (a, b) => a.or(b) }
}

fn meminfo_bytes(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut words = line.strip_prefix(key)?.split_whitespace();
        let kb = words.next()?.parse::<u64>().ok()?;
        (words.next()? == "kB").then_some(())?;
        kb.checked_mul(1024)
    })
}

#[cfg(target_os = "linux")]
fn cgroup_limits() -> (Option<u64>, Option<u64>) {
    use std::path::{Path, PathBuf};
    fn unescape(s: &str) -> String {
        s.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")
    }
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let Some(rel) = membership.lines().find_map(|line| line.strip_prefix("0::")) else { return (None, None) };
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").unwrap_or_default();
    let mut result = (None, None);
    for line in mounts.lines().filter(|line| line.contains(" - cgroup2 ")) {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 6 { continue; }
        let root = PathBuf::from(unescape(fields[3]));
        let mount = PathBuf::from(unescape(fields[4]));
        let relative = Path::new(rel).strip_prefix(&root).ok()
            .or_else(|| (rel == "/").then_some(Path::new("")));
        let Some(relative) = relative else { continue };
        if relative.components().any(|p| matches!(p, std::path::Component::ParentDir)) { continue; }
        let mut current = mount.join(relative);
        for _ in 0..256 {
            let number = |name: &str| std::fs::read_to_string(current.join(name)).ok()?.trim().parse::<u64>().ok();
            // Crossing memory.high causes reclaim/throttling. "max" is ignored.
            // Counting current file cache is conservative: no assumed reclaim.
            let limit = min_known(number("memory.max"), number("memory.high"));
            result.0 = min_known(result.0, limit);
            if let Some(limit) = limit {
                // A known controller ceiling with unknown usage is not free
                // memory. Fail closed for automatic cache allocation.
                let room = number("memory.current").map_or(0, |used| limit.saturating_sub(used));
                result.1 = min_known(result.1, Some(room));
            }
            if current == mount || !current.pop() || !current.starts_with(&mount) { break; }
        }
    }
    result
}

/// Snapshot each load; installed memory is never treated as available memory.
pub fn snapshot() -> MemorySnapshot {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let (limit, headroom) = cgroup_limits();
        return MemorySnapshot {
            physical_total_bytes: meminfo_bytes(&text, "MemTotal:"),
            available_bytes: min_known(meminfo_bytes(&text, "MemAvailable:"), headroom),
            cgroup_limit_bytes: limit, cgroup_headroom_bytes: headroom,
        };
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemStatus { len: u32, load: u32, total: u64, avail: u64, rest: [u64; 5] }
        #[link(name = "kernel32")]
        unsafe extern "system" { fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32; }
        let mut m = MemStatus { len: size_of::<MemStatus>() as u32, load: 0, total: 0, avail: 0, rest: [0; 5] };
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 && m.total > 0 {
            return MemorySnapshot { physical_total_bytes: Some(m.total), available_bytes: Some(m.avail), ..Default::default() };
        }
    }
    #[allow(unreachable_code)]
    MemorySnapshot::default()
}

/// Total physical RAM in bytes, or `None` if undetectable.
pub fn total_ram() -> Option<u64> {
    snapshot().physical_total_bytes
}

/// Usable cap in bytes: `max_ram` if nonzero, else `total - margin`. `None` means unlimited (RAM undetectable).
pub fn ram_cap(margin: u64, max_ram: u64) -> Option<u64> {
    if max_ram > 0 {
        return Some(max_ram);
    }
    let s = snapshot();
    Some(min_known(s.physical_total_bytes, s.cgroup_limit_bytes)?.saturating_sub(margin))
}

/// `auto`, `0` (off), or a ceiling in MiB. Return bytes; None means automatic.
pub fn parse_expert_ram(value: &str) -> anyhow::Result<Option<u64>> {
    if value == "auto" { return Ok(None); }
    let mib = value.parse::<u64>().context("--expert-ram must be auto, 0, or a whole number of MiB")?;
    anyhow::ensure!(mib <= 1_048_576, "--expert-ram exceeds the 1 TiB input limit (units are MiB)");
    Ok(Some(mib * MIB))
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ExpertRamPlan {
    pub memory: MemorySnapshot,
    pub expert_weight_bytes: u64,
    pub os_margin_bytes: u64,
    pub prompt_cache_reserve_bytes: u64,
    pub runtime_reserve_bytes: u64,
    pub existing_owned_bytes: u64,
    pub requested_ceiling_bytes: Option<u64>,
    pub budget_bytes: u64,
    pub all_experts_fit_budget: bool,
}

/// Existing weights are reflected in MemAvailable: subtract them only from an
/// explicit process cap. Future prompt/runtime allocations reduce both bounds.
pub fn expert_plan(memory: MemorySnapshot, expert_bytes: u64, owned_bytes: u64,
    margin: u64, max_ram: u64, prompt_cache: u64, requested: Option<u64>) -> ExpertRamPlan {
    let future = prompt_cache.saturating_add(EXPERT_RUNTIME_RESERVE);
    let available_room = memory.available_bytes.map(|n| n.saturating_sub(margin).saturating_sub(future));
    // If discovery fails, auto stays off. An explicit ceiling may be used but
    // never overrides known available RAM, the OS margin or --max-ram.
    let mut room = available_room.unwrap_or_else(|| requested.unwrap_or(0));
    if max_ram > 0 { room = room.min(max_ram.saturating_sub(owned_bytes).saturating_sub(future)); }
    if let Some(want) = requested { room = room.min(want); }
    let budget = room.min(expert_bytes);
    ExpertRamPlan { memory, expert_weight_bytes: expert_bytes, os_margin_bytes: margin,
        prompt_cache_reserve_bytes: prompt_cache, runtime_reserve_bytes: EXPERT_RUNTIME_RESERVE,
        existing_owned_bytes: owned_bytes, requested_ceiling_bytes: requested,
        budget_bytes: budget, all_experts_fit_budget: expert_bytes > 0 && budget == expert_bytes }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;
    fn host(total: u64, available: u64) -> MemorySnapshot {
        MemorySnapshot { physical_total_bytes: Some(total), available_bytes: Some(available), ..Default::default() }
    }

    #[test]
    fn explicit_max_ram_wins() {
        assert_eq!(ram_cap(512 << 20, 8 << 30), Some(8 << 30));
    }

    #[test]
    fn sixty_two_gib_budget_uses_available_ram_and_excludes_lazy_ple() {
        let p = expert_plan(host(62*GIB, 53*GIB), 60_384_870_400, GIB, 4*GIB, 0, 512*MIB, None);
        assert_eq!(p.budget_bytes, 47*GIB + 512*MIB);
        assert!(!p.all_experts_fit_budget);
        let q = expert_plan(host(62*GIB, 53*GIB), 50_292_326_400, GIB, 4*GIB, 0, 512*MIB, None);
        assert!(q.all_experts_fit_budget);
    }
    #[test]
    fn explicit_ceiling_never_overrides_available_or_process_cap() {
        let p = expert_plan(host(64*GIB, 12*GIB), 56*GIB, 2*GIB, 4*GIB, 8*GIB, GIB, Some(60*GIB));
        assert_eq!(p.budget_bytes, 4*GIB);
        assert_eq!(expert_plan(host(64*GIB, 3*GIB), 56*GIB, 0, 4*GIB, 0, 0, None).budget_bytes, 0);
    }
    #[test]
    fn unknown_memory_is_not_guessed_and_disabled_stays_zero() {
        assert_eq!(expert_plan(MemorySnapshot::default(), 56*GIB, 0, GIB, 0, 0, None).budget_bytes, 0);
        assert_eq!(expert_plan(host(64*GIB, 60*GIB), 56*GIB, 0, GIB, 0, 0, Some(0)).budget_bytes, 0);
    }
    #[test]
    fn memory_and_user_inputs_are_checked() {
        assert_eq!(meminfo_bytes("MemTotal: 100 kB\nMemAvailable: 60 kB\n", "MemAvailable:"), Some(60*1024));
        assert_eq!(meminfo_bytes("MemAvailable: 18446744073709551615 kB", "MemAvailable:"), None);
        assert_eq!(parse_expert_ram("auto").unwrap(), None);
        assert_eq!(parse_expert_ram("0").unwrap(), Some(0));
        assert_eq!(parse_expert_ram("40960").unwrap(), Some(40*GIB));
        for bad in ["-1", "1.5", "64GB", "1048577"] { assert!(parse_expert_ram(bad).is_err()); }
    }
}
