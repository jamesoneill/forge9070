//! Bounded, byte-preserving host cache for routed experts.
//!
//! Each entry owns the gate, up and down matrices of one expert. Leases keep an
//! entry alive through CPU kernels and host-to-device copies. An entry with an
//! outstanding lease cannot be evicted: if every candidate is busy, a lookup
//! borrows the original weights instead of exceeding the configured byte cap.
//! The storage is pageable host memory, not HIP-registered or Vulkan-imported.

use crate::quant::GgmlType;
use crate::weights::Tensor;
use anyhow::{Context, ensure};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

/// Counts for the owned expert payload. Allocator metadata and the bounded I/O
/// and GPU transfer scratch buffers are separate runtime allocations.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ExpertRamStats {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub peak_bytes: usize,
    pub resident_experts: usize,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub bypasses: u64,
    pub prefill_hits: u64,
    pub prefill_bypasses: u64,
    pub warm_admissions: u64,
    pub warm_bypasses: u64,
    pub allocation_failures: u64,
    pub direct_reads: u64,
    pub buffered_reads: u64,
    pub owned_source_copies: u64,
}

/// How an individual matrix was loaded. A model's existing owned override is
/// copied directly; mapped tensors are read using their retained backing file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertReadKind { Direct, Buffered, OwnedCopy }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Admission { Decode, Prefill, Warm }

#[derive(Clone, Copy)]
struct Matrix { ty: GgmlType, cols: usize, rows: usize, off: usize, bytes: usize }

struct OwnedExpert {
    mats: [Matrix; 3],
    data: Box<[u8]>,
}

impl OwnedExpert {
    fn tensors(&self) -> [Tensor<'_>; 3] {
        self.mats.map(|m| Tensor { ty: m.ty, cols: m.cols, rows: m.rows, data: &self.data[m.off..m.off + m.bytes] })
    }
}

/// Borrowed views never outlive the source weights; owned views borrow this
/// lease, rather than the cache mutex or an evictable raw pointer.
pub struct ExpertLease<'a> { storage: LeaseStorage<'a> }

// Keep Arc private: a caller must not retain a Weak pointer and upgrade it
// after the cache checks whether an entry is unleased and accounts it away.
enum LeaseStorage<'a> {
    Borrowed([Tensor<'a>; 3]),
    Owned(Arc<OwnedExpert>),
}

impl<'a> ExpertLease<'a> {
    fn borrowed(t: [Tensor<'a>; 3]) -> Self { Self { storage: LeaseStorage::Borrowed(t) } }
    fn owned(e: Arc<OwnedExpert>) -> Self { Self { storage: LeaseStorage::Owned(e) } }
    pub fn tensors(&self) -> [Tensor<'_>; 3] {
        match &self.storage { LeaseStorage::Borrowed(t) => *t, LeaseStorage::Owned(e) => e.tensors() }
    }
    pub fn is_owned(&self) -> bool { matches!(&self.storage, LeaseStorage::Owned(_)) }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Key { generation: u64, layer: usize, expert: usize, count: usize }

struct Entry { data: Arc<OwnedExpert>, last_use: u64 }

#[derive(Default)]
struct State {
    entries: HashMap<Key, Entry>,
    /// One record per entry: bounded metadata and logarithmic LRU updates,
    /// without scanning tens of thousands of experts on every eviction.
    order: BTreeSet<(u64, Key)>,
    generation: u64,
    sequence: u64,
    stats: ExpertRamStats,
}

impl State {
    fn next_stamp(&mut self) -> u64 {
        if self.sequence == u64::MAX {
            // Preserve ordering before reusing a stamp. This is exceptional,
            // but avoids silent LRU degradation after counter overflow.
            let keys: Vec<Key> = self.order.iter().map(|(_, key)| *key).collect();
            self.order.clear();
            for (i, key) in keys.into_iter().enumerate() {
                self.entries.get_mut(&key).unwrap().last_use = i as u64;
                self.order.insert((i as u64, key));
            }
            self.sequence = self.entries.len() as u64;
        }
        self.sequence += 1;
        self.sequence
    }

    fn hit<'a>(&mut self, key: Key, policy: Admission) -> Option<ExpertLease<'a>> {
        if !self.entries.contains_key(&key) { return None; }
        let now = if policy == Admission::Decode { self.next_stamp() } else { 0 };
        let entry = self.entries.get_mut(&key).unwrap();
        let previous = entry.last_use;
        let data = Arc::clone(&entry.data);
        match policy {
            Admission::Decode => {
                entry.last_use = now;
                self.order.remove(&(previous, key));
                self.order.insert((now, key));
                self.stats.hits += 1;
            }
            Admission::Prefill => self.stats.prefill_hits += 1,
            Admission::Warm => {}
        }
        Some(ExpertLease::owned(data))
    }

    fn bypass(&mut self, policy: Admission) {
        if policy == Admission::Warm { self.stats.warm_bypasses += 1; }
        else { self.stats.bypasses += 1; }
    }
}

#[derive(Default)]
pub(crate) struct ExpertRamCache { state: Mutex<State> }

impl ExpertRamCache {
    pub fn configure(&self, budget_bytes: usize) -> anyhow::Result<()> {
        let mut s = self.state.lock().map_err(|_| anyhow::anyhow!("expert RAM cache lock poisoned"))?;
        if s.stats.budget_bytes == budget_bytes { return Ok(()); }
        ensure!(s.entries.values().all(|e| Arc::strong_count(&e.data) == 1),
            "cannot resize the expert RAM cache while an expert lease is active");
        s.entries.clear();
        s.order.clear();
        s.sequence = 0;
        s.stats = ExpertRamStats { budget_bytes, ..Default::default() };
        Ok(())
    }

    pub fn enabled(&self) -> bool { self.stats().budget_bytes > 0 }

    pub fn stats(&self) -> ExpertRamStats {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).stats
    }

    /// An override change must never return a cached copy of earlier weights.
    /// Retain any active old leases in the accounting until normal eviction can
    /// remove them; a new generation prevents subsequent lookups hitting them.
    pub fn invalidate(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.generation = s.generation.checked_add(1).expect("expert cache generation exhausted");
        s.entries.retain(|_, e| Arc::strong_count(&e.data) > 1);
        s.order = s.entries.iter().map(|(&key, e)| (e.last_use, key)).collect();
        s.stats.resident_experts = s.entries.len();
        s.stats.resident_bytes = s.entries.values().map(|e| e.data.data.len()).sum();
    }

    /// A warm decode hit should not format and resolve all three GGUF tensor
    /// names. No source view or public reference to the entry's Arc escapes.
    pub fn owned_hit<'a>(&self, layer: usize, expert: usize, count: usize, admit: bool)
        -> anyhow::Result<Option<ExpertLease<'a>>> {
        let mut s = self.state.lock().map_err(|_| anyhow::anyhow!("expert RAM cache lock poisoned"))?;
        let key = Key { generation: s.generation, layer, expert, count };
        Ok(s.hit(key, if admit { Admission::Decode } else { Admission::Prefill }))
    }

    /// `admit = false` is the transient prefill policy: existing entries can be
    /// used, but neither misses nor hits disturb the decode cache's LRU order.
    /// The miss loader runs while the mutex is held, coalescing concurrent
    /// requests for the same expert and bounding all live entry allocations.
    pub fn acquire<'a>(
        &self, layer: usize, expert: usize, count: usize, source: [Tensor<'a>; 3], admit: bool,
        load: impl FnMut(usize, &mut [u8]) -> anyhow::Result<ExpertReadKind>,
    ) -> anyhow::Result<(ExpertLease<'a>, bool)> {
        self.acquire_policy(layer, expert, count, source,
            if admit { Admission::Decode } else { Admission::Prefill }, load)
    }

    /// Fill genuinely free capacity only. A saved-profile warmup must not
    /// replace useful entries or change their order before the first request.
    pub fn warm<'a>(
        &self, layer: usize, expert: usize, count: usize, source: [Tensor<'a>; 3],
        load: impl FnMut(usize, &mut [u8]) -> anyhow::Result<ExpertReadKind>,
    ) -> anyhow::Result<(ExpertLease<'a>, bool)> {
        self.acquire_policy(layer, expert, count, source, Admission::Warm, load)
    }

    fn acquire_policy<'a>(
        &self, layer: usize, expert: usize, count: usize, source: [Tensor<'a>; 3], policy: Admission,
        mut load: impl FnMut(usize, &mut [u8]) -> anyhow::Result<ExpertReadKind>,
    ) -> anyhow::Result<(ExpertLease<'a>, bool)> {
        let mut s = self.state.lock().map_err(|_| anyhow::anyhow!("expert RAM cache lock poisoned"))?;
        if s.stats.budget_bytes == 0 { return Ok((ExpertLease::borrowed(source), false)); }
        let key = Key { generation: s.generation, layer, expert, count };
        if let Some(hit) = s.hit(key, policy) { return Ok((hit, false)); }
        if policy == Admission::Prefill {
            s.stats.prefill_bypasses += 1;
            return Ok((ExpertLease::borrowed(source), false));
        }
        if policy == Admission::Decode { s.stats.misses += 1; }
        let bytes = source.iter().try_fold(0usize, |n, t| n.checked_add(t.data.len()))
            .context("expert RAM entry size overflow")?;
        if bytes == 0 || bytes > s.stats.budget_bytes {
            s.bypass(policy);
            return Ok((ExpertLease::borrowed(source), false));
        }
        while s.stats.resident_bytes > s.stats.budget_bytes - bytes {
            if policy == Admission::Warm {
                s.bypass(policy);
                return Ok((ExpertLease::borrowed(source), false));
            }
            let victim = s.order.iter().find(|(_, k)| Arc::strong_count(&s.entries[k].data) == 1).copied();
            let Some((stamp, victim)) = victim else {
                s.bypass(policy);
                return Ok((ExpertLease::borrowed(source), false));
            };
            let removed = s.entries.remove(&victim).unwrap();
            s.order.remove(&(stamp, victim));
            s.stats.resident_bytes -= removed.data.data.len();
            s.stats.resident_experts -= 1;
            s.stats.evictions += 1;
            // Free before allocating the replacement; leased entries are never
            // removed, so resident_bytes includes every live owned payload.
            drop(removed);
        }
        let mut data = Vec::new();
        if data.try_reserve_exact(bytes).is_err() {
            s.stats.allocation_failures += 1;
            s.bypass(policy);
            return Ok((ExpertLease::borrowed(source), false));
        }
        data.resize(bytes, 0);
        // This includes the incoming payload even if a later file read fails.
        s.stats.peak_bytes = s.stats.peak_bytes.max(s.stats.resident_bytes + bytes);
        let mut off = 0;
        let mats = source.map(|t| {
            let m = Matrix { ty: t.ty, cols: t.cols, rows: t.rows, off, bytes: t.data.len() };
            off += m.bytes;
            m
        });
        for (i, m) in mats.iter().enumerate() {
            match load(i, &mut data[m.off..m.off + m.bytes])? {
                ExpertReadKind::Direct => s.stats.direct_reads += 1,
                ExpertReadKind::Buffered => s.stats.buffered_reads += 1,
                ExpertReadKind::OwnedCopy => s.stats.owned_source_copies += 1,
            }
        }
        let data = Arc::new(OwnedExpert { mats, data: data.into_boxed_slice() });
        // New warm entries are untouched, older than entries actually used by
        // decode. Only a later decode access promotes them to recent status.
        let now = if policy == Admission::Warm { 0 } else { s.next_stamp() };
        s.entries.insert(key, Entry { data: Arc::clone(&data), last_use: now });
        s.order.insert((now, key));
        s.stats.resident_bytes += bytes;
        s.stats.resident_experts += 1;
        s.stats.admissions += 1;
        if policy == Admission::Warm { s.stats.warm_admissions += 1; }
        s.stats.peak_bytes = s.stats.peak_bytes.max(s.stats.resident_bytes);
        debug_assert!(s.stats.resident_bytes <= s.stats.budget_bytes);
        Ok((ExpertLease::owned(data), true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn source(bytes: &[u8]) -> [Tensor<'_>; 3] {
        assert_eq!(bytes.len() % 12, 0);
        let n = bytes.len() / 3;
        std::array::from_fn(|i| Tensor { ty: GgmlType::F32, cols: n / 4, rows: 1, data: &bytes[i * n..(i + 1) * n] })
    }

    fn get<'a>(cache: &ExpertRamCache, key: usize, bytes: &'a [u8], admit: bool) -> ExpertLease<'a> {
        let src = source(bytes);
        cache.acquire(0, key, 8, src, admit, |i, out| {
            out.copy_from_slice(src[i].data);
            Ok(ExpertReadKind::OwnedCopy)
        }).unwrap().0
    }

    #[test]
    fn preserves_bytes_metadata_and_owned_lifetime() {
        let cache = ExpertRamCache::default();
        cache.configure(48).unwrap();
        let bytes: Vec<u8> = (0..48).collect();
        let lease = get(&cache, 0, &bytes, true);
        assert!(lease.is_owned());
        for (actual, expected) in lease.tensors().iter().zip(source(&bytes)) {
            assert_eq!(actual.data, expected.data);
            assert_eq!((actual.ty, actual.cols, actual.rows), (expected.ty, expected.cols, expected.rows));
            assert_ne!(actual.data.as_ptr(), expected.data.as_ptr());
        }
        assert_eq!(cache.stats().resident_bytes, 48);
        assert!(cache.configure(0).is_err());
        drop(lease);
        cache.configure(0).unwrap();
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[test]
    fn pinned_entries_bypass_instead_of_overallocating_then_evict() {
        let cache = ExpertRamCache::default();
        cache.configure(48).unwrap();
        let a = vec![1; 48];
        let b = vec![2; 48];
        let lease = get(&cache, 0, &a, true);
        assert!(!get(&cache, 1, &b, true).is_owned());
        assert_eq!(lease.tensors()[0].data, &[1; 16]);
        assert_eq!(cache.stats().peak_bytes, 48);
        assert_eq!(cache.stats().bypasses, 1);
        drop(lease);
        assert!(get(&cache, 1, &b, true).is_owned());
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().resident_bytes, 48);
        assert_eq!(cache.stats().peak_bytes, 48);
    }

    #[test]
    fn transient_prefill_preserves_decode_lru_and_never_admits() {
        let cache = ExpertRamCache::default();
        cache.configure(96).unwrap();
        let bytes = vec![3; 48];
        drop(get(&cache, 0, &bytes, true));
        drop(get(&cache, 1, &bytes, true));
        assert!(get(&cache, 0, &bytes, false).is_owned());
        assert!(!get(&cache, 2, &bytes, false).is_owned());
        drop(get(&cache, 2, &bytes, true));
        // A transient hit of expert 0 did not make it newer than expert 1.
        assert!(!get(&cache, 0, &bytes, false).is_owned());
        assert!(get(&cache, 1, &bytes, false).is_owned());
        assert_eq!(cache.stats().resident_experts, 2);
        assert_eq!(cache.stats().admissions, 3);
    }

    #[test]
    fn concurrent_misses_load_once_and_keep_live_bytes_bounded() {
        let cache = ExpertRamCache::default();
        cache.configure(48).unwrap();
        let bytes = vec![7; 48];
        let reads = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let joins: Vec<_> = (0..8).map(|_| scope.spawn(|| {
                let src = source(&bytes);
                let (lease, _) = cache.acquire(0, 0, 8, src, true, |i, out| {
                    reads.fetch_add(1, Ordering::SeqCst);
                    out.copy_from_slice(src[i].data);
                    Ok(ExpertReadKind::Buffered)
                }).unwrap();
                assert_eq!(lease.tensors()[0].data, &[7; 16]);
            })).collect();
            for j in joins { j.join().unwrap(); }
        });
        assert_eq!(reads.load(Ordering::SeqCst), 3);
        assert_eq!(cache.stats().admissions, 1);
        assert_eq!(cache.stats().hits, 7);
        assert_eq!(cache.stats().peak_bytes, 48);
    }

    #[test]
    fn model_instances_and_override_generations_are_isolated() {
        let first = ExpertRamCache::default();
        let second = ExpertRamCache::default();
        first.configure(96).unwrap();
        second.configure(48).unwrap();
        let a = vec![1; 48];
        let b = vec![2; 48];
        let old = get(&first, 0, &a, true);
        assert_eq!(get(&second, 0, &b, true).tensors()[0].data, &[2; 16]);
        first.invalidate();
        assert_eq!(old.tensors()[0].data, &[1; 16]);
        assert_eq!(get(&first, 0, &b, true).tensors()[0].data, &[2; 16]);
        assert_eq!(first.stats().resident_bytes, 96);
        assert_eq!(second.stats().resident_bytes, 48);
    }

    #[test]
    fn disabled_and_oversized_entries_never_invoke_loader() {
        let cache = ExpertRamCache::default();
        let bytes = vec![5; 48];
        let src = source(&bytes);
        let (lease, _) = cache.acquire(0, 0, 8, src, true, |_, _| panic!("disabled loader")).unwrap();
        assert!(!lease.is_owned());
        cache.configure(24).unwrap();
        let (lease, _) = cache.acquire(0, 0, 8, src, true, |_, _| panic!("oversized loader")).unwrap();
        assert!(!lease.is_owned());
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[test]
    fn failed_fill_does_not_publish_partial_weights() {
        let cache = ExpertRamCache::default();
        cache.configure(48).unwrap();
        let bytes = vec![5; 48];
        let src = source(&bytes);
        let result = cache.acquire(0, 0, 8, src, true, |i, out| {
            if i == 1 { anyhow::bail!("simulated file read failure"); }
            out.copy_from_slice(src[i].data);
            Ok(ExpertReadKind::Direct)
        });
        assert!(result.is_err());
        assert_eq!(cache.stats().resident_bytes, 0);
        assert_eq!(cache.stats().resident_experts, 0);
        assert!(get(&cache, 0, &bytes, true).is_owned());
        assert_eq!(cache.stats().admissions, 1);
    }

    #[test]
    fn stamp_overflow_preserves_lru_and_bounded_index() {
        let cache = ExpertRamCache::default();
        cache.configure(96).unwrap();
        let bytes = vec![3; 48];
        drop(get(&cache, 0, &bytes, true));
        drop(get(&cache, 1, &bytes, true));
        cache.state.lock().unwrap().sequence = u64::MAX;
        drop(get(&cache, 0, &bytes, true));
        drop(get(&cache, 2, &bytes, true));
        assert!(get(&cache, 0, &bytes, false).is_owned());
        assert!(!get(&cache, 1, &bytes, false).is_owned());
        let s = cache.state.lock().unwrap();
        assert_eq!(s.order.len(), s.entries.len());
        assert_eq!(s.order.len(), 2);
        assert_eq!(s.stats.peak_bytes, 96);
    }

    #[test]
    fn owned_hit_skips_source_loader_and_promotes_decode_only() {
        let cache = ExpertRamCache::default();
        cache.configure(96).unwrap();
        let bytes = vec![3; 48];
        drop(get(&cache, 0, &bytes, true));
        drop(get(&cache, 1, &bytes, true));
        let hit = cache.owned_hit(0, 0, 8, true).unwrap().unwrap();
        assert_eq!(hit.tensors()[0].data, &[3; 16]);
        drop(hit);
        assert!(cache.owned_hit(0, 99, 8, true).unwrap().is_none());
        drop(get(&cache, 2, &bytes, true));
        assert!(cache.owned_hit(0, 0, 8, false).unwrap().is_some());
        assert!(cache.owned_hit(0, 1, 8, false).unwrap().is_none());
        assert_eq!(cache.stats().admissions, 3);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn warmup_fills_free_capacity_without_touching_or_evicting() {
        let cache = ExpertRamCache::default();
        cache.configure(96).unwrap();
        let bytes = vec![6; 48];
        let src = source(&bytes);
        drop(get(&cache, 0, &bytes, true));
        let warm = |key| cache.warm(0, key, 8, src, |i, out| {
            out.copy_from_slice(src[i].data);
            Ok(ExpertReadKind::Buffered)
        }).unwrap();
        assert!(!warm(0).1); // An existing entry is not a new admission.
        let (lease, added) = warm(1);
        assert!(added && lease.is_owned());
        drop(lease);
        assert!(!warm(2).1);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().warm_admissions, 1);
        assert_eq!(cache.stats().warm_bypasses, 1);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().peak_bytes, 96);
        // Untouched warm data is older than the actually used decode entry.
        drop(get(&cache, 2, &bytes, true));
        assert!(cache.owned_hit(0, 0, 8, false).unwrap().is_some());
        assert!(cache.owned_hit(0, 1, 8, false).unwrap().is_none());
    }
}
