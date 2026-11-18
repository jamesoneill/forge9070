//! Bounded owned expert bytes with immutable leases. Only unleased entries are
//! evicted. A pinned working set consumes the same capacity as cached entries;
//! misses that cannot fit use the mapping without allocating an overflow copy.
use super::TensorKey;
use super::expert_io::{LoadedBytes, ReadMethod};
use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub(super) struct OwnedBytes(pub(super) Vec<u8>);

#[derive(Debug)]
struct Entry {
    bytes: Arc<OwnedBytes>,
    touched: u128,
}

#[derive(Debug, Default)]
struct State {
    entries: BTreeMap<TensorKey, Entry>,
    recency: BTreeSet<(u128, TensorKey)>,
    resident: usize,
    clock: u128,
    hits: u64,
    misses: u64,
    evictions: u64,
    mapped_bypasses: u64,
    prefill_hits: u64,
    prefill_misses: u64,
    direct_expert_loads: u64,
    buffered_expert_loads: u64,
    direct_fallbacks: u64,
    payload_bytes_read: u64,
    read_errors: u64,
}

#[derive(Debug)]
pub(super) struct ExpertCache {
    capacity: usize,
    state: Mutex<State>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExpertCacheStats {
    pub capacity_bytes: usize,
    /// Owned encoded weight payload, excluding map/page cache and metadata.
    pub resident_bytes: usize,
    /// Resident entries currently borrowed by at least one immutable lease.
    pub pinned_bytes: usize,
    pub resident_entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub mapped_bypasses: u64,
    pub prefill_hits: u64,
    pub prefill_misses: u64,
    pub direct_expert_loads: u64,
    pub buffered_expert_loads: u64,
    pub direct_fallbacks: u64,
    pub payload_bytes_read: u64,
    pub read_errors: u64,
    pub io_scratch_max_bytes: usize,
}

impl ExpertCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(State::default()),
        }
    }

    #[cfg(test)]
    fn lease(&self, key: TensorKey, payload: &[u8]) -> Result<Option<Arc<OwnedBytes>>> {
        self.lease_with(key, payload.len(), || {
            Ok(LoadedBytes {
                bytes: payload.to_vec(),
                method: ReadMethod::Memory,
                direct_fallback: false,
            })
        })
    }

    pub(super) fn lease_with(
        &self,
        key: TensorKey,
        length: usize,
        load: impl FnOnce() -> Result<LoadedBytes>,
    ) -> Result<Option<Arc<OwnedBytes>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
        state.clock = state.clock.saturating_add(1);
        let touched = state.clock;
        if let Some(entry) = state.entries.get_mut(&key) {
            let previous = entry.touched;
            entry.touched = touched;
            let data = Arc::clone(&entry.bytes);
            state.recency.remove(&(previous, key));
            state.recency.insert((touched, key));
            state.hits = state.hits.saturating_add(1);
            return Ok(Some(data));
        }
        state.misses = state.misses.saturating_add(1);
        if length > self.capacity || self.capacity == 0 {
            state.mapped_bypasses = state.mapped_bypasses.saturating_add(1);
            return Ok(None);
        }
        let needed = state.resident.saturating_sub(self.capacity - length);
        if needed != 0 {
            // Inspect only as many oldest entries as needed; avoid scanning the
            // entire (potentially tens-of-thousands-of-experts) cache per miss.
            let mut reclaimable = 0usize;
            let mut victims = Vec::new();
            for &(_, candidate) in &state.recency {
                let entry = &state.entries[&candidate];
                if Arc::strong_count(&entry.bytes) != 1 {
                    continue;
                }
                reclaimable += entry.bytes.0.len();
                victims.push(candidate);
                if reclaimable >= needed {
                    break;
                }
            }
            if reclaimable < needed {
                state.mapped_bypasses = state.mapped_bypasses.saturating_add(1);
                return Ok(None);
            }
            for victim in victims {
                let entry = state
                    .entries
                    .remove(&victim)
                    .expect("victim belongs to locked cache");
                state.recency.remove(&(entry.touched, victim));
                state.resident -= entry.bytes.0.len();
                state.evictions = state.evictions.saturating_add(1);
            }
        }
        // Hold the lock across the bounded offset read: simultaneous misses
        // cannot reserve separate overflow allocations beyond the byte ceiling.
        let loaded = match load() {
            Ok(data) => data,
            Err(error) => {
                state.read_errors = state.read_errors.saturating_add(1);
                return Err(error);
            }
        };
        anyhow::ensure!(
            loaded.bytes.len() == length,
            "expert loader returned wrong payload length"
        );
        Self::record_read(&mut state, &loaded);
        let bytes = Arc::new(OwnedBytes(loaded.bytes));
        state.resident += length;
        state.entries.insert(
            key,
            Entry {
                bytes: Arc::clone(&bytes),
                touched,
            },
        );
        state.recency.insert((touched, key));
        Ok(Some(bytes))
    }

    fn record_read(state: &mut State, loaded: &LoadedBytes) {
        match loaded.method {
            ReadMethod::Direct => {
                state.direct_expert_loads = state.direct_expert_loads.saturating_add(1)
            }
            ReadMethod::Buffered => {
                state.buffered_expert_loads = state.buffered_expert_loads.saturating_add(1)
            }
            #[cfg(test)]
            ReadMethod::Memory => {}
        }
        if loaded.direct_fallback {
            state.direct_fallbacks = state.direct_fallbacks.saturating_add(1);
        }
        state.payload_bytes_read = state
            .payload_bytes_read
            .saturating_add(loaded.bytes.len() as u64);
    }

    /// Admit an expert's gate/up/down as one unit only when all missing bytes
    /// fit without eviction. Nothing is published if any read fails. Returned
    /// counts describe newly loaded matrices and payload bytes, not cache hits.
    pub(super) fn warm_batch(
        &self,
        entries: &[(TensorKey, usize)],
        mut load: impl FnMut(TensorKey, usize) -> Result<LoadedBytes>,
    ) -> Result<Option<(usize, usize)>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
        let unique: std::collections::BTreeSet<_> = entries.iter().map(|(key, _)| *key).collect();
        anyhow::ensure!(
            unique.len() == entries.len(),
            "duplicate byte ranges in expert warmup batch"
        );
        let mut needed = 0usize;
        for &(key, bytes) in entries {
            if !state.entries.contains_key(&key) {
                needed = needed
                    .checked_add(bytes)
                    .ok_or_else(|| anyhow::anyhow!("warmup payload overflow"))?;
            }
        }
        if needed > self.capacity - state.resident {
            return Ok(None);
        }
        let mut loaded = Vec::new();
        for &(key, length) in entries {
            if state.entries.contains_key(&key) {
                continue;
            }
            let data = match load(key, length) {
                Ok(data) => data,
                Err(error) => {
                    state.read_errors = state.read_errors.saturating_add(1);
                    return Err(error);
                }
            };
            anyhow::ensure!(
                data.bytes.len() == length,
                "warmup loader returned wrong payload length"
            );
            Self::record_read(&mut state, &data);
            loaded.push((key, Arc::new(OwnedBytes(data.bytes))));
        }
        let matrices = loaded.len();
        for (key, bytes) in loaded {
            state.clock = state.clock.saturating_add(1);
            let touched = state.clock;
            state.entries.insert(key, Entry { bytes, touched });
            state.recency.insert((touched, key));
        }
        state.resident += needed;
        Ok(Some((matrices, needed)))
    }

    /// Prompt processing can reuse hot experts, but never displace them or
    /// refresh decode recency merely because a long prompt visits an expert.
    pub(super) fn prefill_lease(&self, key: TensorKey) -> Result<Option<Arc<OwnedBytes>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
        let bytes = state
            .entries
            .get(&key)
            .map(|entry| Arc::clone(&entry.bytes));
        if bytes.is_some() {
            state.hits = state.hits.saturating_add(1);
            state.prefill_hits = state.prefill_hits.saturating_add(1);
        } else {
            state.misses = state.misses.saturating_add(1);
            state.prefill_misses = state.prefill_misses.saturating_add(1);
            state.mapped_bypasses = state.mapped_bypasses.saturating_add(1);
        }
        Ok(bytes)
    }

    pub(super) fn stats(&self) -> Result<ExpertCacheStats> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
        Ok(ExpertCacheStats {
            capacity_bytes: self.capacity,
            resident_bytes: state.resident,
            pinned_bytes: state
                .entries
                .values()
                .filter(|e| Arc::strong_count(&e.bytes) > 1)
                .map(|e| e.bytes.0.len())
                .sum(),
            resident_entries: state.entries.len(),
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            mapped_bypasses: state.mapped_bypasses,
            prefill_hits: state.prefill_hits,
            prefill_misses: state.prefill_misses,
            direct_expert_loads: state.direct_expert_loads,
            buffered_expert_loads: state.buffered_expert_loads,
            direct_fallbacks: state.direct_fallbacks,
            payload_bytes_read: state.payload_bytes_read,
            read_errors: state.read_errors,
            io_scratch_max_bytes: 64 * 1024,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(offset: u64, length: u64) -> TensorKey {
        TensorKey {
            model: 1,
            shard: 0,
            offset,
            length,
            quantization: 0,
        }
    }

    #[test]
    fn pinned_leases_block_eviction_without_overflow_allocations() {
        let cache = ExpertCache::new(8);
        let a = cache.lease(key(0, 4), &[1; 4]).unwrap().unwrap();
        let b = cache.lease(key(4, 4), &[2; 4]).unwrap().unwrap();
        assert!(cache.lease(key(8, 4), &[3; 4]).unwrap().is_none());
        let stats = cache.stats().unwrap();
        assert_eq!(
            (
                stats.resident_bytes,
                stats.pinned_bytes,
                stats.evictions,
                stats.mapped_bypasses
            ),
            (8, 8, 0, 1)
        );
        assert_eq!(a.0, [1; 4]);
        drop(a);
        let c = cache.lease(key(8, 4), &[3; 4]).unwrap().unwrap();
        assert_eq!(b.0, [2; 4]);
        assert_eq!(c.0, [3; 4]);
        assert_eq!(cache.stats().unwrap().resident_bytes, 8);
        assert_eq!(cache.stats().unwrap().evictions, 1);
    }

    #[test]
    fn hits_share_bytes_and_lru_replaces_only_unleased_entries() {
        let cache = ExpertCache::new(8);
        let a = cache.lease(key(0, 4), &[1; 4]).unwrap().unwrap();
        let hit = cache.lease(key(0, 4), &[1; 4]).unwrap().unwrap();
        assert!(Arc::ptr_eq(&a, &hit));
        drop(a);
        drop(hit);
        drop(cache.lease(key(4, 4), &[2; 4]).unwrap());
        drop(cache.lease(key(0, 4), &[1; 4]).unwrap());
        drop(cache.lease(key(8, 4), &[3; 4]).unwrap());
        assert_eq!(cache.stats().unwrap().hits, 2);
        assert!(cache.state.lock().unwrap().entries.contains_key(&key(0, 4)));
        assert!(!cache.state.lock().unwrap().entries.contains_key(&key(4, 4)));
        assert!(cache.lease(key(16, 9), &[9; 9]).unwrap().is_none());
        assert!(cache.stats().unwrap().resident_bytes <= 8);
    }

    #[test]
    fn concurrent_misses_never_exceed_capacity_and_retained_leases_stay_valid() {
        let cache = Arc::new(ExpertCache::new(64));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let workers: Vec<_> = (0..8)
            .map(|i| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let lease = cache.lease(key(i * 32, 32), &[i as u8; 32]).unwrap();
                    barrier.wait();
                    assert!(cache.stats().unwrap().resident_bytes <= 64);
                    if let Some(lease) = lease {
                        assert_eq!(lease.0, [i as u8; 32]);
                    }
                })
            })
            .collect();
        for thread in workers {
            thread.join().unwrap();
        }
        assert!(cache.stats().unwrap().resident_bytes <= 64);
    }
    #[test]
    fn warmup_budget_and_read_failure_do_not_publish_partial_experts() {
        let cache = ExpertCache::new(12);
        let entries = [(key(0, 4), 4), (key(4, 4), 4), (key(8, 4), 4)];
        let mut calls = 0;
        let result = cache.warm_batch(&entries, |_, len| {
            calls += 1;
            if calls == 2 {
                anyhow::bail!("injected read error");
            }
            Ok(LoadedBytes {
                bytes: vec![9; len],
                method: ReadMethod::Memory,
                direct_fallback: false,
            })
        });
        assert!(result.is_err());
        assert_eq!(
            (
                cache.stats().unwrap().resident_bytes,
                cache.stats().unwrap().resident_entries
            ),
            (0, 0)
        );
        assert_eq!(cache.stats().unwrap().read_errors, 1);
        let oversized = [(key(0, 8), 8), (key(8, 8), 8)];
        assert!(
            cache
                .warm_batch(&oversized, |_, _| panic!("budget rejected before read"))
                .unwrap()
                .is_none()
        );
        let good = cache
            .warm_batch(&entries, |_, len| {
                Ok(LoadedBytes {
                    bytes: vec![3; len],
                    method: ReadMethod::Memory,
                    direct_fallback: false,
                })
            })
            .unwrap();
        assert_eq!(good, Some((3, 12)));
        assert_eq!(
            (
                cache.stats().unwrap().resident_bytes,
                cache.stats().unwrap().evictions
            ),
            (12, 0)
        );
    }
}
