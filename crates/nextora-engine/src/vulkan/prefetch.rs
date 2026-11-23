//! Opportunistic bounded MoE host-cache prefetch, driven by observed router IDs.
//! Prediction never changes model routing or the set of evaluated experts.
use crate::gguf::Gguf;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{SyncSender, Receiver, TrySendError, sync_channel},
    },
};

const MIN_OBSERVATIONS: u32 = 16;
const SCHEDULE_EVERY_DECODE_TOKENS: usize = 32;
const ONE_REQUEST_CAPACITY: usize = 1;

#[derive(Debug)]
struct Forecast {
    counts: Vec<Vec<u32>>,
    samples: Vec<u32>,
    cursor: usize,
}

impl Forecast {
    fn new(layers: usize, experts: usize) -> Self {
        Self { counts: vec![vec![0; experts]; layers], samples: vec![0; layers], cursor: 0 }
    }

    fn observe(&mut self, layer: usize, selected: &[(usize, f32)]) {
        let Some(counts) = self.counts.get_mut(layer) else { return };
        if self.samples[layer] >= u32::MAX - 1 {
            for count in counts.iter_mut() { *count /= 2; }
            self.samples[layer] /= 2;
        }
        self.samples[layer] += 1;
        for &(id, _) in selected {
            if let Some(count) = counts.get_mut(id) {
                *count = count.saturating_add(1);
            }
        }
    }

    fn prediction(&mut self) -> Option<(usize, usize)> {
        if self.counts.is_empty() { return None }
        // Round-robin prevents fast layers from starving other layers.
        for _ in 0..self.counts.len() {
            let layer = self.cursor % self.counts.len();
            self.cursor = (layer + 1) % self.counts.len();
            let samples = self.samples[layer];
            if samples < MIN_OBSERVATIONS { continue }
            let candidate = self.counts[layer].iter().enumerate()
                .max_by_key(|(id, count)| (**count, std::cmp::Reverse(*id)))?;
            // Require repeatable routing frequency, not random single selections.
            if candidate.1.saturating_mul(4) >= samples {
                return Some((layer, candidate.0));
            }
        }
        None
    }
}

#[derive(Debug, Default)]
pub(super) struct Counts {
    pub(super) scheduled: AtomicU64,
    pub(super) busy_skips: AtomicU64,
    pub(super) completed: AtomicU64,
    pub(super) bytes_loaded: AtomicU64,
    pub(super) budget_skips: AtomicU64,
    pub(super) errors: AtomicU64,
}

#[derive(Debug)]
pub(super) struct Prefetcher {
    forecast: Mutex<Forecast>,
    sender: SyncSender<(usize, usize)>,
    pub(super) counts: Arc<Counts>,
}

fn worker(source: Arc<Gguf>, receiver: Receiver<(usize, usize)>, counts: Arc<Counts>) {
    while let Ok((layer, expert)) = receiver.recv() {
        match source.prefetch_expert_triplet(layer, expert) {
            Ok(Some((_, bytes))) => {
                counts.completed.fetch_add(1, Ordering::Relaxed);
                counts.bytes_loaded.fetch_add(bytes as u64, Ordering::Relaxed);
            }
            Ok(None) => {
                counts.budget_skips.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                // Opportunistic hints never fail user requests.
                counts.errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Prefetcher {
    pub(super) fn new(source: Arc<Gguf>, layers: usize, experts: usize) -> Self {
        let (sender, receiver) = sync_channel(ONE_REQUEST_CAPACITY);
        let counts = Arc::new(Counts::default());
        let worker_counts = Arc::clone(&counts);
        std::thread::spawn(move || worker(source, receiver, worker_counts));
        Self { forecast: Mutex::new(Forecast::new(layers, experts)), sender, counts }
    }

    pub(super) fn observe(&self, layer: usize, selected: &[(usize, f32)], decode: bool) {
        if !decode { return }
        if let Ok(mut forecast) = self.forecast.lock() {
            forecast.observe(layer, selected);
        }
    }

    pub(super) fn schedule(&self, position: usize, decode: bool) {
        if !decode || position % SCHEDULE_EVERY_DECODE_TOKENS != 0 { return }
        let candidate = self.forecast.lock().ok().and_then(|mut f| f.prediction());
        if let Some(candidate) = candidate {
            match self.sender.try_send(candidate) {
                Ok(()) => { self.counts.scheduled.fetch_add(1, Ordering::Relaxed); }
                Err(TrySendError::Full(_)) => {
                    self.counts.busy_skips.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.counts.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_hotspots_need_repeated_evidence() {
        let mut model = Forecast::new(2, 10);
        for i in 0..15 {
            model.observe(0, &[(if i % 2 == 0 { 2 } else { 3 }, 1.0)]);
        }
        assert!(model.prediction().is_none());
        model.observe(0, &[(2, 1.0)]);
        assert_eq!(model.prediction(), Some((0, 2)));
    }

    #[test]
    fn equal_hotspots_have_stable_lowest_id_tie_break() {
        let mut model = Forecast::new(1, 6);
        for _ in 0..32 { model.observe(0, &[(4, 0.5), (1, 0.5)]); }
        assert_eq!(model.prediction(), Some((0, 1)));
    }

    #[test]
    fn no_prediction_for_sparse_random_routes_or_unseen_layers() {
        let mut model = Forecast::new(2, 128);
        for id in 0..32 { model.observe(0, &[(id, 1.0)]); }
        assert_eq!(model.prediction(), None);
        assert_eq!(Forecast::new(0, 0).prediction(), None);
    }
}
