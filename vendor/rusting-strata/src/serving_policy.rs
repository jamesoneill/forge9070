//! Admission capacity is a compatibility policy, not a VRAM fit test or a speed prediction.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

pub const MAX_PARALLEL: usize = 4;
pub const DEFAULT_QUEUE_CAPACITY: usize = 32;
pub const DEFAULT_QUEUE_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_MAX_WAITING: usize = 64;

/// `Auto` reserves the configured compatible capacity; only requests with work
/// to do occupy a generation slot. It does not inspect free GPU memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Parallelism {
    #[default]
    Auto,
    Fixed(usize),
}

impl fmt::Display for Parallelism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Auto => f.write_str("auto"), Self::Fixed(n) => write!(f, "{n}") }
    }
}

impl FromStr for Parallelism {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "1" => Ok(Self::Fixed(1)),
            "2" => Ok(Self::Fixed(2)),
            "3" => Ok(Self::Fixed(3)),
            "4" => Ok(Self::Fixed(4)),
            _ => Err("--parallel must be auto, 1, 2, 3 or 4".into()),
        }
    }
}

impl TryFrom<String> for Parallelism {
    type Error = String;
    fn try_from(value: String) -> std::result::Result<Self, Self::Error> { value.parse() }
}

impl From<Parallelism> for String {
    fn from(value: Parallelism) -> Self { value.to_string() }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelFeatures {
    pub hyper_connections: bool,
    pub per_layer_embeddings: bool,
    pub sparse_attention: bool,
}

impl ModelFeatures {
    pub fn requires_single_slot(self) -> bool {
        self.hyper_connections || self.per_layer_embeddings || self.sparse_attention
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CapacityPlan {
    pub mode: Parallelism,
    pub ceiling: usize,
    pub slots: usize,
    pub reason: &'static str,
}

/// Resolve only the sequence-capacity restrictions. The model loader must still
/// validate its architecture, MTP head, context size and actual allocations.
pub fn resolve_capacity(mode: Parallelism, ceiling: usize, gpu: bool, mtp: usize,
                        features: ModelFeatures) -> Result<CapacityPlan> {
    ensure!((1..=MAX_PARALLEL).contains(&ceiling), "--max-parallel must be between 1 and 4");
    ensure!(gpu || mtp == 0, "--mtp requires GPU execution");
    let (slots, reason) = match mode {
        Parallelism::Fixed(slots) => {
            ensure!((1..=MAX_PARALLEL).contains(&slots), "--parallel must be between 1 and 4 sequences for this Vulkan profile");
            ensure!(gpu || slots == 1,
                "--parallel greater than 1 requires GPU execution; CPU mode accepts concurrent clients but generates their replies sequentially");
            ensure!(slots == 1 || mtp == 0,
                "--parallel greater than 1 and --mtp cannot run together; choose concurrent sequences or single-sequence MTP");
            ensure!(slots == 1 || !features.requires_single_slot(),
                "this checkpoint's hyper-connections, per-layer embeddings or sparse attention require --parallel 1 in this engine");
            (slots, "explicit compatible slot count; GPU memory fit is checked during loading")
        }
        Parallelism::Auto if !gpu => (1, "CPU execution generates one reply at a time; additional requests queue"),
        Parallelism::Auto if features.requires_single_slot() =>
            (1, "this model's hyper-connections, per-layer embeddings or sparse attention support one active sequence; additional requests queue"),
        Parallelism::Auto if mtp > 0 => (1, "MTP uses one active sequence; additional requests queue"),
        Parallelism::Auto => (ceiling, "compatible GPU capacity up to --max-parallel; occupied slots follow demand; GPU memory fit is checked during loading"),
    };
    Ok(CapacityPlan { mode, ceiling, slots, reason })
}

/// Bound waiting jobs independently from the total accepted connection count.
/// A zero-capacity queue accepts a job only when a generation slot is free.
pub fn validate_queue_limits(queue_capacity: usize, timeout_seconds: u64,
                             max_waiting: usize, maximum_slots: usize) -> Result<()> {
    ensure!(queue_capacity <= 4096, "--queue-capacity must be between 0 and 4096");
    ensure!((1..=86400).contains(&timeout_seconds), "--queue-timeout must be between 1 and 86400 seconds");
    ensure!((1..=MAX_PARALLEL).contains(&maximum_slots), "generation capacity must be between 1 and 4");
    let required = queue_capacity + maximum_slots;
    ensure!(max_waiting >= required,
        "--max-waiting must be at least --queue-capacity plus maximum generation capacity ({required}); reduce the queue or raise the connection limit");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_limits_by_backend_architecture_and_speculation() {
        let plain = ModelFeatures::default();
        assert_eq!(resolve_capacity(Parallelism::Auto, 4, true, 0, plain).unwrap().slots, 4);
        assert_eq!(resolve_capacity(Parallelism::Auto, 2, true, 0, plain).unwrap().slots, 2);
        assert_eq!(resolve_capacity(Parallelism::Auto, 4, false, 0, plain).unwrap().slots, 1);
        assert_eq!(resolve_capacity(Parallelism::Auto, 4, true, 2, plain).unwrap().slots, 1);
        for features in [
            ModelFeatures { hyper_connections: true, ..plain },
            ModelFeatures { per_layer_embeddings: true, ..plain },
            ModelFeatures { sparse_attention: true, ..plain },
        ] {
            assert_eq!(resolve_capacity(Parallelism::Auto, 4, true, 0, features).unwrap().slots, 1);
            assert!(resolve_capacity(Parallelism::Fixed(2), 4, true, 0, features).is_err());
        }
    }

    #[test]
    fn explicit_requests_never_silently_downgrade() {
        let plain = ModelFeatures::default();
        assert_eq!(resolve_capacity(Parallelism::Fixed(4), 1, true, 0, plain).unwrap().slots, 4);
        assert!(resolve_capacity(Parallelism::Fixed(4), 4, false, 0, plain).is_err());
        assert!(resolve_capacity(Parallelism::Fixed(2), 4, true, 1, plain).is_err());
        assert!(resolve_capacity(Parallelism::Auto, 4, false, 1, plain).is_err());
        for n in [0, 5, usize::MAX] {
            assert!(resolve_capacity(Parallelism::Fixed(n), 4, true, 0, plain).is_err());
            assert!(resolve_capacity(Parallelism::Auto, n, true, 0, plain).is_err());
        }
    }

    #[test]
    fn only_documented_modes_parse_and_round_trip() {
        for text in ["auto", "1", "2", "3", "4"] {
            let mode: Parallelism = text.parse().unwrap();
            assert_eq!(mode.to_string(), text);
            let value = serde_json::to_string(&mode).unwrap();
            assert_eq!(serde_json::from_str::<Parallelism>(&value).unwrap(), mode);
        }
        for text in ["0", "5", "-1", "1.0", "04", "Auto", "", " auto"] {
            assert!(text.parse::<Parallelism>().is_err());
        }
    }

    #[test]
    fn queue_limits_reserve_connections_without_changing_user_values() {
        assert!(validate_queue_limits(DEFAULT_QUEUE_CAPACITY, DEFAULT_QUEUE_TIMEOUT_SECS, DEFAULT_MAX_WAITING, 4).is_ok());
        assert!(validate_queue_limits(32, 120, 36, 4).is_ok());
        assert!(validate_queue_limits(32, 120, 35, 4).is_err());
        assert!(validate_queue_limits(0, 1, 4, 4).is_ok());
        assert!(validate_queue_limits(0, 1, 3, 4).is_err());
        assert!(validate_queue_limits(4096, 86400, 4100, 4).is_ok());
        for (capacity, timeout) in [(4097, 120), (32, 0), (32, 86401)] {
            assert!(validate_queue_limits(capacity, timeout, 8192, 4).is_err());
        }
    }
}
