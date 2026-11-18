//! Optional, output-changing omission of low-ranked CPU expert misses.
//!
//! Inspired by Avalon's `moe_hosted_tail_skip` at commit
//! 762a3dae8d78ff18acb38e351dfa940328485fd8. This is a fresh Rust implementation
//! of the policy, not a claim of Avalon's HIP kernels or measured throughput.
//! Ranks here are one-based: `min_rank = 8` corresponds to Avalon's index 7.
//! The exact default is zero. Cached experts are never omitted, and a miss is
//! retained if ANY token in the pass ranks that expert above the threshold.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pass {
    #[default]
    Decode,
    Prefill,
    Draft,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Policy { pub min_rank: usize }

impl Policy {
    /// Read once while constructing the model's CPU expert worker.
    pub fn from_env() -> Result<Self> {
        let Some(value) = std::env::var_os("STRATA_EXPERT_TAIL_MIN_RANK") else { return Ok(Self::default()); };
        Self::parse(value.to_str().context("STRATA_EXPERT_TAIL_MIN_RANK must be an integer")?)
    }

    pub fn parse(value: &str) -> Result<Self> {
        let min_rank = value.parse::<usize>().context("STRATA_EXPERT_TAIL_MIN_RANK must be an integer from 0 to 256")?;
        ensure!(min_rank <= 256, "STRATA_EXPERT_TAIL_MIN_RANK must be from 0 to 256 (0 is exact mode)");
        Ok(Self { min_rank })
    }

    pub fn applies(self, pass: Pass) -> bool { self.min_rank > 0 && pass == Pass::Decode }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Stats {
    /// Configured one-based rank; 0 means exact mode.
    pub min_rank: usize,
    /// Omitted (token, expert) assignments across main-model decode/verify passes.
    pub skipped_assignments: u64,
    /// Distinct omitted experts counted once per layer and pass, not globally.
    pub skipped_expert_groups: u64,
}

pub(crate) struct Plan {
    /// True exactly for assignments the CPU still needs to compute.
    pub compute: Vec<bool>,
    pub skipped_assignments: u64,
    pub skipped_expert_groups: u64,
}

/// `ids` and signed-slot bit patterns are token-major arrays of `top_k` entries.
/// Returning None on inconsistent geometry makes the caller use the exact path.
/// Weights are intentionally absent: retained assignments keep original weights.
pub(crate) fn plan(ids: &[u32], slots: &[u32], top_k: usize, min_rank: usize) -> Option<Plan> {
    if top_k == 0 || ids.len() != slots.len() || !ids.len().is_multiple_of(top_k) { return None; }
    let mut result = Plan {
        compute: slots.iter().map(|&slot| (slot as i32) < 0).collect(),
        skipped_assignments: 0,
        skipped_expert_groups: 0,
    };
    if min_rank == 0 || min_rank > top_k { return Some(result); }
    // One best rank and resident marker per expert across ALL rows of the pass.
    // The resident marker is conservative for inconsistent cross-row tables too.
    let mut best: HashMap<u32, (usize, bool)> = HashMap::with_capacity(ids.len());
    for (j, (&expert, &slot)) in ids.iter().zip(slots).enumerate() {
        let item = best.entry(expert).or_insert((usize::MAX, false));
        item.0 = item.0.min(j % top_k + 1);
        item.1 |= (slot as i32) >= 0;
    }
    result.skipped_expert_groups = best.values().filter(|&&(rank, resident)| rank >= min_rank && !resident).count() as u64;
    for (j, expert) in ids.iter().enumerate() {
        let &(rank, resident) = &best[expert];
        if result.compute[j] && rank >= min_rank && !resident {
            result.compute[j] = false;
            result.skipped_assignments += 1;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    const MISS: u32 = (-1i32) as u32;

    #[test]
    fn exact_default_and_prefill_and_draft_are_ineligible() {
        assert!(!Policy::default().applies(Pass::Decode));
        let policy = Policy::parse("8").unwrap();
        assert!(policy.applies(Pass::Decode));
        assert!(!policy.applies(Pass::Prefill));
        assert!(!policy.applies(Pass::Draft));
        for value in ["-1", "257", "no"] { assert!(Policy::parse(value).is_err()); }
    }

    #[test]
    fn rank_is_one_based_and_original_top_rank_misses_remain() {
        let p = plan(&[10, 11, 12, 13], &[MISS; 4], 4, 3).unwrap();
        assert_eq!(p.compute, [true, true, false, false]);
        assert_eq!((p.skipped_assignments, p.skipped_expert_groups), (2, 2));
    }

    #[test]
    fn one_high_rank_in_any_verify_row_preserves_every_use() {
        let ids = [10, 11, 12, 12, 11, 13];
        let p = plan(&ids, &[MISS; 6], 3, 3).unwrap();
        assert_eq!(p.compute, [true, true, true, true, true, false]);
        assert_eq!((p.skipped_assignments, p.skipped_expert_groups), (1, 1));
    }

    #[test]
    fn cached_assignments_are_never_treated_as_cpu_work_or_skipped() {
        let p = plan(&[10, 11, 12, 13], &[MISS, MISS, 0, 7], 4, 3).unwrap();
        assert_eq!(p.compute, [true, true, false, false]);
        assert_eq!((p.skipped_assignments, p.skipped_expert_groups), (0, 0));
        // A resident observation anywhere protects the same expert elsewhere.
        let p = plan(&[10, 12, 11, 12], &[MISS, 0, MISS, MISS], 2, 2).unwrap();
        assert_eq!(p.compute, [true, false, true, true]);
        assert_eq!(p.skipped_assignments, 0);
    }

    #[test]
    fn repeated_tail_expert_counts_one_group_and_multiple_assignments() {
        let p = plan(&[10, 11, 12, 13, 14, 12], &[MISS; 6], 3, 3).unwrap();
        assert_eq!(p.compute, [true, true, false, true, true, false]);
        assert_eq!((p.skipped_assignments, p.skipped_expert_groups), (2, 1));
    }

    #[test]
    fn omitted_contributions_are_zero_without_renormalising_survivors() {
        let weights = [0.6f32, 0.3, 0.1];
        let expert_outputs = [2.0f32, 5.0, 20.0];
        let sum = |p: &Plan| weights.iter().zip(expert_outputs).zip(&p.compute)
            .filter(|(_, keep)| **keep).map(|((&weight, value), _)| weight * value).sum::<f32>();
        let exact = plan(&[0, 1, 2], &[MISS; 3], 3, 0).unwrap();
        let tail = plan(&[0, 1, 2], &[MISS; 3], 3, 3).unwrap();
        let none = plan(&[0, 1, 2], &[MISS; 3], 3, 1).unwrap();
        assert!((sum(&exact) - 4.7).abs() < 1e-6);
        assert!((sum(&tail) - 2.7).abs() < 1e-6);
        assert_eq!(sum(&none), 0.0);
    }

    #[test]
    fn inactive_threshold_and_bad_geometry_cannot_omit_experts() {
        for min_rank in [0, 4] {
            let p = plan(&[0, 1, 2], &[MISS, 0, MISS], 3, min_rank).unwrap();
            assert_eq!(p.compute, [true, false, true]);
            assert_eq!(p.skipped_assignments, 0);
        }
        assert!(plan(&[1], &[MISS], 0, 1).is_none());
        assert!(plan(&[1], &[], 1, 1).is_none());
        assert!(plan(&[1], &[MISS], 2, 1).is_none());
    }
}
