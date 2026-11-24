//! Fixed Inkling 2 Mini loop schedule, NOT an inference implementation.
//!
//! Source design: 24 stored layers; layers 0..5 prelude, layers 6..17
//! evaluated three times, layers 18..23 epilogue (48 layer applications).
//! Learned cross-pass message mixing and attention/MLP residual gates are
//! essential: simply replaying Qwen blocks would be mathematically wrong.
//! This module only models the required scheduling and dependency graph.

use anyhow::{Result, ensure};
use serde::Serialize;

pub const STORED_LAYERS: usize = 24;
pub const PRELUDE_LAYERS: std::ops::Range<usize> = 0..6;
pub const LOOP_LAYERS: std::ops::Range<usize> = 6..18;
pub const EPILOGUE_LAYERS: std::ops::Range<usize> = 18..24;
pub const LOOP_PASSES: usize = 3;
pub const EXECUTED_LAYER_APPLICATIONS: usize =
    PRELUDE_LAYERS.end + LOOP_LAYERS.len() * LOOP_PASSES + EPILOGUE_LAYERS.len();

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Segment {
    Prelude,
    LoopedCore,
    Epilogue,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Visit {
    /// Index into the 24 stored layers. No new weights are created for loops.
    pub stored_layer: usize,
    pub segment: Segment,
    /// Zero-based repeated-pass index; zero for prelude and epilogue.
    pub pass: usize,
    /// A learned cross-pass message from an earlier loop pass is mandatory.
    pub requires_previous_pass_message: bool,
    /// This visit must use the trained gated attention/MLP residual graph.
    pub requires_learned_residual_gating: bool,
}

/// A dependency schedule, not numerically executable without trained
/// message projections, local/global attention and a correct KV state.
pub fn schedule(passes: usize) -> Result<Vec<Visit>> {
    // This model is trained for exactly 3 passes. Variable-pass execution
    // belongs to the separately released research checkpoint.
    ensure!(passes == LOOP_PASSES,
        "Inkling 2 Mini fixed checkpoint requires exactly 3 core passes");
    let mut visits=Vec::with_capacity(EXECUTED_LAYER_APPLICATIONS);
    for layer in PRELUDE_LAYERS {
        visits.push(Visit {
            stored_layer:layer,
            segment:Segment::Prelude,
            pass:0,
            requires_previous_pass_message:false,
            requires_learned_residual_gating:true,
        });
    }
    for pass in 0..LOOP_PASSES {
        for layer in LOOP_LAYERS {
            visits.push(Visit {
                stored_layer:layer,
                segment:Segment::LoopedCore,
                pass,
                requires_previous_pass_message:pass>0,
                requires_learned_residual_gating:true,
            });
        }
    }
    for layer in EPILOGUE_LAYERS {
        visits.push(Visit {
            stored_layer:layer,
            segment:Segment::Epilogue,
            pass:0,
            requires_previous_pass_message:false,
            requires_learned_residual_gating:true,
        });
    }
    ensure!(visits.len()==EXECUTED_LAYER_APPLICATIONS,
        "unexpected Inkling 2 schedule size");
    Ok(visits)
}

/// Compatibility response for future model discovery. Never mark execution
/// supported because a metadata string happens to be recognized.
pub fn compatibility_diagnostics(architecture: &str) -> serde_json::Value {
    serde_json::json!({
        "architecture":architecture,
        "recognized":architecture=="inkling2_loop",
        "native_inference_supported":false,
        "pretrained_model_verified":false,
        "fixed_loop_passes":LOOP_PASSES,
        "stored_layers":STORED_LAYERS,
        "scheduled_layer_applications":EXECUTED_LAYER_APPLICATIONS,
        "missing_before_execution":[
            "independent reference tensor-layout and tokenizer validation",
            "trained cross-pass residual/message gates and attention/MLP mixing",
            "local/global attention and pass-aware KV cache ownership",
            "NVFP4/MXFP8 or supported GGUF dequantization parity",
            "fixture-based CPU reference output, then Vulkan numerical parity",
            "model-level quality and RX 9070 XT physical-GPU benchmarks"
        ],
        "message":"Recognized architectural layout only. Nextora does not load or generate with Inkling 2 Mini."
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_model_uses_24_stored_layers_but_48_applications() {
        let visit=schedule(3).unwrap();
        assert_eq!(visit.len(),48);
        assert_eq!(visit.iter().filter(|v|v.segment==Segment::Prelude).count(),6);
        assert_eq!(visit.iter().filter(|v|v.segment==Segment::LoopedCore).count(),36);
        assert_eq!(visit.iter().filter(|v|v.segment==Segment::Epilogue).count(),6);
        for layer in 0..24 {
            assert_eq!(visit.iter().filter(|v|v.stored_layer==layer).count(),
                if (6..18).contains(&layer) {3} else {1});
        }
    }
    #[test]
    fn cross_pass_message_required_after_first_loop_pass() {
        let visit=schedule(3).unwrap();
        assert_eq!(visit.iter().filter(|v|v.requires_previous_pass_message).count(),24);
        assert!(visit.iter().all(|v|v.requires_learned_residual_gating));
        assert_eq!(visit[6].stored_layer,6);
        assert_eq!(visit[6].pass,0);
        assert_eq!(visit[18].stored_layer,6);
        assert_eq!(visit[18].pass,1);
        assert_eq!(visit[30].stored_layer,6);
        assert_eq!(visit[30].pass,2);
        assert_eq!(visit[42].stored_layer,18);
    }
    #[test]
    fn never_treat_other_pass_counts_as_supported() {
        for passes in [0,1,2,4,5,10] {
            assert!(schedule(passes).is_err());
        }
        let report=compatibility_diagnostics("inkling2_loop");
        assert_eq!(report["native_inference_supported"],false);
        assert_eq!(report["scheduled_layer_applications"],48);
    }
}
