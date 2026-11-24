//! Checked, static YaRN for ordinary split-half Qwen3 RoPE.
//!
//! Equations and defaults follow the public Hugging Face Transformers reference:
//! https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
//! Only the declared GGUF scaling is used; selecting a smaller request context
//! never changes the frequencies of an already configured model.

use crate::{gguf::Value, math};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct YarnScaling {
    pub factor: f32,
    /// Native context used in the YaRN correction ramp, before extension.
    pub original_context_length: usize,
    /// Multiplier on both sine and cosine, applied once to Q and K.
    pub attention_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finetuned: Option<bool>,
}

impl YarnScaling {
    pub fn inverse_frequencies(&self, dimension: usize, base: f32) -> Result<Vec<f32>> {
        ensure!(
            dimension > 0 && dimension % 2 == 0,
            "YaRN dimension must be positive and even"
        );
        ensure!(
            base.is_finite() && base > 1.0,
            "YaRN base must be finite and greater than one"
        );
        ensure!(
            self.factor.is_finite() && self.factor >= 1.0,
            "YaRN factor must be finite and at least one"
        );
        ensure!(
            self.original_context_length > 0,
            "YaRN original context must be positive"
        );
        ensure!(
            self.beta_fast == 32.0 && self.beta_slow == 1.0,
            "custom YaRN correction parameters are unsupported"
        );
        let correction = |rotations: f32| {
            dimension as f32
                * (self.original_context_length as f32 / (rotations * 2.0 * std::f32::consts::PI))
                    .ln()
                / (2.0 * base.ln())
        };
        let low = correction(self.beta_fast).floor().max(0.0);
        let high = correction(self.beta_slow)
            .ceil()
            .min(dimension as f32 - 1.0);
        // Equal bounds use the reference's small denominator; a range outside
        // the rotary dimensions is left outside rather than shifted inward.
        let high = if high == low { high + 0.001 } else { high };
        let frequencies = (0..dimension / 2)
            .map(|i| {
                let positional = base.powf((2 * i) as f32 / dimension as f32);
                let extrapolated = positional.recip();
                let interpolated = (self.factor * positional).recip();
                let extrapolation = 1.0 - ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                interpolated * (1.0 - extrapolation) + extrapolated * extrapolation
            })
            .collect::<Vec<_>>();
        math::finite(&frequencies, "YaRN inverse frequencies")?;
        Ok(frequencies)
    }
}

/// Parse the complete scaling namespace even for header-only catalogue checks.
/// Custom YaRN variants must not silently use these ordinary Qwen defaults.
pub(crate) fn from_gguf_metadata(
    metadata: &BTreeMap<String, Value>,
    architecture: &str,
    maximum_context: usize,
) -> Result<Option<YarnScaling>> {
    let prefix = format!("{architecture}.rope.scaling.");
    for key in metadata.keys().filter(|key| key.starts_with(&prefix)) {
        ensure!(
            matches!(
                &key[prefix.len()..],
                "type" | "factor" | "original_context_length" | "finetuned"
            ),
            "unsupported RoPE scaling metadata {key}"
        );
    }
    let key = |suffix: &str| format!("{prefix}{suffix}");
    let kind = match metadata.get(&key("type")) {
        None => "none",
        Some(value) => value
            .as_str()
            .with_context(|| format!("{}: expected a string", key("type")))?,
    };
    ensure!(
        matches!(kind, "none" | "yarn"),
        "unsupported RoPE scaling type {kind:?}"
    );
    let finetuned = metadata
        .get(&key("finetuned"))
        .map(|value| {
            value
                .as_bool()
                .with_context(|| format!("{}: expected a boolean", key("finetuned")))
        })
        .transpose()?;
    // Legacy unscaled metadata is accepted only when it truly has no effect.
    for suffix in ["rope.freq_scale", "rope.scale_linear"] {
        let legacy = format!("{architecture}.{suffix}");
        if let Some(value) = metadata.get(&legacy) {
            ensure!(
                value.as_f32() == Some(1.0),
                "unsupported scaled RoPE metadata {legacy}"
            );
        }
    }
    if kind == "none" {
        if let Some(value) = metadata.get(&key("factor")) {
            ensure!(
                value.as_f32() == Some(1.0),
                "{}: unscaled RoPE requires factor one",
                key("factor")
            );
        }
        ensure!(
            !metadata.contains_key(&key("original_context_length")),
            "{} requires YaRN scaling",
            key("original_context_length")
        );
        return Ok(None);
    }
    let factor = metadata
        .get(&key("factor"))
        .with_context(|| format!("{}: required YaRN metadata is missing", key("factor")))?
        .as_f32()
        .filter(|value| value.is_finite() && *value >= 1.0)
        .with_context(|| format!("{}: expected a finite factor at least one", key("factor")))?;
    let original_context_length = metadata
        .get(&key("original_context_length"))
        .with_context(|| {
            format!(
                "{}: required YaRN metadata is missing",
                key("original_context_length")
            )
        })?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0 && *value <= maximum_context)
        .with_context(|| {
            format!(
                "{}: expected a positive integer within the declared model context",
                key("original_context_length")
            )
        })?;
    ensure!(
        maximum_context as f64 <= original_context_length as f64 * factor as f64,
        "declared model context exceeds the configured YaRN extension"
    );
    let attention_factor = 1.0 + 0.1 * factor.ln();
    ensure!(
        attention_factor.is_finite(),
        "YaRN attention factor overflowed"
    );
    Ok(Some(YarnScaling {
        factor,
        original_context_length,
        attention_factor,
        beta_fast: 32.0,
        beta_slow: 1.0,
        finetuned,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> BTreeMap<String, Value> {
        BTreeMap::from([
            (
                "qwen3moe.rope.scaling.type".into(),
                Value::String("yarn".into()),
            ),
            ("qwen3moe.rope.scaling.factor".into(), Value::F32(4.0)),
            (
                "qwen3moe.rope.scaling.original_context_length".into(),
                Value::U32(262144),
            ),
        ])
    }

    #[test]
    fn qwen_coder_yarn_matches_fixed_reference_frequencies_and_long_positions() {
        let scaling = from_gguf_metadata(&metadata(), "qwen3moe", 1048576)
            .unwrap()
            .unwrap();
        let frequencies = scaling.inverse_frequencies(128, 10000000.0).unwrap();
        assert!((scaling.attention_factor - 1.138629436111989).abs() < 2e-7);
        // Fixed published-equation reference values, not values calculated by
        // the implementation under test. The selected pairs span both ramp
        // boundaries and the interpolated low-frequency tail.
        let reference = [
            (0, 1.0),
            (28, 0.0008659643233600654),
            (35, 0.00009655802111623038),
            (43, 0.000004952391946375847),
            (63, 0.000000032159923623424366),
        ];
        for (i, expected) in reference {
            assert!(
                (frequencies[i] as f64 - expected).abs() <= expected * 3e-7,
                "pair {i}: {} != {expected}",
                frequencies[i]
            );
        }
        let positions = [0usize, 262143, 262144, 524287];
        // Independent f64 trigonometry with the fixed frequencies; high-position
        // tolerance accounts for the reference runtime's f32 angle product.
        for position in positions {
            let mut cos = vec![0.0; 64];
            let mut sin = vec![0.0; 64];
            math::rope_angles_scaled(
                position,
                &frequencies,
                scaling.attention_factor,
                &mut cos,
                &mut sin,
            )
            .unwrap();
            for (i, frequency) in reference {
                let angle = position as f64 * frequency;
                let magnitude = 1.138629436111989f64;
                assert!(
                    (cos[i] as f64 - magnitude * angle.cos()).abs() < 0.0003,
                    "cos position {position}, pair {i}"
                );
                assert!(
                    (sin[i] as f64 - magnitude * angle.sin()).abs() < 0.0003,
                    "sin position {position}, pair {i}"
                );
            }
        }
    }

    #[test]
    fn invalid_and_unsupported_scaling_metadata_is_explicitly_rejected() {
        let parse =
            |metadata: &BTreeMap<String, Value>| from_gguf_metadata(metadata, "qwen3moe", 1048576);
        for suffix in ["factor", "original_context_length"] {
            let mut invalid = metadata();
            invalid.remove(&format!("qwen3moe.rope.scaling.{suffix}"));
            assert!(parse(&invalid).unwrap_err().to_string().contains(suffix));
        }
        for value in [
            Value::F32(0.0),
            Value::F32(0.5),
            Value::F32(f32::NAN),
            Value::F32(f32::INFINITY),
            Value::String("4".into()),
        ] {
            let mut invalid = metadata();
            invalid.insert("qwen3moe.rope.scaling.factor".into(), value);
            assert!(parse(&invalid).is_err());
        }
        for value in [
            Value::U32(0),
            Value::I32(-1),
            Value::F32(262144.0),
            Value::U32(2097152),
        ] {
            let mut invalid = metadata();
            invalid.insert(
                "qwen3moe.rope.scaling.original_context_length".into(),
                value,
            );
            assert!(parse(&invalid).is_err());
        }
        for suffix in [
            "attn_factor",
            "yarn_attn_factor",
            "yarn_beta_fast",
            "yarn_beta_slow",
            "yarn_ext_factor",
            "yarn_log_multiplier",
            "yarn_truncate",
            "unknown",
        ] {
            let mut invalid = metadata();
            invalid.insert(format!("qwen3moe.rope.scaling.{suffix}"), Value::F32(1.0));
            assert!(parse(&invalid).unwrap_err().to_string().contains(suffix));
        }
        for value in [Value::String("linear".into()), Value::U32(1)] {
            let mut invalid = metadata();
            invalid.insert("qwen3moe.rope.scaling.type".into(), value);
            assert!(parse(&invalid).is_err());
        }
        let mut invalid = metadata();
        invalid.insert("qwen3moe.rope.scaling.factor".into(), Value::F32(2.0));
        assert!(
            parse(&invalid)
                .unwrap_err()
                .to_string()
                .contains("extension")
        );
        let mut invalid = metadata();
        invalid.insert(
            "qwen3moe.rope.scaling.finetuned".into(),
            Value::String("true".into()),
        );
        assert!(
            parse(&invalid)
                .unwrap_err()
                .to_string()
                .contains("finetuned")
        );
    }
}
