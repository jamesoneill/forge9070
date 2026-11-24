//! Scalar reference operations for the independently implemented Forge CPU engine.
//!
//! Model semantics are the ordinary RMSNorm, split-half RoPE, causal grouped-query
//! attention and SwiGLU equations. Functions operate on caller-owned buffers so a
//! sequence can retain its scratch allocations between tokens. No upstream engine
//! code or backend is called here.

use anyhow::{Result, ensure};

/// Reject a broken activation rather than silently selecting a plausible token.
pub fn finite(values: &[f32], operation: &str) -> Result<()> {
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "{operation}: non-finite value"
    );
    Ok(())
}

/// RMSNorm with one learned scale per element; the mean uses the whole input row.
pub fn rms_norm(input: &[f32], scales: &[f32], epsilon: f32, out: &mut [f32]) -> Result<()> {
    ensure!(
        !input.is_empty() && input.len() == scales.len() && out.len() == input.len(),
        "RMSNorm: inconsistent or empty dimensions"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "RMSNorm: epsilon must be finite and positive"
    );
    finite(input, "RMSNorm input")?;
    let sum: f32 = input.iter().map(|x| x * x).sum();
    ensure!(
        sum.is_finite(),
        "RMSNorm: squared activation sum overflowed"
    );
    let inverse = (sum / input.len() as f32 + epsilon).sqrt().recip();
    for ((y, x), w) in out.iter_mut().zip(input).zip(scales) {
        *y = *x * inverse * *w;
    }
    finite(out, "RMSNorm output")
}

/// Independent per-head RMSNorm; scales are shared across heads, not head-specific.
pub fn rms_norm_heads(values: &mut [f32], scales: &[f32], epsilon: f32) -> Result<()> {
    ensure!(
        !scales.is_empty() && values.len() % scales.len() == 0,
        "QK RMSNorm: inconsistent dimensions"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "QK RMSNorm: invalid epsilon"
    );
    finite(values, "QK RMSNorm input")?;
    for head in values.chunks_exact_mut(scales.len()) {
        let sum: f32 = head.iter().map(|x| x * x).sum();
        ensure!(
            sum.is_finite(),
            "QK RMSNorm: squared activation sum overflowed"
        );
        let inverse = (sum / head.len() as f32 + epsilon).sqrt().recip();
        for (x, scale) in head.iter_mut().zip(scales) {
            *x *= inverse * *scale;
        }
    }
    finite(values, "QK RMSNorm output")
}

/// Ordinary (unscaled) RoPE frequencies, calculated once when a model loads.
pub fn rope_inverse_frequencies(dimension: usize, base: f32) -> Result<Vec<f32>> {
    ensure!(
        dimension > 0 && dimension % 2 == 0,
        "RoPE dimension must be positive and even"
    );
    ensure!(
        base.is_finite() && base > 0.0,
        "RoPE base must be finite and positive"
    );
    let frequencies: Vec<_> = (0..dimension / 2)
        .map(|i| base.powf(-((2 * i) as f32) / dimension as f32))
        .collect();
    finite(&frequencies, "RoPE inverse frequencies")?;
    Ok(frequencies)
}

/// Reused by every layer and both Q/K at the current token position.
pub fn rope_angles(
    position: usize,
    inverse: &[f32],
    cos: &mut [f32],
    sin: &mut [f32],
) -> Result<()> {
    rope_angles_scaled(position, inverse, 1.0, cos, sin)
}

/// Static YaRN scales sine and cosine once, before the shared Q/K rotations.
pub fn rope_angles_scaled(
    position: usize,
    inverse: &[f32],
    attention_factor: f32,
    cos: &mut [f32],
    sin: &mut [f32],
) -> Result<()> {
    ensure!(
        inverse.len() == cos.len() && cos.len() == sin.len(),
        "RoPE angle dimensions differ"
    );
    ensure!(
        attention_factor.is_finite() && attention_factor > 0.0,
        "RoPE attention factor must be finite and positive"
    );
    // f32 is the reference arithmetic here. Guard positions that could not be
    // represented exactly instead of silently conflating distant positions.
    ensure!(
        position <= 1 << 24,
        "RoPE position exceeds exact f32 integer range"
    );
    for ((frequency, c), s) in inverse.iter().zip(cos).zip(sin) {
        let angle = position as f32 * frequency;
        ensure!(angle.is_finite(), "RoPE angle overflowed");
        (*s, *c) = angle.sin_cos();
        if attention_factor != 1.0 {
            *s *= attention_factor;
            *c *= attention_factor;
        }
    }
    Ok(())
}

/// Split-half / NeoX layout: coordinate i rotates with i + rotary_dimension/2.
/// Coordinates beyond the rotary prefix, if any, are unchanged.
pub fn rope_heads(values: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) -> Result<()> {
    ensure!(
        head_dim > 0
            && values.len() % head_dim == 0
            && cos.len() == sin.len()
            && cos.len() * 2 <= head_dim,
        "RoPE head dimensions differ"
    );
    let half = cos.len();
    for head in values.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = b * cos[i] + a * sin[i];
        }
    }
    finite(values, "RoPE output")
}

pub fn softmax_in_place(values: &mut [f32]) -> Result<()> {
    ensure!(!values.is_empty(), "softmax requires at least one value");
    finite(values, "softmax input")?;
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    ensure!(sum.is_finite() && sum > 0.0, "softmax normalization failed");
    for value in values {
        *value /= sum;
    }
    Ok(())
}

/// Causal attention for the current token. KV is token-major, then head, then
/// coordinate. `tokens` includes the just-appended current key/value. Only that
/// prefix is read, so capacity reserved for future tokens cannot affect output.
pub fn attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) -> Result<()> {
    ensure!(
        tokens > 0
            && query_heads > 0
            && kv_heads > 0
            && head_dim > 0
            && query_heads % kv_heads == 0,
        "attention: invalid head/token geometry"
    );
    let query_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("attention width overflow"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("attention KV width overflow"))?;
    let kv_len = tokens
        .checked_mul(kv_width)
        .ok_or_else(|| anyhow::anyhow!("attention cache size overflow"))?;
    ensure!(
        query.len() == query_width
            && out.len() == query_width
            && keys.len() == kv_len
            && values.len() == kv_len,
        "attention: inconsistent cache/query dimensions"
    );
    scores.resize(tokens, 0.0);
    let scale = (head_dim as f32).sqrt().recip();
    let group = query_heads / kv_heads;
    for head in 0..query_heads {
        let q = &query[head * head_dim..(head + 1) * head_dim];
        let kv_head = head / group;
        for (position, score) in scores.iter_mut().enumerate() {
            let start = position * kv_width + kv_head * head_dim;
            *score = q
                .iter()
                .zip(&keys[start..start + head_dim])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
        }
        softmax_in_place(scores)?;
        let dest = &mut out[head * head_dim..(head + 1) * head_dim];
        dest.fill(0.0);
        for (position, weight) in scores.iter().enumerate() {
            let start = position * kv_width + kv_head * head_dim;
            for (y, v) in dest.iter_mut().zip(&values[start..start + head_dim]) {
                *y += weight * v;
            }
        }
    }
    finite(out, "attention output")
}

/// Replaces the gate vector with SiLU(gate) * up, retaining the same allocation.
pub fn swiglu(gate: &mut [f32], up: &[f32]) -> Result<()> {
    ensure!(gate.len() == up.len(), "SwiGLU dimensions differ");
    finite(gate, "SwiGLU gate")?;
    finite(up, "SwiGLU up")?;
    for (g, u) in gate.iter_mut().zip(up) {
        *g = (*g / (1.0 + (-*g).exp())) * *u;
    }
    finite(gate, "SwiGLU output")
}

/// Top-k routed experts, with a deterministic lower-index tie break. Normalized
/// routing computes a softmax over the selected logits; this is algebraically
/// equivalent to full-router softmax followed by top-k renormalization, without
/// losing tiny selected probabilities to underflow in the full softmax.
pub fn route_experts(
    logits: &[f32],
    count: usize,
    normalize: bool,
    scale: f32,
    selected: &mut Vec<(usize, f32)>,
    probabilities: &mut Vec<f32>,
) -> Result<()> {
    ensure!(
        count > 0 && count <= logits.len(),
        "router: invalid selected expert count"
    );
    ensure!(
        scale.is_finite() && scale > 0.0,
        "router: scale must be finite and positive"
    );
    finite(logits, "router logits")?;
    selected.clear();
    selected.extend(logits.iter().copied().enumerate());
    selected.sort_unstable_by(|a, b| {
        if a.1 == b.1 {
            a.0.cmp(&b.0)
        } else {
            b.1.total_cmp(&a.1)
        }
    });
    selected.truncate(count);
    probabilities.clear();
    if normalize {
        probabilities.extend(selected.iter().map(|x| x.1));
        softmax_in_place(probabilities)?;
        for ((_, weight), probability) in selected.iter_mut().zip(probabilities.iter()) {
            *weight = *probability * scale;
        }
    } else {
        probabilities.extend_from_slice(logits);
        softmax_in_place(probabilities)?;
        for (expert, weight) in selected.iter_mut() {
            *weight = probabilities[*expert] * scale;
        }
    }
    ensure!(
        selected.iter().all(|x| x.1.is_finite()),
        "router weights overflowed"
    );
    Ok(())
}

pub fn residual_add(residual: &mut [f32], update: &[f32]) -> Result<()> {
    ensure!(residual.len() == update.len(), "residual dimensions differ");
    for (x, delta) in residual.iter_mut().zip(update) {
        *x += delta;
    }
    finite(residual, "residual output")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_rope_angles_apply_magnitude_once_and_keep_unscaled_output_identical() {
        let frequencies = rope_inverse_frequencies(128, 10000000.0).unwrap();
        let magnitude = 1.1386294;
        for position in [0, 262143, 262144, 524287] {
            let mut cos = vec![0.0; 64];
            let mut sin = vec![0.0; 64];
            rope_angles(position, &frequencies, &mut cos, &mut sin).unwrap();
            let mut identical_cos = vec![0.0; 64];
            let mut identical_sin = vec![0.0; 64];
            rope_angles_scaled(
                position,
                &frequencies,
                1.0,
                &mut identical_cos,
                &mut identical_sin,
            )
            .unwrap();
            assert_eq!(cos, identical_cos);
            assert_eq!(sin, identical_sin);
            rope_angles_scaled(
                position,
                &frequencies,
                magnitude,
                &mut identical_cos,
                &mut identical_sin,
            )
            .unwrap();
            for i in 0..64 {
                assert_eq!(identical_cos[i], cos[i] * magnitude);
                assert_eq!(identical_sin[i], sin[i] * magnitude);
            }
        }
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                rope_angles_scaled(0, &frequencies, invalid, &mut [0.0; 64], &mut [0.0; 64])
                    .is_err()
            );
        }
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 2e-6, "{a} != {b}");
        }
    }

    #[test]
    fn rms_norm_is_per_head_and_uses_learned_scale() {
        let mut heads = vec![3.0, 4.0, 30.0, 40.0];
        rms_norm_heads(&mut heads, &[2.0, 0.5], 1e-6).unwrap();
        let inv = (12.5f32 + 1e-6).sqrt().recip();
        close(&heads[..2], &[6.0 * inv, 2.0 * inv]);
        close(&heads[..2], &heads[2..]);
    }

    #[test]
    fn rope_uses_split_halves_and_preserves_unrotated_tail() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 19.0, 23.0];
        rope_heads(&mut x, 6, &[0.0, 1.0], &[1.0, 0.0]).unwrap();
        close(&x, &[-3.0, 2.0, 1.0, 4.0, 19.0, 23.0]);
    }

    #[test]
    fn rope_position_zero_is_identity_and_next_frequency_is_correct() {
        let frequencies = rope_inverse_frequencies(4, 10000.0).unwrap();
        close(&frequencies, &[1.0, 0.01]);
        let mut cos = vec![0.0; 2];
        let mut sin = vec![0.0; 2];
        rope_angles(0, &frequencies, &mut cos, &mut sin).unwrap();
        close(&cos, &[1.0, 1.0]);
        close(&sin, &[0.0, 0.0]);
    }

    #[test]
    fn gqa_maps_each_group_to_its_own_kv_head() {
        let mut scores = vec![];
        let mut out = vec![0.0; 4];
        attention(
            &[0.0; 4],
            &[2.0, 3.0, 4.0, 5.0],
            &[10.0, 100.0, 30.0, 300.0],
            2,
            4,
            2,
            1,
            &mut scores,
            &mut out,
        )
        .unwrap();
        close(&out, &[20.0, 20.0, 200.0, 200.0]);
    }

    #[test]
    fn attention_is_a_scaled_softmax_weighted_sum() {
        let mut out = vec![0.0; 2];
        let mut scores = vec![];
        attention(
            &[1.0, 0.0],
            &[0.0, 0.0, 2.0f32.sqrt(), 0.0],
            &[2.0, 4.0, 8.0, 10.0],
            2,
            1,
            1,
            2,
            &mut scores,
            &mut out,
        )
        .unwrap();
        let p = 1.0 / (1.0 + 1.0f32.exp());
        close(
            &out,
            &[2.0 * p + 8.0 * (1.0 - p), 4.0 * p + 10.0 * (1.0 - p)],
        );
    }

    #[test]
    fn router_renormalizes_selected_experts_and_keeps_ties_stable() {
        let mut selected = vec![];
        let mut scratch = vec![];
        route_experts(
            &[0.0, 2.0, 2.0, -1000.0],
            2,
            true,
            1.0,
            &mut selected,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(selected, vec![(1, 0.5), (2, 0.5)]);
        route_experts(
            &[0.0, 2.0, 2.0, -1000.0],
            1,
            true,
            1.0,
            &mut selected,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(selected, vec![(1, 1.0)]);
        route_experts(&[-0.0, 0.0], 1, true, 1.0, &mut selected, &mut scratch).unwrap();
        assert_eq!(selected, vec![(0, 1.0)]);
    }

    #[test]
    fn unnormalized_router_retains_full_softmax_mass() {
        let mut selected = vec![];
        let mut scratch = vec![];
        route_experts(
            &[0.0, 0.0, 0.0, 0.0],
            2,
            false,
            1.5,
            &mut selected,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(selected, vec![(0, 0.375), (1, 0.375)]);
    }

    #[test]
    fn nonlinear_and_probability_operations_reject_nonfinite_data() {
        assert!(softmax_in_place(&mut [0.0, f32::NAN]).is_err());
        assert!(rms_norm(&[f32::MAX], &[1.0], 1e-6, &mut [0.0]).is_err());
        assert!(swiglu(&mut [f32::NAN], &[0.0]).is_err());
        assert!(
            route_experts(
                &[0.0, f32::INFINITY],
                1,
                true,
                1.0,
                &mut vec![],
                &mut vec![]
            )
            .is_err()
        );
        assert!(attention(&[1.0], &[1.0], &[1.0], 1, 1, 0, 1, &mut vec![], &mut [0.0]).is_err());
    }
}
