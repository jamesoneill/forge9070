//! Independently implemented GGML storage decoding and scalar/SIMD block dots.
//!
//! Format references (layout/interoperability, not imported implementation):
//! https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//! https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c
//! The public format IDs, IEEE encodings and Q4_0/Q8_0 block layouts are used
//! here. This module contains no RustingStrata code or execution dependency.

use anyhow::{bail, ensure, Context, Result};
use half::f16;
use serde::Serialize;

#[path = "quant_tables.rs"]
mod tables;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q2_0,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
}

impl TryFrom<u32> for QuantType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        Ok(match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            30 => Self::BF16,
            42 => Self::Q2_0,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            22 => Self::IQ2_S,
            18 => Self::IQ3_XXS,
            21 => Self::IQ3_S,
            20 => Self::IQ4_NL,
            23 => Self::IQ4_XS,
            other => bail!(
                "GGML tensor type {other} is unsupported by the independent Forge engine; supported: F32, F16, BF16, Q4_0, Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS"
            ),
        })
    }
}

impl QuantType {
    pub const fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::BF16 => 30,
            Self::Q2_0 => 42,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::IQ2_S => 22,
            Self::IQ3_XXS => 18,
            Self::IQ3_S => 21,
            Self::IQ4_NL => 20,
            Self::IQ4_XS => 23,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q2_0 => "Q2_0",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::IQ2_S => "IQ2_S",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ3_S => "IQ3_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ4_XS => "IQ4_XS",
        }
    }

    pub const fn block_elements(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q8_0 | Self::IQ4_NL => 32,
            Self::Q2_0 => 64,
            Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ4_XS => 256,
            _ => 1,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q2_0 | Self::IQ4_NL => 18,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::IQ2_S => 82,
            Self::IQ3_XXS => 98,
            Self::IQ3_S => 110,
            Self::IQ4_XS => 136,
        }
    }

    /// A row must contain complete storage blocks. No implicit padding or
    /// rounding is allowed: that could change the address of every later row.
    pub fn row_bytes(self, width: usize) -> Result<usize> {
        ensure!(width > 0, "tensor row width is zero");
        ensure!(
            width % self.block_elements() == 0,
            "{} row width {width} is not divisible by {}",
            self.name(),
            self.block_elements()
        );
        (width / self.block_elements())
            .checked_mul(self.block_bytes())
            .context("tensor row byte count overflow")
    }
}

#[inline]
fn little_half(bytes: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

/// Decode exactly one storage row into caller-owned scratch. The output and
/// byte lengths must match exactly; non-finite model values are errors.
pub fn decode_row(ty: QuantType, bytes: &[u8], output: &mut [f32]) -> Result<()> {
    ensure!(
        bytes.len() == ty.row_bytes(output.len())?,
        "tensor row byte length mismatch"
    );
    match ty {
        QuantType::F32 => {
            for (out, word) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                *out = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            }
        }
        QuantType::F16 => {
            for (out, word) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                *out = little_half(word);
            }
        }
        QuantType::BF16 => {
            for (out, word) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                *out = f32::from_bits((u16::from_le_bytes([word[0], word[1]]) as u32) << 16);
            }
        }
        QuantType::Q4_0 => {
            for (out, block) in output.chunks_exact_mut(32).zip(bytes.chunks_exact(18)) {
                let scale = little_half(block);
                for (i, &packed) in block[2..].iter().enumerate() {
                    out[i] = scale * (i32::from(packed & 15) - 8) as f32;
                    out[i + 16] = scale * (i32::from(packed >> 4) - 8) as f32;
                }
            }
        }
        QuantType::Q8_0 => {
            for (out, block) in output.chunks_exact_mut(32).zip(bytes.chunks_exact(34)) {
                let scale = little_half(block);
                for (value, &quantized) in out.iter_mut().zip(&block[2..]) {
                    *value = scale * (quantized as i8) as f32;
                }
            }
        }
        _ => {
            for (values, block) in output
                .chunks_exact_mut(ty.block_elements())
                .zip(bytes.chunks_exact(ty.block_bytes()))
            {
                decode_extended_block(ty, block, values);
            }
        }
    }
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "non-finite {} tensor value",
        ty.name()
    );
    Ok(())
}

/// Direct dot product over stored blocks. There is no row-sized allocation or
/// whole-model conversion to F32. The public entry also validates its input.
pub fn dot(ty: QuantType, bytes: &[u8], input: &[f32]) -> Result<f32> {
    ensure!(
        input.iter().all(|v| v.is_finite()),
        "non-finite matvec input"
    );
    dot_finite_input(ty, bytes, input)
}

/// Portable accumulation used as an independent numerical oracle for the
/// runtime-selected vector implementation and for machine-local comparisons.
pub fn dot_scalar(ty: QuantType, bytes: &[u8], input: &[f32]) -> Result<f32> {
    ensure!(
        input.iter().all(|v| v.is_finite()),
        "non-finite matvec input"
    );
    dot_scalar_finite(ty, bytes, input)
}

/// Report the kernel that a row of this shape will actually use. Small rows
/// remain scalar; vector F16 conversion additionally requires F16C.
pub fn dot_kernel(ty: QuantType, width: usize) -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if width >= 256
        && matches!(ty, QuantType::Q4_K | QuantType::Q6_K)
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("fma")
    {
        return "avx512f_bw_dq_fma_k_quant";
    }
    #[cfg(target_arch = "x86_64")]
    if width >= 128
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        return if matches!(ty, QuantType::Q4_K | QuantType::Q6_K) {
            "avx2_f64_k_quant"
        } else if ty == QuantType::F16 {
            if std::arch::is_x86_feature_detected!("f16c") {
                "avx2_fma_f16c"
            } else {
                "scalar"
            }
        } else {
            "avx2_fma"
        };
    }
    if matches!(ty, QuantType::Q4_K | QuantType::Q6_K) {
        "portable_f64_k_quant"
    } else {
        "scalar"
    }
}

/// The matrix entry point checks the activation vector once, before dispatching
/// rows. This internal function still checks all dimensions and final results.
pub(crate) fn dot_finite_input(ty: QuantType, bytes: &[u8], input: &[f32]) -> Result<f32> {
    ensure!(
        bytes.len() == ty.row_bytes(input.len())?,
        "dot-product row byte length mismatch"
    );
    // The dispatch checks each ISA before entering its target_feature function.
    // All loads operate on complete blocks validated by row_bytes above.
    let value = match dot_kernel(ty, input.len()) {
        #[cfg(target_arch = "x86_64")]
        "avx512f_bw_dq_fma_k_quant" => unsafe { dot_k_avx512(ty, bytes, input) },
        #[cfg(target_arch = "x86_64")]
        "avx2_f64_k_quant" => unsafe { dot_k_avx2_wide(ty, bytes, input) },
        #[cfg(target_arch = "x86_64")]
        "avx2_fma_f16c" => unsafe { dot_f16_avx2_fma_f16c(bytes, input) },
        #[cfg(target_arch = "x86_64")]
        "avx2_fma" => unsafe { dot_avx2_fma(ty, bytes, input) },
        "portable_f64_k_quant" => dot_k_wide(ty, bytes, input),
        _ => return dot_scalar_finite(ty, bytes, input),
    };
    ensure!(value.is_finite(), "non-finite {} dot-product result", ty.name());
    Ok(value)
}

/// The previous checked AVX2 path, retained for paired CPU-kernel measurements.
/// Unsupported hosts use the same portable scalar reference.
pub(crate) fn dot_avx2_baseline(ty: QuantType, bytes: &[u8], input: &[f32]) -> Result<f32> {
    ensure!(
        input.iter().all(|v| v.is_finite()),
        "non-finite matvec input"
    );
    ensure!(
        bytes.len() == ty.row_bytes(input.len())?,
        "dot-product row byte length mismatch"
    );
    #[cfg(target_arch = "x86_64")]
    if input.len() >= 128
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && (ty != QuantType::F16 || std::arch::is_x86_feature_detected!("f16c"))
    {
        // SAFETY: all features, exact row size and block geometry are checked.
        let value = unsafe {
            if ty == QuantType::F16 {
                dot_f16_avx2_fma_f16c(bytes, input)
            } else {
                dot_avx2_fma(ty, bytes, input)
            }
        };
        ensure!(
            value.is_finite(),
            "non-finite {} dot-product result",
            ty.name()
        );
        return Ok(value);
    }
    dot_scalar_finite(ty, bytes, input)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,fma")]
unsafe fn dot_k_avx512(ty: QuantType, bytes: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: callers check all ISA features and complete 256-value blocks.
    // Each unaligned integer load reads exactly sixteen bytes from that block;
    // each activation load reads sixteen values from its corresponding group.
    unsafe {
        let mut first = _mm512_setzero_ps();
        let mut second = _mm512_setzero_ps();
        let nibble = _mm_set1_epi8(15);
        match ty {
            QuantType::Q4_K => {
                for (block, values) in bytes.chunks_exact(144).zip(input.chunks_exact(256)) {
                    let d = little_half(block);
                    let minimum = little_half(&block[2..]);
                    let scales = &block[4..16];
                    for group in 0..8 {
                        let (scale, min) = if group < 4 {
                            (scales[group] & 63, scales[group + 4] & 63)
                        } else {
                            (
                                (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                            )
                        };
                        let scale = _mm512_set1_ps(d * f32::from(scale));
                        let min = _mm512_set1_ps(minimum * f32::from(min));
                        let packed = block.as_ptr().add(16 + (group / 2) * 32);
                        for half in 0..2 {
                            let bits = _mm_loadu_si128(packed.add(half * 16).cast());
                            let bits = if group % 2 == 0 {
                                bits
                            } else {
                                _mm_srli_epi16::<4>(bits)
                            };
                            let codes = _mm512_cvtepu8_epi32(_mm_and_si128(bits, nibble));
                            // Keep format reconstruction separate from the FMA;
                            // scale*code-min uses the scalar decoder's rounding.
                            let weights =
                                _mm512_sub_ps(_mm512_mul_ps(scale, _mm512_cvtepi32_ps(codes)), min);
                            let x = _mm512_loadu_ps(values.as_ptr().add(group * 32 + half * 16));
                            if half == 0 {
                                first = _mm512_fmadd_ps(weights, x, first);
                            } else {
                                second = _mm512_fmadd_ps(weights, x, second);
                            }
                        }
                    }
                }
            }
            QuantType::Q6_K => {
                let two_bits = _mm_set1_epi8(3);
                for (block, values) in bytes.chunks_exact(210).zip(input.chunks_exact(256)) {
                    let d = little_half(&block[208..]);
                    for half in 0..2 {
                        for quadrant in 0..4 {
                            for part in 0..2 {
                                let lane = part * 16;
                                let lower = _mm_loadu_si128(
                                    block
                                        .as_ptr()
                                        .add(half * 64 + (quadrant % 2) * 32 + lane)
                                        .cast(),
                                );
                                let lower = if quadrant < 2 {
                                    lower
                                } else {
                                    _mm_srli_epi16::<4>(lower)
                                };
                                let higher = _mm_loadu_si128(
                                    block.as_ptr().add(128 + half * 32 + lane).cast(),
                                );
                                let higher = match quadrant {
                                    0 => higher,
                                    1 => _mm_srli_epi16::<2>(higher),
                                    2 => _mm_srli_epi16::<4>(higher),
                                    _ => _mm_srli_epi16::<6>(higher),
                                };
                                let codes = _mm_or_si128(
                                    _mm_and_si128(lower, nibble),
                                    _mm_slli_epi16::<4>(_mm_and_si128(higher, two_bits)),
                                );
                                let codes = _mm512_sub_epi32(
                                    _mm512_cvtepu8_epi32(codes),
                                    _mm512_set1_epi32(32),
                                );
                                let scale = d * f32::from(
                                    block[192 + half * 8 + quadrant * 2 + part] as i8,
                                );
                                let weights =
                                    _mm512_mul_ps(_mm512_set1_ps(scale), _mm512_cvtepi32_ps(codes));
                                let x = _mm512_loadu_ps(
                                    values.as_ptr().add(half * 128 + quadrant * 32 + lane),
                                );
                                if part == 0 {
                                    first = _mm512_fmadd_ps(weights, x, first);
                                } else {
                                    second = _mm512_fmadd_ps(weights, x, second);
                                }
                            }
                        }
                    }
                }
            }
            _ => unreachable!("AVX512 K-quant dispatch only accepts Q4_K/Q6_K"),
        }
        _mm512_reduce_add_ps(_mm512_add_ps(first, second))
    }
}

// Keep the cancellation-sensitive K-format fallback accurate on hosts without
// AVX-512. Independent double lanes let LLVM vectorize the block reduction.
#[inline(always)]
fn dot_k_wide(ty: QuantType, bytes: &[u8], input: &[f32]) -> f32 {
    let mut sums = [0.0f64; 4];
    let mut decoded = [0.0f32; 256];
    for (block, values) in bytes.chunks_exact(ty.block_bytes()).zip(input.chunks_exact(256)) {
        decode_extended_block(ty, block, &mut decoded);
        for (weights, activations) in decoded.chunks_exact(4).zip(values.chunks_exact(4)) {
            for lane in 0..4 {
                sums[lane] += f64::from(weights[lane]) * f64::from(activations[lane]);
            }
        }
    }
    sums.into_iter().sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_k_avx2_wide(ty: QuantType, bytes: &[u8], input: &[f32]) -> f32 {
    dot_k_wide(ty, bytes, input)
}

fn dot_scalar_finite(ty: QuantType, bytes: &[u8], input: &[f32]) -> Result<f32> {
    ensure!(
        bytes.len() == ty.row_bytes(input.len())?,
        "dot-product row byte length mismatch"
    );
    let mut sum = 0.0f32;
    match ty {
        QuantType::F32 => {
            for (word, &x) in bytes.chunks_exact(4).zip(input) {
                let w = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                sum += w * x;
            }
        }
        QuantType::F16 => {
            for (word, &x) in bytes.chunks_exact(2).zip(input) {
                sum += little_half(word) * x;
            }
        }
        QuantType::BF16 => {
            for (word, &x) in bytes.chunks_exact(2).zip(input) {
                let w = f32::from_bits((u16::from_le_bytes([word[0], word[1]]) as u32) << 16);
                sum += w * x;
            }
        }
        QuantType::Q4_0 => {
            for (block, activation) in bytes.chunks_exact(18).zip(input.chunks_exact(32)) {
                let scale = little_half(block);
                // Two passes preserve storage row order, making the scalar
                // reference and the direct dot's accumulation order agree.
                for (packed, &x) in block[2..].iter().zip(&activation[..16]) {
                    sum += (scale * (i32::from(*packed & 15) - 8) as f32) * x;
                }
                for (packed, &x) in block[2..].iter().zip(&activation[16..]) {
                    sum += (scale * (i32::from(*packed >> 4) - 8) as f32) * x;
                }
            }
        }
        QuantType::Q8_0 => {
            for (block, activation) in bytes.chunks_exact(34).zip(input.chunks_exact(32)) {
                let scale = little_half(block);
                for (&quantized, &x) in block[2..].iter().zip(activation) {
                    sum += (scale * (quantized as i8) as f32) * x;
                }
            }
        }
        _ => {
            let mut decoded = [0.0f32; 256];
            let width = ty.block_elements();
            for (block, values) in bytes
                .chunks_exact(ty.block_bytes())
                .zip(input.chunks_exact(width))
            {
                decode_extended_block(ty, block, &mut decoded[..width]);
                for (w, x) in decoded[..width].iter().zip(values) {
                    sum += w * x;
                }
            }
        }
    }
    ensure!(
        sum.is_finite(),
        "non-finite {} dot-product result",
        ty.name()
    );
    Ok(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(ty: QuantType, bytes: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller verifies AVX2/FMA and the exact encoded row length.
    // Loads use unaligned intrinsics. No references to reinterpreted weights
    // are created, and no access passes either provided slice's endpoint.
    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        let mut tail = 0.0f32;
        match ty {
            QuantType::F32 => {
                let vector_end = input.len() / 8 * 8;
                for offset in (0..vector_end).step_by(8) {
                    let weights = _mm256_loadu_ps(bytes.as_ptr().add(offset * 4).cast());
                    let values = _mm256_loadu_ps(input.as_ptr().add(offset));
                    accumulator = _mm256_fmadd_ps(weights, values, accumulator);
                }
                for offset in vector_end..input.len() {
                    let start = offset * 4;
                    let word = &bytes[start..start + 4];
                    tail +=
                        f32::from_le_bytes([word[0], word[1], word[2], word[3]]) * input[offset];
                }
            }
            QuantType::BF16 => {
                let vector_end = input.len() / 8 * 8;
                for offset in (0..vector_end).step_by(8) {
                    let packed = _mm_loadu_si128(bytes.as_ptr().add(offset * 2).cast());
                    let words = _mm256_cvtepu16_epi32(packed);
                    let weights = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(words));
                    let values = _mm256_loadu_ps(input.as_ptr().add(offset));
                    accumulator = _mm256_fmadd_ps(weights, values, accumulator);
                }
                for offset in vector_end..input.len() {
                    let word = u16::from_le_bytes([bytes[offset * 2], bytes[offset * 2 + 1]]);
                    tail += f32::from_bits((word as u32) << 16) * input[offset];
                }
            }
            QuantType::Q8_0 => {
                for (block, values) in bytes.chunks_exact(34).zip(input.chunks_exact(32)) {
                    let scale = _mm256_set1_ps(little_half(block));
                    for offset in (0..32).step_by(8) {
                        let packed = _mm_loadl_epi64(block.as_ptr().add(2 + offset).cast());
                        let signed = _mm256_cvtepi8_epi32(packed);
                        let weights = _mm256_mul_ps(_mm256_cvtepi32_ps(signed), scale);
                        let activation = _mm256_loadu_ps(values.as_ptr().add(offset));
                        accumulator = _mm256_fmadd_ps(weights, activation, accumulator);
                    }
                }
            }
            QuantType::Q4_0 => {
                let mask = _mm256_set1_epi32(15);
                let zero = _mm256_set1_epi32(8);
                for (block, values) in bytes.chunks_exact(18).zip(input.chunks_exact(32)) {
                    let scale = _mm256_set1_ps(little_half(block));
                    for offset in [0usize, 8] {
                        let packed = _mm_loadl_epi64(block.as_ptr().add(2 + offset).cast());
                        let words = _mm256_cvtepu8_epi32(packed);
                        let lower = _mm256_sub_epi32(_mm256_and_si256(words, mask), zero);
                        let upper = _mm256_sub_epi32(_mm256_srli_epi32::<4>(words), zero);
                        let lo_weights = _mm256_mul_ps(_mm256_cvtepi32_ps(lower), scale);
                        let hi_weights = _mm256_mul_ps(_mm256_cvtepi32_ps(upper), scale);
                        accumulator = _mm256_fmadd_ps(
                            lo_weights,
                            _mm256_loadu_ps(values.as_ptr().add(offset)),
                            accumulator,
                        );
                        accumulator = _mm256_fmadd_ps(
                            hi_weights,
                            _mm256_loadu_ps(values.as_ptr().add(16 + offset)),
                            accumulator,
                        );
                    }
                }
            }
            QuantType::F16 => unreachable!("F16 has a separate F16C vector kernel"),
            _ => {
                // One fixed stack block, never a row or model-sized allocation.
                let mut decoded = [0.0f32; 256];
                let width = ty.block_elements();
                for (block, values) in bytes
                    .chunks_exact(ty.block_bytes())
                    .zip(input.chunks_exact(width))
                {
                    decode_extended_block(ty, block, &mut decoded[..width]);
                    for offset in (0..width).step_by(8) {
                        let weights = _mm256_loadu_ps(decoded.as_ptr().add(offset));
                        let activation = _mm256_loadu_ps(values.as_ptr().add(offset));
                        accumulator = _mm256_fmadd_ps(weights, activation, accumulator);
                    }
                }
            }
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        lanes.into_iter().sum::<f32>() + tail
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_avx2_fma_f16c(bytes: &[u8], input: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller checks AVX2/FMA/F16C and bytes.len()==input.len()*2.
    // Each full iteration reads exactly eight half words and eight activations
    // using unaligned loads. The remaining at-most-seven half words use safe
    // scalar indexing; no load can cross either input slice's end.
    unsafe {
        let mut accumulator = _mm256_setzero_ps();
        let vector_end = input.len() / 8 * 8;
        for offset in (0..vector_end).step_by(8) {
            let packed = _mm_loadu_si128(bytes.as_ptr().add(offset * 2).cast());
            let weights = _mm256_cvtph_ps(packed);
            let activation = _mm256_loadu_ps(input.as_ptr().add(offset));
            accumulator = _mm256_fmadd_ps(weights, activation, accumulator);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        let mut sum = lanes.into_iter().sum::<f32>();
        for offset in vector_end..input.len() {
            sum += little_half(&bytes[offset * 2..offset * 2 + 2]) * input[offset];
        }
        sum
    }
}

/// Decode one complete extended GGML block into fixed caller scratch. Public
/// entry points have already checked exact byte lengths and storage geometry.
/// Codebook bytes are MIT-licensed GGML format data; see GGML-NOTICE.md.
fn decode_extended_block(ty: QuantType, b: &[u8], out: &mut [f32]) {
    use tables::*;
    match ty {
        QuantType::Q2_0 => {
            let d = little_half(b);
            for (i, value) in out.iter_mut().enumerate() {
                *value = d * (((b[2 + i / 4] >> (2 * (i % 4))) & 3) as i32 - 1) as f32;
            }
        }
        QuantType::IQ4_NL => {
            let d = little_half(b);
            for (i, value) in out.iter_mut().enumerate() {
                let code = (b[2 + i % 16] >> (4 * (i / 16))) & 15;
                *value = d * KVALUES_IQ4NL[code as usize] as f32;
            }
        }
        QuantType::Q3_K => {
            let d = little_half(&b[108..]);
            for (i, value) in out.iter_mut().enumerate() {
                let group = i / 16;
                let low = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
                let high = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
                let scale = (low | (high << 4)) as i32 - 32;
                let packed = b[32 + (i / 128) * 32 + i % 32];
                let lower = (packed >> (2 * ((i % 128) / 32))) & 3;
                let negative = (b[i % 32] >> (i / 32)) & 1 == 0;
                let code = lower as i32 - if negative { 4 } else { 0 };
                *value = (d * scale as f32) * code as f32;
            }
        }
        QuantType::Q4_K | QuantType::Q5_K => {
            let d = little_half(b);
            let minimum = little_half(&b[2..]);
            let scales = &b[4..16];
            let five = ty == QuantType::Q5_K;
            let packed = &b[if five { 48 } else { 16 }..];
            for (i, value) in out.iter_mut().enumerate() {
                let group = i / 32;
                let lane = i % 32;
                let (scale, min) = if group < 4 {
                    (scales[group] & 63, scales[group + 4] & 63)
                } else {
                    (
                        (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                        (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                    )
                };
                let mut code = (packed[(group / 2) * 32 + lane] >> (4 * (group % 2))) & 15;
                if five {
                    code |= ((b[16 + lane] >> group) & 1) << 4;
                }
                *value = (d * scale as f32) * code as f32 - minimum * min as f32;
            }
        }
        QuantType::Q6_K => {
            let d = little_half(&b[208..]);
            for (i, value) in out.iter_mut().enumerate() {
                let half = i / 128;
                let quadrant = (i % 128) / 32;
                let lane = i % 32;
                let lower =
                    (b[half * 64 + (quadrant % 2) * 32 + lane] >> (4 * (quadrant / 2))) & 15;
                let higher = (b[128 + half * 32 + lane] >> (2 * quadrant)) & 3;
                let code = (lower | (higher << 4)) as i32 - 32;
                let scale = b[192 + half * 8 + quadrant * 2 + lane / 16] as i8;
                *value = (d * scale as f32) * code as f32;
            }
        }
        QuantType::IQ4_XS => {
            let d = little_half(b);
            let high = u16::from_le_bytes([b[2], b[3]]);
            for group in 0..8 {
                let lower = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
                let scale = (lower | (((high >> (group * 2)) as u8 & 3) << 4)) as i32 - 32;
                let d = d * scale as f32;
                for i in 0..32 {
                    let code = (b[8 + group * 16 + i % 16] >> (4 * (i / 16))) & 15;
                    out[group * 32 + i] = d * KVALUES_IQ4NL[code as usize] as f32;
                }
            }
        }
        QuantType::IQ2_S => {
            let d = little_half(b);
            for group in 0..8 {
                for octet in 0..4 {
                    let scale_code = (b[74 + group] >> (4 * (octet / 2))) & 15;
                    let scale = d * (0.5 + scale_code as f32) * 0.25;
                    let key = b[2 + group * 4 + octet] as usize
                        | (((b[66 + group] >> (2 * octet)) & 3) as usize) << 8;
                    let values = IQ2S_GRID[key].to_le_bytes();
                    let signs = b[34 + group * 4 + octet];
                    for i in 0..8 {
                        let sign = if signs & (1 << i) == 0 { 1.0 } else { -1.0 };
                        out[group * 32 + octet * 8 + i] = scale * values[i] as f32 * sign;
                    }
                }
            }
        }
        QuantType::IQ3_XXS => {
            let d = little_half(b);
            for group in 0..8 {
                let start = 66 + group * 4;
                let codes = u32::from_le_bytes(b[start..start + 4].try_into().unwrap());
                let scale = d * (0.5 + (codes >> 28) as f32) * 0.5;
                for octet in 0..4 {
                    let signs = KSIGNS_IQ2XS[((codes >> (7 * octet)) & 127) as usize];
                    for quartet in 0..2 {
                        let key = b[2 + group * 8 + octet * 2 + quartet] as usize;
                        let values = IQ3XXS_GRID[key].to_le_bytes();
                        for i in 0..4 {
                            let lane = quartet * 4 + i;
                            let sign = if signs & (1 << lane) == 0 { 1.0 } else { -1.0 };
                            out[group * 32 + octet * 8 + lane] = scale * values[i] as f32 * sign;
                        }
                    }
                }
            }
        }
        QuantType::IQ3_S => {
            let d = little_half(b);
            for group in 0..8 {
                let scale = d * (1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15)) as f32;
                for octet in 0..4 {
                    let signs = b[74 + group * 4 + octet];
                    for quartet in 0..2 {
                        let packed_index = octet * 2 + quartet;
                        let key = b[2 + group * 8 + packed_index] as usize
                            | (((b[66 + group] >> packed_index) & 1) as usize) << 8;
                        let values = IQ3S_GRID[key].to_le_bytes();
                        for i in 0..4 {
                            let lane = quartet * 4 + i;
                            let sign = if signs & (1 << lane) == 0 { 1.0 } else { -1.0 };
                            out[group * 32 + octet * 8 + lane] = scale * values[i] as f32 * sign;
                        }
                    }
                }
            }
        }
        _ => unreachable!("extended decoder called for a basic storage format"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q3_k_signed_subscales_and_all_high_mask_planes_decode_and_dot() {
        let scales = [
            -32i32, -31, -17, -16, -1, 0, 1, 15, 16, 17, 23, 24, 28, 29, 30, 31,
        ];
        let mut block = [0u8; 110];
        block[108..].copy_from_slice(&half::f16::from_f32(-0.125).to_bits().to_le_bytes());
        for (group, &scale) in scales.iter().enumerate() {
            let code = (scale + 32) as u8;
            block[96 + group % 8] |= (code & 15) << (4 * (group / 8));
            block[104 + group % 4] |= (code >> 4) << (2 * (group / 4));
        }
        let mut expected = [0.0f32; 256];
        for (i, out) in expected.iter_mut().enumerate() {
            let q = ((i * 5 + i / 7) % 8) as i32 - 4;
            let stored = (q & 3) as u8;
            block[32 + (i / 128) * 32 + i % 32] |= stored << (2 * ((i % 128) / 32));
            if q >= 0 {
                block[i % 32] |= 1 << (i / 32);
            }
            *out = -0.125 * scales[i / 16] as f32 * q as f32;
        }
        let mut decoded = [0.0; 256];
        decode_row(QuantType::Q3_K, &block, &mut decoded).unwrap();
        assert_eq!(decoded, expected);
        let input: Vec<f32> = (0..256).map(|i| (i % 19) as f32 - 9.0).collect();
        let expected: f32 = input.iter().zip(expected).map(|(x, w)| x * w).sum();
        assert_eq!(
            dot_scalar(QuantType::Q3_K, &block, &input).unwrap(),
            expected
        );
        assert!((dot(QuantType::Q3_K, &block, &input).unwrap() - expected).abs() < 1e-4);
        assert_eq!(QuantType::try_from(11).unwrap(), QuantType::Q3_K);
        assert_eq!(QuantType::Q3_K.row_bytes(512).unwrap(), 220);
        assert!(QuantType::Q3_K.row_bytes(640).is_err());
        assert!(decode_row(QuantType::Q3_K, &block[..109], &mut decoded).is_err());
    }

    #[test]
    fn q4_known_nibbles_keep_two_halves_and_negative_scale() {
        let mut bytes = f16::from_f32(-0.5).to_bits().to_le_bytes().to_vec();
        bytes.extend((0u8..16).map(|i| i | ((15 - i) << 4)));
        let mut row = [0.0f32; 32];
        decode_row(QuantType::Q4_0, &bytes, &mut row).unwrap();
        for i in 0..16 {
            assert_eq!(row[i], -0.5 * (i as i32 - 8) as f32);
            assert_eq!(row[i + 16], -0.5 * (7 - i as i32) as f32);
        }
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 10.0) * 0.125).collect();
        let expected: f32 = row.iter().zip(&input).map(|(w, x)| w * x).sum();
        assert_eq!(dot(QuantType::Q4_0, &bytes, &input).unwrap(), expected);
    }

    #[test]
    fn q8_signed_extremes_and_multiple_blocks() {
        let mut bytes = Vec::new();
        let values = [-128i8, -64, -1, 0, 1, 63, 64, 127];
        for scale in [0.25, -2.0] {
            bytes.extend(f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend((0..32).map(|i| values[i % values.len()] as u8));
        }
        let mut row = [0.0; 64];
        decode_row(QuantType::Q8_0, &bytes, &mut row).unwrap();
        for i in 0..64 {
            let scale = if i < 32 { 0.25 } else { -2.0 };
            assert_eq!(row[i], scale * values[i % values.len()] as f32);
        }
        let input: Vec<f32> = (0..64).map(|i| (i % 7) as f32 - 3.0).collect();
        let expected: f32 = row.iter().zip(&input).map(|(w, x)| w * x).sum();
        assert_eq!(dot(QuantType::Q8_0, &bytes, &input).unwrap(), expected);
    }

    #[test]
    fn ieee_encodings_and_direct_dots_match_reference() {
        let values = [1.0f32, -2.0, 0.0, 0.5];
        let input = [-2.0, 0.5, 12.0, 4.0];
        for ty in [QuantType::F32, QuantType::F16, QuantType::BF16] {
            let bytes: Vec<u8> = match ty {
                QuantType::F32 => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
                QuantType::F16 => values
                    .iter()
                    .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
                    .collect(),
                _ => values
                    .iter()
                    .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            };
            let mut row = [9.0; 4];
            decode_row(ty, &bytes, &mut row).unwrap();
            assert_eq!(row, values);
            assert_eq!(dot(ty, &bytes, &input).unwrap(), -1.0);
        }
    }

    #[test]
    fn malformed_rows_and_nonfinite_values_fail() {
        assert!(QuantType::Q4_0.row_bytes(31).is_err());
        assert!(QuantType::F32.row_bytes(usize::MAX).is_err());
        assert!(QuantType::try_from(43).is_err());
        assert!(decode_row(QuantType::Q8_0, &[0; 33], &mut [0.0; 32]).is_err());
        assert!(dot(QuantType::F32, &1.0f32.to_le_bytes(), &[f32::NAN]).is_err());
        assert!(decode_row(QuantType::F32, &f32::INFINITY.to_le_bytes(), &mut [0.0]).is_err());
        assert!(dot(QuantType::F32, &f32::INFINITY.to_le_bytes(), &[0.0]).is_err());
        let mut bytes = f16::NAN.to_bits().to_le_bytes().to_vec();
        bytes.resize(18, 0x88);
        assert!(dot(QuantType::Q4_0, &bytes, &[1.0; 32]).is_err());
    }

    #[test]
    fn runtime_dot_matches_scalar_across_blocks_unaligned_reads_and_float_tails() {
        for ty in [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4_0,
            QuantType::Q8_0,
        ] {
            for width in [128usize, 256, 1024] {
                let width = if ty.block_elements() == 1 {
                    width + 3
                } else {
                    width
                };
                let mut encoded = vec![0x55u8]; // Deliberately offset all reads by one byte.
                match ty {
                    QuantType::F32 => encoded.extend(
                        (0..width)
                            .flat_map(|i| (((i * 19 % 101) as f32 - 50.0) * 0.13).to_le_bytes()),
                    ),
                    QuantType::F16 => encoded.extend((0..width).flat_map(|i| {
                        f16::from_f32((i % 19) as f32 * 0.17 - 2.0)
                            .to_bits()
                            .to_le_bytes()
                    })),
                    QuantType::BF16 => encoded.extend((0..width).flat_map(|i| {
                        ((((i % 19) as f32 * 0.17 - 2.0).to_bits() >> 16) as u16).to_le_bytes()
                    })),
                    QuantType::Q4_0 | QuantType::Q8_0 => {
                        for block in 0..width / 32 {
                            let scale = f16::from_f32(if block % 2 == 0 { 0.13 } else { -0.71 });
                            encoded.extend(scale.to_bits().to_le_bytes());
                            let count = ty.block_bytes() - 2;
                            encoded.extend((0..count).map(|i| (i * 31 + block * 13) as u8));
                        }
                    }
                    _ => unreachable!("fixed five-format test"),
                }
                let input: Vec<f32> = (0..width).map(|i| (i % 23) as f32 * 0.093 - 0.83).collect();
                let expected = dot_scalar(ty, &encoded[1..], &input).unwrap();
                let actual = dot(ty, &encoded[1..], &input).unwrap();
                let tolerance = 0.002 + expected.abs() * 0.00003;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{ty:?} width {width}: {actual} != {expected}"
                );
                assert!(dot(ty, &encoded[2..], &input).is_err());
            }
        }
    }

    #[test]
    fn f16_vector_dispatch_preserves_special_values_bounds_and_odd_tails() {
        let width = 131;
        let patterns = [
            0x0000u16, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
        ];
        let mut encoded = vec![0x55];
        encoded.extend((0..width).flat_map(|i| patterns[i % patterns.len()].to_le_bytes()));
        let input: Vec<f32> = (0..width).map(|i| (i % 7) as f32 * 0.125 - 0.25).collect();
        let expected = dot_scalar(QuantType::F16, &encoded[1..], &input).unwrap();
        let actual = dot(QuantType::F16, &encoded[1..], &input).unwrap();
        assert!((actual - expected).abs() <= 0.002 + expected.abs() * 0.00003);
        assert!(dot(QuantType::F16, &encoded[2..], &input).is_err());
        assert_eq!(dot_kernel(QuantType::F16, 127), "scalar");
        #[cfg(target_arch = "x86_64")]
        {
            let vector = std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c");
            assert_eq!(
                dot_kernel(QuantType::F16, width),
                if vector { "avx2_fma_f16c" } else { "scalar" }
            );
        }
        // Poison both a full-vector position and the scalar tail. Zero
        // activations must not hide NaN or infinity stored in a model.
        for index in [0usize, 7, 8, 127, 130] {
            for poison in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
                let mut bad = encoded.clone();
                let start = 1 + index * 2;
                bad[start..start + 2].copy_from_slice(&poison.to_bits().to_le_bytes());
                assert!(dot(QuantType::F16, &bad[1..], &vec![0.0; width]).is_err());
                assert!(dot_scalar(QuantType::F16, &bad[1..], &vec![0.0; width]).is_err());
            }
        }
    }
}

#[cfg(test)]
mod avx512_k_tests {
    use super::*;

    fn encoded(ty: QuantType, width: usize, seed: usize, d: f16, min: f16) -> Vec<u8> {
        let mut bytes = vec![0x79]; // Exercise unaligned source rows.
        for block in 0..width / 256 {
            let start = bytes.len();
            bytes.extend(
                (0..ty.block_bytes()).map(|i| (seed * 79 + block * 131 + i * 37 + i / 3) as u8),
            );
            if ty == QuantType::Q4_K {
                bytes[start..start + 2].copy_from_slice(&d.to_le_bytes());
                bytes[start + 2..start + 4].copy_from_slice(&min.to_le_bytes());
            } else {
                bytes[start + 208..start + 210].copy_from_slice(&d.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn k_quant_fallback_preserves_residual_after_large_terms_cancel() {
        for ty in [QuantType::Q4_K, QuantType::Q6_K] {
            // Repeat one nonzero weight across three blocks. Each lane sees
            // +2^32, +1 and -2^32, so F32 accumulation loses the residual.
            let seed = encoded(ty, 256, 3, f16::from_f32(1.0), f16::from_f32(0.0));
            let bytes = seed[1..].repeat(3);
            let mut weights = [0.0; 256];
            decode_extended_block(ty, &seed[1..], &mut weights);
            let lane = weights.iter().position(|w| *w != 0.0).unwrap();
            let mut input = vec![0.0; 768];
            input[lane] = 4294967296.0;
            input[lane + 256] = 1.0;
            input[lane + 512] = -input[lane];
            assert_eq!(dot_k_wide(ty, &bytes, &input), weights[lane]);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { dot_k_avx2_wide(ty, &bytes, &input) }, weights[lane]);
            }
        }
    }

    #[test]
    fn q4_q6_full_rows_match_scalar_with_signed_extreme_scales_and_unaligned_loads() {
        for ty in [QuantType::Q4_K, QuantType::Q6_K] {
            for width in [256, 512, 2048, 5120] {
                for seed in 0..12 {
                    for (d, min) in [
                        (0.0, 0.0),
                        (-0.0, -0.0),
                        (0.00390625, 0.015625),
                        (-0.125, -0.03125),
                        (65504.0, -65504.0),
                        (f32::from_bits(0x33800000), 0.0),
                    ] {
                        let bytes = encoded(ty, width, seed, f16::from_f32(d), f16::from_f32(min));
                        let input: Vec<f32> = (0..width)
                            .map(|i| ((i * 73 + seed * 23) % 257) as f32 / 127.0 - 1.0)
                            .collect();
                        let scalar = dot_scalar(ty, &bytes[1..], &input).unwrap();
                        let actual = dot(ty, &bytes[1..], &input).unwrap();
                        let previous = dot_avx2_baseline(ty, &bytes[1..], &input).unwrap();
                        let mut decoded = vec![0.0; width];
                        decode_row(ty, &bytes[1..], &mut decoded).unwrap();
                        let absolute_sum: f64 = decoded
                            .iter()
                            .zip(&input)
                            .map(|(w, x)| (f64::from(*w) * f64::from(*x)).abs())
                            .sum();
                        // Bound error against the absolute product sum, including
                        // cancellations, rather than hiding them behind a ratio.
                        let bound = (absolute_sum * 3e-6).max(2e-6);
                        assert!(
                            f64::from((actual - scalar).abs()) <= bound,
                            "{ty:?}/{width}/{seed}: {actual} vs {scalar}, bound {bound}"
                        );
                        assert!(
                            f64::from((previous - scalar).abs()) <= bound,
                            "previous {ty:?}/{width}/{seed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q4_k_all_group_scale_bits_and_nibbles_have_independent_expected_values() {
        let mut block = vec![255u8; 144];
        block[..2].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
        block[16..].fill(0xf0);
        // Every packed group reconstructs scale=63,min=63. Alternating
        // 32-value groups carry code0/code15 independently of SIMD unpacking.
        for group in 0..8 {
            let mut input = vec![0.0; 256];
            input[group * 32..(group + 1) * 32].fill(1.0);
            let code = if group % 2 == 0 { 0.0 } else { 15.0 };
            let expected = (0.125 * 63.0 * code - 0.25 * 63.0) * 32.0;
            assert_eq!(dot(QuantType::Q4_K, &block, &input).unwrap(), expected);
        }
    }

    #[test]
    fn q6_k_signed_scale_extremes_and_high_planes_have_independent_expected_values() {
        let mut block = vec![0u8; 210];
        block[208..].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
        block[..128].fill(0xf0);
        block[128..192].fill(0xe4); // high planes0,1,2,3 in quadrant order.
        for (i, s) in block[192..208].iter_mut().enumerate() {
            *s = if i % 2 == 0 { 128 } else { 127 };
        }
        for group in 0..16 {
            let quadrant = (group % 8) / 2;
            let low = if quadrant < 2 { 0 } else { 15 };
            let signed_code = low + (quadrant as i32) * 16 - 32;
            let scale = if group % 2 == 0 { -128 } else { 127 };
            let mut input = vec![0.0; 256];
            input[group * 16..group * 16 + 16].fill(1.0);
            let expected = 0.25 * (scale as f32) * (signed_code as f32) * 16.0;
            assert_eq!(
                dot(QuantType::Q6_K, &block, &input).unwrap(),
                expected,
                "group {group}"
            );
        }
    }

    #[test]
    fn k_quant_dispatch_rejects_malformed_rows_and_nonfinite_metadata_or_input() {
        for ty in [QuantType::Q4_K, QuantType::Q6_K] {
            let bytes = encoded(ty, 512, 7, f16::from_f32(0.125), f16::from_f32(0.25));
            let input = vec![0.0; 512];
            assert!(dot(ty, &bytes[2..], &input).is_err());
            assert!(dot(ty, &bytes[1..bytes.len() - 1], &input).is_err());
            for invalid_width in [0, 1, 128, 255, 257, 511, 513] {
                assert!(dot(ty, &bytes[1..], &vec![0.0; invalid_width]).is_err());
            }
            for poison in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
                let offsets = if ty == QuantType::Q4_K {
                    vec![0, 2, 144, 146]
                } else {
                    vec![208, 418]
                };
                for offset in offsets {
                    let mut invalid = bytes.clone();
                    invalid[1 + offset..3 + offset].copy_from_slice(&poison.to_le_bytes());
                    assert!(dot(ty, &invalid[1..], &input).is_err());
                    assert!(dot_scalar(ty, &invalid[1..], &input).is_err());
                }
            }
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = input.clone();
                invalid[511] = poison;
                assert!(dot(ty, &bytes[1..], &invalid).is_err());
            }
            let overflow = vec![f32::MAX; 512];
            assert!(dot(ty, &bytes[1..], &overflow).is_err());
            assert!(dot_scalar(ty, &bytes[1..], &overflow).is_err());
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("fma")
            {
                assert_eq!(dot_kernel(ty, 512), "avx512f_bw_dq_fma_k_quant");
            }
        }
    }
}
