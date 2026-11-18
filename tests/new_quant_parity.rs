//! Independent decoder validation against the explicitly attributed reference
//! engine. The production forge-engine crate has no dependency on this oracle.
use forge_engine::quant::{self, QuantType};
use rusting_strata::quant::{GgmlType, dequant_row};

const FORMATS: [QuantType; 14] = [
    QuantType::F32,
    QuantType::F16,
    QuantType::BF16,
    QuantType::Q4_0,
    QuantType::Q8_0,
    QuantType::Q2_0,
    QuantType::Q4_K,
    QuantType::Q5_K,
    QuantType::Q6_K,
    QuantType::IQ2_S,
    QuantType::IQ3_XXS,
    QuantType::IQ3_S,
    QuantType::IQ4_NL,
    QuantType::IQ4_XS,
];

fn pattern(ty: QuantType, seed: u32) -> (Vec<u8>, usize) {
    let count = if ty.block_elements() == 1 {
        259
    } else {
        7 * ty.block_elements()
    };
    let mut encoded = vec![0u8; ty.row_bytes(count).unwrap()];
    let mut state = seed.wrapping_mul(112183).wrapping_add(849983);
    for byte in &mut encoded {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *byte = state as u8;
    }
    for (i, block) in encoded.chunks_exact_mut(ty.block_bytes()).enumerate() {
        let scale: f32 = [0.0, 0.125, -0.4375, 2.5, -4.0, 1.0 / 1024.0][(i + seed as usize) % 6];
        match ty {
            QuantType::F32 => block.copy_from_slice(&scale.to_le_bytes()),
            QuantType::F16 => {
                block.copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes())
            }
            QuantType::BF16 => {
                block.copy_from_slice(&half::bf16::from_f32(scale).to_bits().to_le_bytes())
            }
            _ => {
                let offset = if ty == QuantType::Q6_K { 208 } else { 0 };
                block[offset..offset + 2]
                    .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                if matches!(ty, QuantType::Q4_K | QuantType::Q5_K) {
                    block[2..4].copy_from_slice(
                        &half::f16::from_f32(0.03125 * (seed % 7) as f32)
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
            }
        }
    }
    (encoded, count)
}

#[test]
fn every_supported_format_matches_reference_values_across_mixed_bit_patterns() {
    for ty in FORMATS {
        let reference_type = GgmlType::from_u32(ty.id()).unwrap();
        assert_eq!(ty.block_elements(), reference_type.block_elems());
        assert_eq!(ty.block_bytes(), reference_type.block_bytes());
        for seed in 1..=32 {
            let (data, count) = pattern(ty, seed);
            let mut reference = vec![0.; count];
            let mut independent = vec![0.; count];
            dequant_row(reference_type, &data, &mut reference);
            quant::decode_row(ty, &data, &mut independent).unwrap();
            for (index, (&actual, &expected)) in independent.iter().zip(&reference).enumerate() {
                assert_eq!(actual, expected, "{ty:?}, seed {seed}, value {index}");
            }
        }
    }
}

#[test]
fn quantized_direct_and_vector_dots_match_reference_without_row_allocations() {
    for ty in FORMATS {
        for seed in 1..=8 {
            let (data, count) = pattern(ty, seed);
            let mut reference = vec![0.; count];
            dequant_row(GgmlType::from_u32(ty.id()).unwrap(), &data, &mut reference);
            let input: Vec<f32> = (0..count)
                .map(|i| ((i * 19 % 131) as f32 - 65.) / 67.)
                .collect();
            let expected: f32 = reference.iter().zip(&input).map(|(w, x)| w * x).sum();
            let scalar = quant::dot_scalar(ty, &data, &input).unwrap();
            assert_eq!(scalar, expected, "{ty:?} scalar seed{seed}");
            let selected = quant::dot(ty, &data, &input).unwrap();
            // Reduction order changes under SIMD; compare absolute+relative
            // error rather than inferring correctness from identical token IDs.
            let tolerance = 0.008 + expected.abs() * 0.00008;
            assert!(
                (selected - expected).abs() <= tolerance,
                "{ty:?} vector seed{seed}: {selected} vs {expected}, tolerance{tolerance}"
            );
            let mut unaligned = vec![0xff];
            unaligned.extend(&data);
            assert_eq!(quant::dot(ty, &unaligned[1..], &input).unwrap(), selected);
            assert!(quant::dot(ty, &data[1..], &input).is_err());
        }
    }
}

#[test]
fn extended_storage_rejects_invalid_geometry_and_nonfinite_scales() {
    for ty in FORMATS.into_iter().filter(|t| t.block_elements() > 1) {
        assert!(ty.row_bytes(ty.block_elements() - 1).is_err());
        let (mut bytes, count) = pattern(ty, 1);
        let scale_offset = if ty == QuantType::Q6_K { 208 } else { 0 };
        bytes[scale_offset..scale_offset + 2]
            .copy_from_slice(&half::f16::INFINITY.to_bits().to_le_bytes());
        assert!(quant::decode_row(ty, &bytes, &mut vec![0.; count]).is_err());
        assert!(quant::dot_scalar(ty, &bytes, &vec![0.; count]).is_err());
        assert!(quant::dot(ty, &bytes, &vec![0.; count]).is_err());
    }
}
