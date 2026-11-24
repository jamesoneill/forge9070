//! Independent CPU oracle for the runtime Q4_64 cache (not a GGUF format).
use half::f16;

pub(super) fn row_words(width: usize) -> usize {
    width.div_ceil(128) * 17
}

pub(super) fn pack_row(values: &[f32]) -> Result<Vec<u32>, &'static str> {
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite KV input");
    }
    let mut output = vec![0u32; row_words(values.len())];
    for (block, source) in values.chunks(128).enumerate() {
        let mut scales = [0.0f32; 2];
        for group in 0..2 {
            let maximum = source
                .iter()
                .skip(group * 64)
                .take(64)
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            let scale = if maximum == 0.0 {
                0.0
            } else {
                (maximum / 7.0).max(f16::from_bits(1).to_f32())
            };
            if scale > 65504.0 {
                return Err("KV scale exceeds F16");
            }
            let bits = f16::from_f32(scale).to_bits();
            scales[group] = f16::from_bits(bits).to_f32();
            output[block * 17] |= u32::from(bits) << (group * 16);
        }
        for (i, &value) in source.iter().enumerate() {
            let scale = scales[i / 64];
            let quantized = if scale == 0.0 {
                0
            } else {
                (value / scale).round_ties_even().clamp(-7.0, 7.0) as i32
            };
            output[block * 17 + 1 + i / 8] |= (quantized as u32 & 15) << ((i % 8) * 4);
        }
    }
    Ok(output)
}

pub(super) fn unpack(words: &[u32], width: usize, token: usize, coordinate: usize) -> f32 {
    let block = token * row_words(width) + (coordinate / 128) * 17;
    let within = coordinate % 128;
    let scale = f16::from_bits((words[block] >> ((within / 64) * 16)) as u16).to_f32();
    let code = ((words[block + 1 + within / 8] >> ((within % 8) * 4)) & 15) as i32;
    let signed = if code >= 8 { code - 16 } else { code };
    signed as f32 * scale
}

pub(super) fn attention_reference(
    query: &[f32],
    keys: &[u32],
    values: &[u32],
    history: usize,
    heads: usize,
    kv_heads: usize,
    dimension: usize,
) -> Vec<f32> {
    let width = kv_heads * dimension;
    let mut output = vec![0.0; heads * dimension];
    for head in 0..heads {
        let kv_head = head / (heads / kv_heads);
        let scores: Vec<f64> = (0..history)
            .map(|token| {
                (0..dimension)
                    .map(|i| {
                        query[head * dimension + i] as f64
                            * unpack(keys, width, token, kv_head * dimension + i) as f64
                    })
                    .sum::<f64>()
                    / (dimension as f64).sqrt()
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probabilities: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f64 = probabilities.iter().sum();
        for i in 0..dimension {
            output[head * dimension + i] = (probabilities
                .iter()
                .enumerate()
                .map(|(token, p)| p * unpack(values, width, token, kv_head * dimension + i) as f64)
                .sum::<f64>()
                / denominator) as f32;
        }
    }
    output
}

#[test]
fn q4_kv_reference_layout_rounding_padding_and_bound() {
    let mut values = vec![0.0; 64];
    values.extend([7.0, -7.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]);
    values.resize(169, 0.0);
    for (i, value) in values[128..].iter_mut().enumerate() {
        *value = (i as f32 - 19.0) / 17.0;
    }
    let packed = pack_row(&values).unwrap();
    assert_eq!(packed.len(), 34);
    assert_eq!(packed[0] as u16, 0);
    assert_eq!(packed[0] >> 16, f16::ONE.to_bits() as u32);
    let expected = [7.0, -7.0, 0.0, 2.0, 2.0, 0.0, -2.0, -2.0];
    for (i, expected) in expected.into_iter().enumerate() {
        assert_eq!(unpack(&packed, 169, 0, 64 + i), expected);
    }
    for (i, &value) in values.iter().enumerate() {
        let scale =
            f16::from_bits((packed[(i / 128) * 17] >> (((i % 128) / 64) * 16)) as u16).to_f32();
        assert!((value - unpack(&packed, 169, 0, i)).abs() <= scale * 0.5 + 1e-6);
    }
    for i in values.len()..256 {
        assert_eq!(unpack(&packed, 169, 0, i), 0.0);
    }
    assert_eq!(pack_row(&[0.0; 128]).unwrap(), vec![0; 17]);
    assert!(pack_row(&[f32::NAN]).is_err());
    assert!(pack_row(&[f32::INFINITY]).is_err());
    assert!(pack_row(&[1e20]).is_err());
    let tiny = pack_row(&[1e-9, -1e-9]).unwrap();
    assert!(unpack(&tiny, 2, 0, 0).is_finite());
}

#[test]
fn q4_kv_reference_full_context_addresses_and_oldest_history() {
    assert_eq!(row_words(512), 68);
    assert_eq!(row_words(512) * 524288 * 4 * 2 * 48, 13_690_208_256);
    assert_eq!(row_words(512) * 524287 + 3 * 17 + 16, 35_651_583);
    let query = vec![0.0; 8];
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for token in 0..3 {
        keys.extend(pack_row(&[0.0; 8]).unwrap());
        values.extend(pack_row(&[if token == 0 { 7.0 } else { 0.0 }; 8]).unwrap());
    }
    let actual = attention_reference(&query, &keys, &values, 3, 1, 1, 8);
    for value in actual {
        assert!((value - 7.0 / 3.0).abs() < 1e-6);
    }
}
