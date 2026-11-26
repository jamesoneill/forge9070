//! Experimental independent implementation of the Prism folded-weight basis.
//! The normal build rejects this format. This module is deliberately limited to
//! explicit-sign, text Qwen35 GGUF metadata and the existing standard Q2_0 type.
//! Reference: https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF
//! Format details: https://github.com/ggml-org/llama.cpp/issues/29058
use crate::gguf::{Gguf, Value};
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

const PREFIX: &str = "prism.hadamard.";
const KNOWN: &[&str] = &[
    "version", "block_size", "transform", "axis", "sign_mode",
    "sign_widths", "sign_values", "weight_names", "inverse_weight_names",
    "gdn_v_grouped", "tied_output",
];

pub(crate) fn declared(meta: &BTreeMap<String, Value>) -> bool {
    meta.keys().any(|k| k == "prism.hadamard" || k.starts_with(PREFIX))
}

/// Apply the orthonormal Sylvester transform independently to each block.
fn hadamard(values: &mut [f32], block: usize) {
    let scale = (block as f32).sqrt().recip();
    for chunk in values.chunks_exact_mut(block) {
        let mut stride = 1;
        while stride < block {
            for pair in chunk.chunks_exact_mut(stride * 2) {
                let (left, right) = pair.split_at_mut(stride);
                for (a, b) in left.iter_mut().zip(right) {
                    let (x, y) = (*a, *b);
                    *a = x + y;
                    *b = x - y;
                }
            }
            stride *= 2;
        }
        for value in chunk {
            *value *= scale;
        }
    }
}

/// GGUF Qwen35's ordinary recurrent output is tiled [rep, key, head].
/// A grouped folded ssm_out expects [key, rep, head] before sign/rotation.
fn grouped(input: &[f32], out: &mut [f32], keys: usize, heads: usize) -> Result<()> {
    ensure!(keys > 0 && heads > 0 && heads % keys == 0 &&
        input.len() == out.len() && input.len() % heads == 0,
        "invalid Prism grouped DeltaNet dimensions");
    let width = input.len() / heads;
    let repeats = heads / keys;
    for k in 0..keys {
        for r in 0..repeats {
            let src = (r * keys + k) * width;
            let dst = (k * repeats + r) * width;
            out[dst..dst+width].copy_from_slice(&input[src..src+width]);
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct Basis {
    block: usize,
    signs: BTreeMap<usize, Vec<f32>>,
    forward: BTreeSet<String>,
    inverse: BTreeSet<String>,
    grouped_gdn: bool,
    tied: bool,
}

impl Basis {
    pub(crate) fn from_gguf(g: &Gguf) -> Result<Option<Self>> {
        if !declared(&g.meta) { return Ok(None); }
        ensure!(cfg!(feature = "experimental-prism"),
            "Prism Hadamard-folded weights require the experimental-prism build");
        ensure!(g.meta.get("general.architecture").and_then(Value::as_str) == Some("qwen35"),
            "experimental Prism basis is restricted to dense Qwen35");
        Self::parse(&g.meta, &g.tensors).map(Some)
    }

    fn parse(meta: &BTreeMap<String, Value>,
             tensors: &BTreeMap<String, crate::gguf::TensorInfo>) -> Result<Self> {
        for key in meta.keys().filter(|k| k.as_str() == "prism.hadamard" || k.starts_with(PREFIX)) {
            ensure!(key.strip_prefix(PREFIX).is_some_and(|k| KNOWN.contains(&k)),
                "unknown Prism basis metadata {key}");
        }
        let value = |suffix: &str| meta.get(&format!("{PREFIX}{suffix}"));
        let number = |suffix: &str| -> Result<usize> {
            Ok(usize::try_from(value(suffix).and_then(Value::as_u64)
                .with_context(|| format!("missing/invalid Prism {suffix}"))?)?)
        };
        let array = |suffix: &str| -> Result<&[Value]> {
            value(suffix).and_then(Value::as_array)
                .with_context(|| format!("missing/invalid Prism {suffix}"))
        };
        let boolean = |suffix: &str| -> Result<bool> {
            match value(suffix) {
                None => Ok(false),
                Some(v) => v.as_bool().context("Prism boolean must be a bool"),
            }
        };
        let version = number("version")?;
        let tied = boolean("tied_output")?;
        ensure!(matches!(version, 1 | 2) && tied == (version == 2),
            "unsupported Prism schema/tied output combination");
        let block = number("block_size")?;
        ensure!(block.is_power_of_two() && block <= 65536, "invalid Prism block size");
        for (key, expected) in [
            ("transform", "normalized-sylvester-walsh-hadamard"),
            ("axis", "input-last-dimension"),
            ("sign_mode", "explicit"),
        ] {
            ensure!(value(key).and_then(Value::as_str) == Some(expected),
                "unsupported Prism {key}");
        }
        let raw_signs = array("sign_values")?;
        let mut signs = BTreeMap::new();
        let mut cursor: usize = 0;
        for width in array("sign_widths")? {
            let width = usize::try_from(width.as_u64().context("invalid Prism sign width")?)?;
            ensure!(width > 0 && width % block == 0, "invalid Prism sign width");
            let end = cursor.checked_add(width).context("Prism sign count overflow")?;
            let slice = raw_signs.get(cursor..end).context("truncated Prism signs")?;
            let mut v = Vec::with_capacity(width);
            for sign in slice {
                // Reject floats and all non-unit signed integers.
                let s = match sign {
                    Value::I8(-1) | Value::I16(-1) | Value::I32(-1) | Value::I64(-1) => -1.0,
                    _ if sign.as_u64() == Some(1) => 1.0,
                    _ => anyhow::bail!("invalid Prism sign; expected integer ±1"),
                };
                v.push(s);
            }
            ensure!(signs.insert(width, v).is_none(), "duplicate Prism sign width");
            cursor = end;
        }
        ensure!(cursor == raw_signs.len() && cursor > 0, "unused or missing Prism signs");
        let names = |suffix: &str, optional: bool| -> Result<BTreeSet<String>> {
            if optional && value(suffix).is_none() { return Ok(BTreeSet::new()); }
            let mut names = BTreeSet::new();
            for name in array(suffix)? {
                let name = name.as_str().context("Prism weight name must be a string")?;
                let tensor = tensors.get(name).with_context(|| format!("missing Prism tensor {name}"))?;
                ensure!(tensor.dims.len() == 2, "Prism requires a matrix: {name}");
                let width = usize::try_from(tensor.dims[0])?;
                ensure!(signs.contains_key(&width), "missing Prism signs for {name}");
                ensure!(names.insert(name.to_owned()), "duplicate Prism tensor {name}");
            }
            Ok(names)
        };
        let forward = names("weight_names", false)?;
        let inverse = names("inverse_weight_names", true)?;
        ensure!(!forward.is_empty(), "empty Prism forward tensor set");
        for name in &forward {
            let layer_projection = name.strip_prefix("blk.").and_then(|v| v.split_once('.'))
                .is_some_and(|(layer, suffix)| layer.parse::<usize>().is_ok() && matches!(suffix,
                    "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_qkv.weight" |
                    "attn_gate.weight" | "attn_output.weight" | "ffn_gate.weight" |
                    "ffn_up.weight" | "ffn_down.weight" | "ssm_out.weight"));
            ensure!(name == "output.weight" || layer_projection,
                "unverified Prism forward tensor {name}");
        }
        ensure!(inverse.iter().all(|n| n == "token_embd.weight") &&
            forward.is_disjoint(&inverse), "unsupported Prism inverse tensor");
        if tied {
            ensure!(!tensors.contains_key("output.weight") && inverse.contains("token_embd.weight"),
                "invalid Prism tied output");
        } else if inverse.contains("token_embd.weight") {
            ensure!(tensors.contains_key("output.weight"),
                "latent embedding requires a separate output or explicit tied schema");
        }
        let grouped_gdn = boolean("gdn_v_grouped")?;
        Ok(Self { block, signs, forward, inverse, grouped_gdn, tied })
    }

    pub(crate) fn activation(&self, name: &str, input: &[f32],
                             keys: usize, heads: usize) -> Result<Option<Vec<f32>>> {
        if !self.forward.contains(name) && !(self.tied && name == "token_embd.weight") {
            return Ok(None);
        }
        let signs = self.signs.get(&input.len()).context("Prism activation width mismatch")?;
        let mut out = vec![0.0; input.len()];
        if self.grouped_gdn && name.ends_with(".ssm_out.weight") {
            grouped(input, &mut out, keys, heads)?;
        } else {
            out.copy_from_slice(input);
        }
        for (v, sign) in out.iter_mut().zip(signs) { *v *= sign; }
        hadamard(&mut out, self.block);
        crate::math::finite(&out, "Prism forward activation")?;
        Ok(Some(out))
    }

    pub(crate) fn embedding(&self, name: &str, row: &mut [f32]) -> Result<()> {
        if !self.inverse.contains(name) { return Ok(()); }
        let signs = self.signs.get(&row.len()).context("Prism embedding width mismatch")?;
        hadamard(row, self.block);
        for (v, sign) in row.iter_mut().zip(signs) { *v *= sign; }
        crate::math::finite(row, "Prism inverse embedding")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gguf::TensorInfo, quant::QuantType};
    fn text(s: &str) -> Value { Value::String(s.into()) }
    fn fixture() -> (BTreeMap<String, Value>, BTreeMap<String, TensorInfo>) {
        let mut meta = BTreeMap::new();
        let mut insert = |k: &str, v| { meta.insert(format!("{PREFIX}{k}"), v); };
        insert("version", Value::U32(1));
        insert("block_size", Value::U32(4));
        insert("transform", text("normalized-sylvester-walsh-hadamard"));
        insert("axis", text("input-last-dimension"));
        insert("sign_mode", text("explicit"));
        insert("sign_widths", Value::Array(vec![Value::U32(4)]));
        insert("sign_values", Value::Array(vec![Value::I32(1), Value::I32(-1), Value::I32(1), Value::I32(-1)]));
        insert("weight_names", Value::Array(vec![text("blk.0.ffn_gate.weight"), text("output.weight")]));
        insert("inverse_weight_names", Value::Array(vec![text("token_embd.weight")]));
        let tensors = ["blk.0.ffn_gate.weight", "output.weight", "token_embd.weight"]
            .into_iter().map(|n| (n.to_owned(), TensorInfo { name:n.into(), dims:vec![4,4],
                ty:QuantType::F32, offset:0, byte_len:64, file_offset:0, shard_index:0 })).collect();
        (meta, tensors)
    }

    #[test]
    fn forward_inverse_and_projection_agree_with_explicit_matrix() {
        let (meta, tensors) = fixture();
        let basis = Basis::parse(&meta, &tensors).unwrap();
        let x = [1.,2.,3.,4.];
        let forward = basis.activation("blk.0.ffn_gate.weight", &x, 1, 1).unwrap().unwrap();
        // H * S * x = [-1, 5, 0, -2] in natural Sylvester order.
        assert_eq!(forward, vec![-1.,5.,0.,-2.]);
        let mut row = forward.clone();
        basis.embedding("token_embd.weight", &mut row).unwrap();
        assert_eq!(row, x);
        let w = [2.,3.,5.,7.];
        let folded = basis.activation("output.weight", &w, 1, 1).unwrap().unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a,b)|a*b).sum::<f32>();
        assert_eq!(dot(&w,&x), dot(&folded,&forward));
        assert!(basis.activation("unfolded", &x, 1, 1).unwrap().is_none());
    }

    #[test]
    fn grouped_delta_permutation_and_invalid_geometry() {
        let input = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.];
        let mut out = [0.;12];
        grouped(&input, &mut out, 2, 6).unwrap();
        assert_eq!(out, [0.,1.,4.,5.,8.,9.,2.,3.,6.,7.,10.,11.]);
        assert!(grouped(&input,&mut out,4,6).is_err());
    }

    #[test]
    fn strict_schema_rejects_bad_signs_names_and_tied_output() {
        let (meta, tensors) = fixture();
        for (key, value) in [
            ("version", Value::U32(3)),
            ("block_size", Value::U32(3)),
            ("future_field", Value::Bool(true)),
            ("sign_values", Value::Array(vec![Value::I32(0);4])),
            ("weight_names", Value::Array(vec![text("missing.weight")])),
            ("tied_output", Value::Bool(true)),
        ] {
            let mut bad = meta.clone();
            bad.insert(format!("{PREFIX}{key}"), value);
            assert!(Basis::parse(&bad, &tensors).is_err(), "{key}");
        }
        let mut meta = meta;
        let mut tensors = tensors;
        meta.insert(format!("{PREFIX}version"), Value::U32(2));
        meta.insert(format!("{PREFIX}tied_output"), Value::Bool(true));
        meta.insert(format!("{PREFIX}weight_names"), Value::Array(vec![text("blk.0.ffn_gate.weight")]));
        tensors.remove("output.weight");
        let basis = Basis::parse(&meta,&tensors).unwrap();
        assert!(basis.activation("token_embd.weight", &[1.;4], 1, 1).unwrap().is_some());
    }
}
