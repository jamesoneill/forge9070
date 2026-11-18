//! Cross-implementation numerical checks. RustingStrata is an oracle in these
//! integration tests only; the independent forge-engine crate never calls it.
//! Fixtures are synthetic and do not establish real-model/GPU performance.

use forge_engine::{
    gguf::{Gguf, Value},
    model::{Model, Sequence, Step},
    tokenizer::Tokenizer,
};
use rusting_strata::{
    gguf::{Gguf as ReferenceGguf, Value as ReferenceValue},
    model::{LayerState, Model as ReferenceModel, State as ReferenceState},
    ops,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct Checkpoint {
    directory: PathBuf,
    path: PathBuf,
}
impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// Give the existing deterministic fixture a nonconstant, signed output
/// projection. This prevents an implementation that only returns token A from
/// passing the numerical gate. Its tokenizer, attention and expert weights
/// remain the committed reproducible values.
#[derive(Clone, Copy, Debug)]
enum Storage {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
}

impl Storage {
    fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 30,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
        }
    }
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

fn metadata_type(value: &ReferenceValue) -> u32 {
    match value {
        ReferenceValue::U8(_) => 0,
        ReferenceValue::I8(_) => 1,
        ReferenceValue::U16(_) => 2,
        ReferenceValue::I16(_) => 3,
        ReferenceValue::U32(_) => 4,
        ReferenceValue::I32(_) => 5,
        ReferenceValue::F32(_) => 6,
        ReferenceValue::Bool(_) => 7,
        ReferenceValue::Str(_) => 8,
        ReferenceValue::Arr(_) => 9,
        ReferenceValue::U64(_) => 10,
        ReferenceValue::I64(_) => 11,
        ReferenceValue::F64(_) => 12,
    }
}

fn metadata_payload(out: &mut Vec<u8>, value: &ReferenceValue) {
    match value {
        ReferenceValue::U8(v) => out.push(*v),
        ReferenceValue::I8(v) => out.push(*v as u8),
        ReferenceValue::U16(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::I16(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::U32(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::I32(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::U64(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::I64(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::F32(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::F64(v) => out.extend(v.to_le_bytes()),
        ReferenceValue::Bool(v) => out.push(u8::from(*v)),
        ReferenceValue::Str(v) => string(out, v),
        ReferenceValue::Arr(values) => {
            // The only empty array in these fixtures is the string merge list.
            let kind = values.first().map(metadata_type).unwrap_or(8);
            out.extend(kind.to_le_bytes());
            out.extend((values.len() as u64).to_le_bytes());
            for value in values {
                assert_eq!(metadata_type(value), kind);
                metadata_payload(out, value);
            }
        }
    }
}

fn encode_weights(values: &[f32], storage: Storage) -> Vec<u8> {
    match storage {
        Storage::F32 => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Storage::F16 => values
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect(),
        Storage::BF16 => values
            .iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
            .collect(),
        Storage::Q4_0 | Storage::Q8_0 => {
            assert_eq!(values.len() % 32, 0);
            let mut encoded = Vec::new();
            for block in values.chunks_exact(32) {
                let maximum = block.iter().fold(0.0f32, |a, b| a.max(b.abs()));
                let range = if matches!(storage, Storage::Q4_0) {
                    7.0
                } else {
                    127.0
                };
                let scale = half::f16::from_f32(maximum / range);
                encoded.extend(scale.to_le_bytes());
                let scale = scale.to_f32();
                let quantize = |x: f32| {
                    if scale == 0.0 {
                        0
                    } else {
                        (x / scale).round() as i32
                    }
                };
                if matches!(storage, Storage::Q4_0) {
                    for i in 0..16 {
                        let low = (quantize(block[i]).clamp(-8, 7) + 8) as u8;
                        let high = (quantize(block[i + 16]).clamp(-8, 7) + 8) as u8;
                        encoded.push(low | (high << 4));
                    }
                } else {
                    encoded.extend(
                        block
                            .iter()
                            .map(|x| quantize(*x).clamp(-127, 127) as i8 as u8),
                    );
                }
            }
            encoded
        }
    }
}

fn varied_checkpoint(moe: bool) -> Checkpoint {
    checkpoint_variant(moe, Storage::F32, false)
}

fn checkpoint_variant(moe: bool, storage: Storage, grouped_query: bool) -> Checkpoint {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(if moe {
        "tests/fixtures/tiny-qwen3moe.gguf"
    } else {
        "tests/fixtures/tiny-qwen3.gguf"
    });
    let fixture = ReferenceGguf::open(&source).unwrap();
    assert!(fixture.tensors.values().all(|tensor| tensor.ty == 0));
    let mut meta = fixture.meta.clone();
    let arch = if moe { "qwen3moe" } else { "qwen3" };
    if grouped_query {
        for (key, value) in [
            ("attention.head_count", 4),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 8),
            ("rope.dimension_count", 8),
        ] {
            meta.insert(format!("{arch}.{key}"), ReferenceValue::U32(value));
        }
    }
    let mut data = Vec::new();
    let mut descriptions = Vec::new();
    for tensor in fixture.tensors.values() {
        let mut dims = tensor.dims.clone();
        let mut values: Vec<f32> = fixture
            .bytes(tensor)
            .chunks_exact(4)
            .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
            .collect();
        if tensor.name == "output.weight" {
            let columns = dims[0] as usize;
            for (index, value) in values.iter_mut().enumerate() {
                let (token, column) = (index / columns, index % columns);
                *value = (((token * 97 + column * 31 + token * column * 7) % 211) as f32 - 105.0)
                    / 256.0;
            }
        }
        if grouped_query
            && matches!(
                tensor.name.as_str(),
                "blk.0.attn_k.weight" | "blk.0.attn_v.weight"
            )
        {
            dims[1] = 16;
            values.truncate(32 * 16);
        }
        if grouped_query
            && matches!(
                tensor.name.as_str(),
                "blk.0.attn_q_norm.weight" | "blk.0.attn_k_norm.weight"
            )
        {
            dims[0] = 8;
            values.truncate(8);
        }
        let ty = if dims.len() == 1 {
            Storage::F32
        } else {
            storage
        };
        let encoded = encode_weights(&values, ty);
        string(&mut descriptions, &tensor.name);
        descriptions.extend((dims.len() as u32).to_le_bytes());
        for dimension in &dims {
            descriptions.extend(dimension.to_le_bytes());
        }
        descriptions.extend(ty.id().to_le_bytes());
        descriptions.extend((data.len() as u64).to_le_bytes());
        data.extend(encoded);
        data.resize(data.len().next_multiple_of(32), 0);
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((fixture.tensors.len() as u64).to_le_bytes());
    bytes.extend((meta.len() as u64).to_le_bytes());
    for (key, value) in &meta {
        string(&mut bytes, key);
        bytes.extend(metadata_type(value).to_le_bytes());
        metadata_payload(&mut bytes, value);
    }
    bytes.extend(descriptions);
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend(data);
    let unique = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "forge-engine-parity-{}-{stamp}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("varied.gguf");
    std::fs::write(&path, bytes).unwrap();
    Checkpoint { directory, path }
}

fn close(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{label} vector lengths");
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && expected.is_finite(),
            "{label}[{index}] contains a non-finite value"
        );
        let bound = 2.0e-5 + 2.0e-5 * expected.abs();
        assert!(
            (got - expected).abs() <= bound,
            "{label}[{index}] got {got}, expected {expected}, difference {}, bound {bound}",
            (got - expected).abs()
        );
    }
}

struct OracleStep {
    logits: Vec<f32>,
    normalized_hidden: Vec<f32>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
}

fn oracle_trace(path: &Path, tokens: &[u32]) -> Vec<OracleStep> {
    let gguf = ReferenceGguf::open(path).unwrap();
    let model = ReferenceModel::load(&gguf).unwrap();
    let norm = model.w.get("output_norm.weight").unwrap().to_f32();
    let mut sequence = ReferenceState::new(&model.cfg);
    tokens
        .iter()
        .map(|&token| {
            let (logits, residual) = model.step_h(&mut sequence, token);
            let mut normalized_hidden = vec![0.0; residual.len()];
            // The reference returns pre-normalization residuals; the independent
            // Step.hidden contract is the normalized input to the output matrix.
            ops::rms_norm(&residual, &norm, model.cfg.eps, &mut normalized_hidden);
            let kv = sequence
                .layers
                .iter()
                .map(|layer| match layer {
                    LayerState::Attn { k, v, .. } => (k.clone(), v.clone()),
                    _ => panic!("these fixtures exercise ordinary attention"),
                })
                .collect();
            OracleStep {
                logits,
                normalized_hidden,
                kv,
            }
        })
        .collect()
}

fn compare(step: &Step, sequence: &Sequence, expected: &OracleStep, position: usize) {
    close("all vocabulary logits", &step.logits, &expected.logits);
    close(
        "normalized hidden",
        &step.hidden,
        &expected.normalized_hidden,
    );
    assert_eq!(sequence.position(), position);
    assert_eq!(ops::argmax(&step.logits), ops::argmax(&expected.logits));
    for (layer, (expected_k, expected_v)) in expected.kv.iter().enumerate() {
        let (k, v) = sequence.kv(layer).expect("attention layer state");
        close("key cache", k, expected_k);
        close("value cache", v, expected_v);
    }
}

fn parity(moe: bool, storage: Storage, grouped_query: bool) {
    let checkpoint = checkpoint_variant(moe, storage, grouped_query);
    let gguf = Arc::new(Gguf::open(&checkpoint.path).unwrap());
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let original = ReferenceGguf::open(&checkpoint.path).unwrap();
    let reference_tokenizer = rusting_strata::tokenizer::Tokenizer::from_gguf(&original).unwrap();
    let text = "Rust £\nA\0😀";
    let mut tokens = tokenizer.encode(text).unwrap();
    assert_eq!(tokens, reference_tokenizer.encode(text).unwrap());
    assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
    tokens.extend([3, 65, 191, 0, 258]); // teacher force controls/bytes too
    let expected = oracle_trace(&checkpoint.path, &tokens);
    assert!(
        expected[0]
            .logits
            .iter()
            .filter(|x| x.abs() > 1.0e-4)
            .count()
            > 128,
        "the parity fixture must expose a nonconstant projection, not only token A"
    );
    assert!(
        expected
            .windows(2)
            .any(|pair| pair[0].normalized_hidden != pair[1].normalized_hidden)
    );
    let model = Model::load(gguf, 64).unwrap();
    let mut sequence = model.new_sequence();
    for (index, (&token, expected)) in tokens.iter().zip(&expected).enumerate() {
        compare(
            &model.step(&mut sequence, token).unwrap(),
            &sequence,
            expected,
            index + 1,
        );
    }
}

#[test]
fn dense_teacher_forced_full_logits_hidden_and_kv_match_reference() {
    parity(false, Storage::F32, false);
}

#[test]
fn moe_teacher_forced_full_logits_hidden_and_kv_match_reference() {
    parity(true, Storage::F32, false);
}

#[test]
fn dense_f16_bf16_q4_and_q8_full_model_numerical_parity() {
    for storage in [Storage::F16, Storage::BF16, Storage::Q4_0, Storage::Q8_0] {
        parity(false, storage, false);
    }
}

#[test]
fn moe_f16_bf16_q4_and_q8_full_model_numerical_parity() {
    for storage in [Storage::F16, Storage::BF16, Storage::Q4_0, Storage::Q8_0] {
        parity(true, storage, false);
    }
}

#[test]
fn grouped_query_multihead_dense_and_moe_match_reference() {
    for moe in [false, true] {
        parity(moe, Storage::F32, true);
    }
}

#[test]
fn interleaved_sequences_and_reset_match_isolated_reference_runs() {
    let checkpoint = varied_checkpoint(true);
    let model = Model::load(Arc::new(Gguf::open(&checkpoint.path).unwrap()), 32).unwrap();
    let left = b"left\0context";
    let right = b"different right";
    let left_ids: Vec<_> = left.iter().map(|v| u32::from(*v)).collect();
    let right_ids: Vec<_> = right.iter().map(|v| u32::from(*v)).collect();
    let a = oracle_trace(&checkpoint.path, &left_ids);
    let b = oracle_trace(&checkpoint.path, &right_ids);
    let (mut left_state, mut right_state) = (model.new_sequence(), model.new_sequence());
    for i in 0..left.len().max(right.len()) {
        if let Some(&token) = left_ids.get(i) {
            let step = model.step(&mut left_state, token).unwrap();
            compare(&step, &left_state, &a[i], i + 1);
        }
        if let Some(&token) = right_ids.get(i) {
            let step = model.step(&mut right_state, token).unwrap();
            compare(&step, &right_state, &b[i], i + 1);
        }
    }
    left_state.reset();
    assert_eq!(left_state.position(), 0);
    let (k, v) = left_state.kv(0).unwrap();
    assert!(k.is_empty() && v.is_empty());
    compare(
        &model.step(&mut left_state, left_ids[0]).unwrap(),
        &left_state,
        &a[0],
        1,
    );
    assert_eq!(
        right_state.position(),
        right_ids.len(),
        "reset must not touch another sequence"
    );
}

#[test]
fn invalid_token_context_and_foreign_sequence_fail_without_state_changes() {
    let checkpoint = varied_checkpoint(false);
    let gguf = Arc::new(Gguf::open(&checkpoint.path).unwrap());
    let model = Model::load(gguf.clone(), 2).unwrap();
    let other = Model::load(gguf, 2).unwrap();
    let mut state = model.new_sequence();
    model.step(&mut state, 65).unwrap();
    let snapshot = {
        let (k, v) = state.kv(0).unwrap();
        (k.to_vec(), v.to_vec())
    };
    assert!(model.step(&mut state, model.vocab_size() as u32).is_err());
    assert!(other.step(&mut state, 66).is_err());
    assert_eq!(state.position(), 1);
    assert_eq!(
        state.kv(0).unwrap(),
        (snapshot.0.as_slice(), snapshot.1.as_slice())
    );
    model.step(&mut state, 66).unwrap();
    let snapshot = {
        let (k, v) = state.kv(0).unwrap();
        (k.to_vec(), v.to_vec())
    };
    assert!(model.step(&mut state, 67).is_err());
    assert_eq!(state.position(), 2);
    assert_eq!(
        state.kv(0).unwrap(),
        (snapshot.0.as_slice(), snapshot.1.as_slice())
    );
}

#[test]
fn incomplete_qwen38_metadata_is_rejected_without_fallback() {
    let checkpoint = varied_checkpoint(true);
    let mut gguf = Gguf::open(&checkpoint.path).unwrap();
    gguf.meta.insert(
        "general.architecture".into(),
        Value::String("qwen4exp".into()),
    );
    let error = Model::load(Arc::new(gguf), 32).unwrap_err().to_string();
    assert!(
        error.contains("qwen4exp.") && error.contains("required"),
        "{error}"
    );
}
