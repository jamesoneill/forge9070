//! Qwen3.8 numerical integration against an independently invoked reference.
//! Original synthetic checkpoint: hybrid layers, HC, PLE, QSA and shared MoE.
//! No downloaded model weights are used; this is not a pretrained-quality test.
use forge_engine::{
    gguf::{Gguf, Tensor, Value},
    model::{MatvecExecutor, Model, Step},
};
use rusting_strata::{
    gguf::{Gguf as ReferenceGguf, Value as ReferenceValue},
    model::{LayerState, Model as ReferenceModel, State as ReferenceState},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
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

/// Storage variants for the original deterministic hybrid fixture below.
/// Every projection has nonconstant signed values; only the byte tokenizer is
/// reused from the committed tiny model. The fixture is untrained.
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

fn fixture(storage: Storage, combined: bool, indexer: bool, ple: bool) -> Checkpoint {
    let base = ReferenceGguf::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3moe.gguf"),
    )
    .unwrap();
    let mut meta: BTreeMap<String, ReferenceValue> = base
        .meta
        .iter()
        .filter(|(k, _)| k.starts_with("tokenizer."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    meta.insert(
        "general.architecture".into(),
        ReferenceValue::Str("qwen4exp".into()),
    );
    meta.insert("general.type".into(), ReferenceValue::Str("model".into()));
    for (name, value) in [
        ("block_count", 4),
        ("context_length", 64),
        ("embedding_length", 32),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 8),
        ("attention.value_length", 8),
        ("rope.dimension_count", 4),
        ("expert_count", 4),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 32),
        ("expert_shared_feed_forward_length", 32),
        ("hyper_connection.count", 2),
        ("hyper_connection.low_rank", 4),
        ("full_attention_interval", 2),
        ("ssm.group_count", 2),
        ("ssm.time_step_rank", 4),
        ("ssm.state_size", 4),
        ("ssm.inner_size", 16),
        ("ssm.conv_kernel", 4),
    ] {
        meta.insert(format!("qwen4exp.{name}"), ReferenceValue::U32(value));
    }
    meta.insert(
        "qwen4exp.rope.freq_base".into(),
        ReferenceValue::F32(10000.),
    );
    meta.insert(
        "qwen4exp.attention.layer_norm_rms_epsilon".into(),
        ReferenceValue::F32(1e-6),
    );
    meta.insert(
        "qwen4exp.rope.dimension_sections".into(),
        ReferenceValue::Arr([1, 1, 0, 0].into_iter().map(ReferenceValue::U32).collect()),
    );
    if indexer {
        for (name, value) in [
            ("attention.indexer.head_count", 2),
            ("attention.indexer.key_length", 4),
            ("attention.indexer.top_k", 4),
        ] {
            meta.insert(format!("qwen4exp.{name}"), ReferenceValue::U32(value));
        }
        meta.insert(
            "qwen4exp.attention.compress_ratios".into(),
            ReferenceValue::Arr([0, 4, 0, 4].into_iter().map(ReferenceValue::U32).collect()),
        );
    }
    if ple {
        for (name, value) in [
            ("embedding_length_per_layer_input", 2),
            ("ple.ngram_size", 3),
            ("ple.heads_per_ngram", 8),
            ("ple.conv_kernel", 4),
            ("ple.eos_token_id", 258),
        ] {
            meta.insert(format!("qwen4exp.{name}"), ReferenceValue::U32(value));
        }
        for (name, values) in [
            (
                "ple.layer_multipliers",
                vec![23703573157769, 20109073645365, 8052911324071],
            ),
            ("ple.head_offsets", (0..16).map(|i| i * 7).collect()),
            ("ple.head_vocab_sizes", vec![7; 16]),
            ("ple.layers", vec![1]),
        ] {
            meta.insert(
                format!("qwen4exp.{name}"),
                ReferenceValue::Arr(values.into_iter().map(ReferenceValue::U64).collect()),
            );
        }
    }
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut serial = 0usize;
    let mut add = |name: String, dims: Vec<u64>, mode: u8| {
        serial += 1;
        let length = dims.iter().product::<u64>() as usize;
        let values = (0..length)
            .map(|i| {
                let v = (((i * 37 + serial * 17 + i / 3 * 11) % 101) as f32 - 50.) / 512.;
                match mode {
                    1 => 1. + v * 0.3,
                    2 => -0.7 + v,
                    3 => v * 0.5,
                    _ => v,
                }
            })
            .collect();
        tensors.push((name, dims, values));
    };
    add("token_embd.weight".into(), vec![32, 261], 0);
    add("output.weight".into(), vec![32, 261], 0);
    for group in std::iter::once("output_hc_".to_string())
        .chain((0..4).flat_map(|i| [format!("blk.{i}.hc_attn_"), format!("blk.{i}.hc_ffn_")]))
    {
        add(format!("{group}norm.weight"), vec![64], 1);
        add(format!("{group}down.weight"), vec![64, 4], 0);
        add(format!("{group}up.weight"), vec![4, 64], 0);
        if group != "output_hc_" {
            add(format!("{group}inject.weight"), vec![64, 2], 0);
        }
    }
    for layer in 0..4 {
        let p = |s: &str| format!("blk.{layer}.{s}");
        if layer % 2 == 0 {
            add(p("attn_qkv.weight"), vec![32, 32], 0);
            add(p("attn_gate.weight"), vec![32, 16], 0);
            if combined {
                add(p("ssm_ba.weight"), vec![32, 8], 0);
            } else {
                add(p("ssm_beta.weight"), vec![32, 4], 0);
                add(p("ssm_alpha.weight"), vec![32, 4], 0);
            }
            add(p("ssm_dt.bias"), vec![4], 0);
            add(p("ssm_a"), vec![4], 2);
            add(p("ssm_conv1d.weight"), vec![4, 32], 0);
            add(p("ssm_norm.weight"), vec![4], 1);
            add(p("ssm_out.weight"), vec![16, 32], 0);
        } else {
            add(p("attn_q.weight"), vec![32, 64], 0);
            add(p("attn_k.weight"), vec![32, 16], 0);
            add(p("attn_v.weight"), vec![32, 16], 0);
            add(p("attn_output.weight"), vec![32, 32], 0);
            add(p("attn_q_norm.weight"), vec![8], 1);
            add(p("attn_k_norm.weight"), vec![8], 1);
            if indexer {
                add(p("indexer.q_proj.weight"), vec![32, 8], 0);
                add(p("indexer.k_proj.weight"), vec![32, 4], 0);
                add(p("indexer.q_norm.weight"), vec![4], 1);
                add(p("indexer.k_norm.weight"), vec![4], 1);
            }
        }
        add(p("ffn_gate_inp.weight"), vec![32, 4], 0);
        add(p("ffn_gate_exps.weight"), vec![32, 32, 4], 0);
        add(p("ffn_up_exps.weight"), vec![32, 32, 4], 0);
        add(p("ffn_down_exps.weight"), vec![32, 32, 4], 0);
        add(p("ffn_gate_shexp.weight"), vec![32, 32], 0);
        add(p("ffn_up_shexp.weight"), vec![32, 32], 0);
        add(p("ffn_down_shexp.weight"), vec![32, 32], 0);
        add(p("ffn_gate_inp_shexp.weight"), vec![32], 0);
        if ple && layer == 1 {
            add(p("ple_key.weight"), vec![32, 64], 0);
            add(p("ple_value.weight"), vec![32, 32], 0);
            add(p("ple_norm_key.weight"), vec![64], 1);
            add(p("ple_norm_query.weight"), vec![64], 1);
            add(p("ple_norm_conv.weight"), vec![64], 1);
            add(p("ple_conv1d.weight"), vec![4, 64], 3);
        }
    }
    if ple {
        add("per_layer_token_embd.weight".into(), vec![2, 112], 0);
    }
    let mut data = Vec::new();
    let mut descriptions = Vec::new();
    for (name, dims, values) in &tensors {
        let stored = if dims.len() == 1
            || (matches!(storage, Storage::Q4_0 | Storage::Q8_0) && dims[0] % 32 != 0)
        {
            Storage::F32
        } else {
            storage
        };
        string(&mut descriptions, name);
        descriptions.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            descriptions.extend(d.to_le_bytes());
        }
        descriptions.extend(stored.id().to_le_bytes());
        descriptions.extend((data.len() as u64).to_le_bytes());
        data.extend(encode_weights(values, stored));
        data.resize(data.len().next_multiple_of(32), 0);
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
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
        "forge-qwen38-parity-{}-{stamp}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("hybrid.gguf");
    std::fs::write(&path, bytes).unwrap();
    Checkpoint { directory, path }
}

fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 3e-5 + 5e-5 * e.abs();
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance,
            "{label}[{i}]: actual={a} reference={e} delta={} tolerance={tolerance}",
            (a - e).abs()
        );
    }
}
fn run_parity(storage: Storage, combined: bool, indexer: bool, ple: bool) {
    run_parity_with_executor(storage, combined, indexer, ple, None);
}

fn run_parity_with_executor(
    storage: Storage,
    combined: bool,
    indexer: bool,
    ple: bool,
    executor: Option<Arc<dyn MatvecExecutor>>,
) {
    let checkpoint = fixture(storage, combined, indexer, ple);
    let g = Arc::new(Gguf::open_with_cache(&checkpoint.path, 128 * 1024).unwrap());
    let own = if let Some(executor) = executor {
        Model::load_with_executor(g, 64, executor)
    } else {
        Model::load(g, 64)
    }
    .unwrap();
    let reference_g = ReferenceGguf::open(&checkpoint.path).unwrap();
    let oracle = ReferenceModel::load(&reference_g).unwrap();
    let mut reference = ReferenceState::new(&oracle.cfg);
    let mut state = own.new_sequence();
    let tokens = [
        65, 13, 9, 5, 258, 76, 39, 103, 8, 0, 17, 11, 19, 211, 99, 73, 2, 93, 199, 71,
    ];
    for (position, token) in tokens.into_iter().enumerate() {
        let result = own.step(&mut state, token).unwrap();
        let (logits, hidden) = oracle.step_h(&mut reference, token);
        close("all logits", &result.logits, &logits);
        close("HC output head", &result.hidden, &hidden);
        assert_eq!(state.position(), position + 1);
        for (i, layer) in reference.layers.iter().enumerate() {
            match layer {
                LayerState::Attn { k, v, ik } => {
                    let (ak, av) = state.kv(i).unwrap();
                    close("attention K", ak, k);
                    close("attention V", av, v);
                    close("raw indexer K", state.indexer_keys(i).unwrap(), ik);
                }
                LayerState::Recurrent { conv, s } => {
                    let (ac, as_) = state.recurrent(i).unwrap();
                    close("recurrent conv", ac, conv);
                    close("recurrent matrix", as_, s);
                }
            }
        }
        let (history, previous) = state.ple_history().unwrap();
        close("PLE history", history, &reference.ple_hist);
        assert_eq!(previous, reference.ple_prev);
    }
    state.reset();
    let result = own.prefill(&mut state, &tokens).unwrap();
    let mut fresh = own.new_sequence();
    let mut expected = Step::default();
    for token in tokens {
        own.step_into(&mut fresh, token, &mut expected).unwrap();
    }
    close("prefill replay", &result.logits, &expected.logits);
}

#[test]
fn hybrid_hyper_ple_qsa_shared_moe_f32_full_trace() {
    // Explicit opt-in export keeps the checked-in original fixture reproducible
    // without requiring a GPU or writing any files during normal validation.
    if let Some(destination) = std::env::var_os("FORGE_QWEN38_FIXTURE_OUTPUT") {
        let generated = fixture(Storage::F32, false, true, true);
        std::fs::copy(&generated.path, destination).unwrap();
    }
    run_parity(Storage::F32, false, true, true);
}
#[test]
fn hybrid_combined_beta_alpha_full_trace() {
    run_parity(Storage::F32, true, true, true);
}
#[test]
fn hybrid_f16_bf16_full_trace() {
    for storage in [Storage::F16, Storage::BF16] {
        run_parity(storage, false, true, true);
    }
}
#[test]
fn hybrid_q4_q8_full_trace() {
    for storage in [Storage::Q4_0, Storage::Q8_0] {
        run_parity(storage, false, true, true);
    }
}
#[test]
fn hybrid_without_sparse_indexer() {
    run_parity(Storage::F32, false, false, true);
}

#[derive(Debug)]
struct FailOutput {
    enabled: AtomicBool,
}
impl MatvecExecutor for FailOutput {
    fn name(&self) -> &'static str {
        "test_failure_injection"
    }
    fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
        if output.len() == 261 && self.enabled.load(Ordering::Relaxed) {
            anyhow::bail!("injected output failure");
        }
        tensor.matvec_into(input, output)
    }
}
#[test]
fn rejected_projection_and_prefill_restore_all_initialized_state() {
    let file = fixture(Storage::F32, false, true, true);
    let executor = Arc::new(FailOutput {
        enabled: AtomicBool::new(false),
    });
    let own = Model::load_with_executor(
        Arc::new(Gguf::open(&file.path).unwrap()),
        64,
        executor.clone(),
    )
    .unwrap();
    let mut state = own.new_sequence();
    let initial = own
        .prefill(&mut state, &[4, 9, 18, 29, 71, 11, 19, 31, 5])
        .unwrap();
    let recurrent = state.recurrent(0).unwrap().1.to_vec();
    let conv = state.recurrent(0).unwrap().0.to_vec();
    let keys = state.kv(1).unwrap().0.to_vec();
    let (history, prev) = state.ple_history().unwrap();
    let history = history.to_vec();
    executor.enabled.store(true, Ordering::Relaxed);
    assert!(own.step(&mut state, 65).is_err());
    assert_eq!(state.position(), 9);
    assert_eq!(
        state.recurrent(0).unwrap(),
        (conv.as_slice(), recurrent.as_slice())
    );
    assert_eq!(state.kv(1).unwrap().0, keys);
    assert_eq!(state.ple_history().unwrap(), (history.as_slice(), prev));
    assert!(own.prefill(&mut state, &[7, 8, 9]).is_err());
    assert_eq!(state.position(), 9);
    assert_eq!(
        state.recurrent(0).unwrap(),
        (conv.as_slice(), recurrent.as_slice())
    );
    assert_eq!(state.kv(1).unwrap().0, keys);
    assert_eq!(state.ple_history().unwrap(), (history.as_slice(), prev));
    executor.enabled.store(false, Ordering::Relaxed);
    let actual = own.step(&mut state, 65).unwrap();
    let mut fresh = own.new_sequence();
    let expected = own
        .prefill(&mut fresh, &[4, 9, 18, 29, 71, 11, 19, 31, 5, 65])
        .unwrap();
    close("rollback continuation", &actual.logits, &expected.logits);
    assert_ne!(actual.logits, initial.logits);
}
#[test]
fn independent_hybrid_sequences_and_bad_metadata() {
    let file = fixture(Storage::F32, false, true, true);
    let own = Model::load(Arc::new(Gguf::open(&file.path).unwrap()), 64).unwrap();
    let mut a = own.new_sequence();
    let mut b = own.new_sequence();
    own.prefill(&mut a, &[5, 9, 258]).unwrap();
    own.prefill(&mut b, &[8, 3, 99, 18]).unwrap();
    let actual = own.step(&mut a, 71).unwrap();
    let mut fresh = own.new_sequence();
    let expected = own.prefill(&mut fresh, &[5, 9, 258, 71]).unwrap();
    close("independent states", &actual.logits, &expected.logits);
    assert_ne!(a.recurrent(0).unwrap().1, b.recurrent(0).unwrap().1);
    for (key, value) in [
        ("qwen4exp.ssm.time_step_rank", Value::U32(3)),
        ("qwen4exp.attention.indexer.top_k", Value::U32(3)),
        ("qwen4exp.ple.ngram_size", Value::U32(4)),
        ("qwen4exp.nextn_predict_layers", Value::U32(1)),
    ] {
        let mut g = Gguf::open(&file.path).unwrap();
        g.meta.insert(key.into(), value);
        assert!(Model::load(Arc::new(g), 64).is_err(), "must reject {key}");
    }
}

/// Force every projection (including very small HC and one-row shared gates)
/// through Vulkan. Production's size threshold is deliberately bypassed only
/// here, so this gate cannot pass by dispatching these tiny fixtures to CPU.
#[derive(Debug)]
struct ForcedGpu(forge_engine::vulkan::GpuMatvecExecutor);
impl MatvecExecutor for ForcedGpu {
    fn name(&self) -> &'static str {
        "vulkan_forced_test"
    }
    fn matvec(&self, tensor: &Tensor<'_>, input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
        self.0.matvec_gpu(tensor, input, output)
    }
}

#[test]
#[ignore = "requires explicit Vulkan gate; software Vulkan is accepted only for correctness diagnostics"]
fn hybrid_vulkan_all_projection_full_trace() {
    assert_eq!(
        std::env::var("FORGE_VULKAN_TEST").as_deref(),
        Ok("1"),
        "Set FORGE_VULKAN_TEST=1 explicitly before submitting Vulkan compute"
    );
    for storage in [Storage::F32] {
        let executor = Arc::new(ForcedGpu(
            forge_engine::vulkan::GpuMatvecExecutor::new(forge_engine::vulkan::DeviceOptions {
                device_index: None,
                allow_software: true,
                memory_budget_bytes: 64 * 1024 * 1024,
            })
            .unwrap(),
        ));
        run_parity_with_executor(storage, false, true, true, Some(executor.clone()));
        let stats = executor.0.stats();
        assert!(
            stats["gpu_projection_calls"].as_u64().unwrap() > 1000,
            "every projected tensor must execute on Vulkan"
        );
        for counter in [
            "cpu_small_projection_calls",
            "cpu_unsupported_format_calls",
            "cpu_budget_fallback_calls",
        ] {
            assert_eq!(stats[counter].as_u64(), Some(0), "{counter}");
        }
        eprintln!("Qwen3.8 forced Vulkan {storage:?}: {stats}");
    }
}

fn compare_resident_state(
    state: &mut forge_engine::vulkan::QwenGpuSequence,
    reference: &ReferenceState,
) {
    for (i, layer) in reference.layers.iter().enumerate() {
        match layer {
            LayerState::Attn { k, v, ik } => {
                let (actual_k, actual_v) = state.read_kv(i).unwrap();
                close("resident attention K", &actual_k, k);
                close("resident attention V", &actual_v, v);
                if let Some(actual) = state.read_indexer_keys(i).unwrap() {
                    close("resident raw indexer K", &actual, ik);
                } else {
                    assert!(ik.is_empty());
                }
            }
            LayerState::Recurrent { conv, s } => {
                let (actual_conv, actual_matrix) = state.read_recurrent(i).unwrap().unwrap();
                close("resident recurrent convolution", &actual_conv, conv);
                close("resident recurrent matrix", &actual_matrix, s);
            }
        }
    }
    if let Some((history, previous)) = state.read_ple_history().unwrap() {
        close("resident PLE history", &history, &reference.ple_hist);
        assert_eq!(previous, reference.ple_prev);
    } else {
        assert!(reference.ple_hist.is_empty());
    }
}

/// Device-resident nonlinear operations and all persistent state must agree
/// with the independent numerical oracle. This is distinct from the projection
/// executor gate above: no CPU recurrent, HC, PLE, QSA or attention path is used.
#[test]
#[ignore = "requires explicit Vulkan gate; software Vulkan is a correctness diagnostic only"]
fn hybrid_vulkan_resident_state_full_trace() {
    assert_eq!(
        std::env::var("FORGE_VULKAN_TEST").as_deref(),
        Ok("1"),
        "Set FORGE_VULKAN_TEST=1 explicitly before submitting Vulkan compute"
    );
    let tokens = [
        65, 13, 9, 5, 258, 76, 39, 103, 8, 0, 17, 11, 19, 211, 99, 73, 2, 93, 199, 71,
    ];
    for (storage, combined, indexer, ple) in [
        (Storage::F32, false, true, true),
        (Storage::Q4_0, true, true, true),
        (Storage::F32, false, false, true),
    ] {
        let file = fixture(storage, combined, indexer, ple);
        let source = Arc::new(Gguf::open_with_cache(&file.path, 128 * 1024).unwrap());
        let gpu = forge_engine::vulkan::QwenGpuModel::load(
            source,
            64,
            forge_engine::vulkan::DeviceOptions {
                device_index: None,
                allow_software: true,
                memory_budget_bytes: 64 * 1024 * 1024,
            },
        )
        .unwrap();
        let reference_g = ReferenceGguf::open(&file.path).unwrap();
        let oracle = ReferenceModel::load(&reference_g).unwrap();
        let mut reference = ReferenceState::new(&oracle.cfg);
        let mut state = gpu.new_sequence().unwrap();
        let mut other = gpu.new_sequence().unwrap();
        let mut actual = Step::default();
        let mut expected_logits = Vec::new();
        let mut expected_hidden = Vec::new();
        for (position, token) in tokens.into_iter().enumerate() {
            gpu.step_into(&mut state, token, &mut actual).unwrap();
            (expected_logits, expected_hidden) = oracle.step_h(&mut reference, token);
            close("resident all logits", &actual.logits, &expected_logits);
            close("resident HC output", &actual.hidden, &expected_hidden);
            assert_eq!(state.position(), position + 1);
            compare_resident_state(&mut state, &reference);
            if position == 7 {
                // Interleave a different history on the same immutable model;
                // the following original-sequence tokens must remain unchanged.
                let mut other_reference = ReferenceState::new(&oracle.cfg);
                for alternate in [71, 19, 258, 5, 6] {
                    let result = gpu.step(&mut other, alternate).unwrap();
                    let (logits, hidden) = oracle.step_h(&mut other_reference, alternate);
                    close("resident interleaved logits", &result.logits, &logits);
                    close("resident interleaved hidden", &result.hidden, &hidden);
                }
                compare_resident_state(&mut other, &other_reference);
            }
        }
        let stats = gpu.memory_stats();
        assert_eq!(stats["cpu_state_operations"].as_bool(), Some(false));
        assert_eq!(stats["selected_experts_omitted"].as_u64(), Some(0));
        assert_eq!(stats["cpu_expert_fallbacks"].as_u64(), Some(0));
        assert!(stats["gpu_expert_executions"].as_u64().unwrap() > 0);

        // A rejected token is a preflight error, not a submitted failing graph.
        assert!(gpu.step(&mut state, gpu.vocab_size() as u32).is_err());
        assert_eq!(state.position(), tokens.len());
        compare_resident_state(&mut state, &reference);

        state.reset().unwrap();
        gpu.prefill_into(&mut state, &tokens[..tokens.len() - 1], &mut actual)
            .unwrap();
        let before = gpu.memory_stats();
        gpu.step_into(&mut state, *tokens.last().unwrap(), &mut actual)
            .unwrap();
        let after = gpu.memory_stats();
        close(
            "resident reset/prefill logits",
            &actual.logits,
            &expected_logits,
        );
        close(
            "resident reset/prefill hidden",
            &actual.hidden,
            &expected_hidden,
        );
        compare_resident_state(&mut state, &reference);
        let submissions = after["queue_submissions"].as_u64().unwrap()
            - before["queue_submissions"].as_u64().unwrap();
        assert!(
            submissions <= gpu.config().block_count as u64 + 1,
            "warm graph should submit at most once per layer plus output; observed {submissions}"
        );
        eprintln!(
            "Qwen3.8 resident Vulkan {storage:?} combined={combined} indexer={indexer} ple={ple}: warm submissions={submissions}; {after}"
        );
    }
}

#[test]
#[ignore = "requires explicit Vulkan compute; intentionally tests a submitted numerical error"]
fn hybrid_vulkan_submitted_error_requires_reset() {
    use std::io::{Seek, SeekFrom, Write};
    assert_eq!(
        std::env::var("FORGE_VULKAN_TEST").as_deref(),
        Ok("1"),
        "Set FORGE_VULKAN_TEST=1 explicitly before submitting Vulkan compute"
    );
    let file = fixture(Storage::F32, false, true, true);
    let bad_token = 260u32;
    let (offset, width) = {
        let g = Gguf::open(&file.path).unwrap();
        let embedding = g.tensor("token_embd.weight").unwrap();
        (
            embedding.cache_key().offset + (bad_token as u64 * embedding.width() as u64 * 4),
            embedding.width(),
        )
    };
    // The stored row is finite, so lazy decoding succeeds. Its squared norm
    // overflows only after GPU submission, exercising the runtime poison path
    // rather than a metadata/token preflight rejection. Other rows stay valid.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .open(&file.path)
        .unwrap();
    output.seek(SeekFrom::Start(offset)).unwrap();
    for _ in 0..width {
        output.write_all(&1e30f32.to_le_bytes()).unwrap();
    }
    drop(output);
    let gpu = forge_engine::vulkan::QwenGpuModel::load(
        Arc::new(Gguf::open(&file.path).unwrap()),
        64,
        forge_engine::vulkan::DeviceOptions {
            device_index: None,
            allow_software: true,
            memory_budget_bytes: 64 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut state = gpu.new_sequence().unwrap();
    gpu.prefill(&mut state, &[65, 13, 9, 5, 258]).unwrap();
    let result = gpu.step(&mut state, bad_token);
    assert!(
        result.is_err(),
        "overflowing GPU norm must reject the token"
    );
    let error = result.err().unwrap();
    assert!(format!("{error:#}").contains("non-finite"));
    assert_eq!(state.position(), 5);
    let result = gpu.step(&mut state, 13);
    assert!(result.is_err(), "failed GPU state must require reset");
    let error = result.err().unwrap();
    assert!(format!("{error:#}").contains("requires reset"));
    state.reset().unwrap();
    assert_eq!(state.position(), 0);
    for layer in 0..gpu.config().block_count {
        if let Some((convolution, matrix)) = state.read_recurrent(layer).unwrap() {
            assert!(convolution.iter().chain(&matrix).all(|x| *x == 0.0));
        }
    }
    let (history, previous) = state.read_ple_history().unwrap().unwrap();
    assert!(history.iter().all(|x| *x == 0.0));
    assert_eq!(previous, [-1, -1]);
    let reference_g = ReferenceGguf::open(&file.path).unwrap();
    let oracle = ReferenceModel::load(&reference_g).unwrap();
    let mut reference = ReferenceState::new(&oracle.cfg);
    let (logits, hidden) = oracle.step_h(&mut reference, 13);
    let recovered = gpu.step(&mut state, 13).unwrap();
    close(
        "reset after submitted failure logits",
        &recovered.logits,
        &logits,
    );
    close(
        "reset after submitted failure hidden",
        &recovered.hidden,
        &hidden,
    );
    compare_resident_state(&mut state, &reference);
}
