//! A real, tiny CPU forward pass. The committed GGUF is reproducible with
//! scripts/make_tiny_fixture.py; no trained weights or GPU are needed.
use rusting_strata::{
    gguf::Gguf,
    model::{LayerState, Model, State},
    ops,
    tokenizer::Tokenizer,
};
use std::path::Path;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny-qwen3.gguf")
}

#[test]
fn moe_fixture_executes_routing_and_changes_with_active_expert_count() {
    use rusting_strata::gguf::Value;
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3moe.gguf");
    let g=Gguf::open(&path).unwrap();
    let model=Model::load(&g).unwrap();
    assert_eq!((model.cfg.n_expert,model.cfg.n_expert_used),(4,2));
    let mut state=State::new(&model.cfg);
    let (logits,two)=model.step_h(&mut state,b'h' as u32);
    assert_eq!(ops::argmax(&logits),b'A' as usize);
    assert!(two.iter().chain(&logits).all(|v|v.is_finite()));
    let mut one_g=Gguf::open(&path).unwrap();
    one_g.meta.insert("qwen3moe.expert_used_count".into(),Value::U32(1));
    let one_model=Model::load(&one_g).unwrap();
    let (_,one)=one_model.step_h(&mut State::new(&one_model.cfg),b'h' as u32);
    assert!(one.iter().zip(&two).any(|(a,b)|(a-b).abs()>1e-6),
        "the routed expert mixture must affect the actual transformer hidden state");
}

#[test]
fn cpu_forward_updates_attention_and_returns_expected_tokens() {
    let g = Gguf::open(&fixture()).expect("load committed fixture");
    let model = Model::load(&g).expect("load real Qwen3 model");
    let tok = Tokenizer::from_gguf(&g).expect("build full byte tokenizer");
    let prompt = tok.encode("hello").unwrap();
    assert_eq!(tok.decode(&prompt).unwrap(), "hello");
    let mut state = State::new(&model.cfg);
    let mut logits = Vec::new();
    let mut hidden_states = Vec::new();
    for &id in &prompt {
        let (next, hidden) = model.step_h(&mut state, id);
        logits = next;
        hidden_states.push(hidden);
    }
    assert_eq!(state.pos, prompt.len());
    assert!(hidden_states.windows(2).any(|pair| pair[0] != pair[1]));
    match &state.layers[0] {
        LayerState::Attn { k, v, .. } => {
            assert_eq!(k.len(), prompt.len() * model.cfg.head_dim);
            assert_eq!(v.len(), k.len());
            assert!(k.iter().chain(v).all(|x| x.is_finite()));
            assert!(k.iter().any(|x| x.abs() > 0.01));
            assert!(v.iter().any(|x| x.abs() > 0.01));
        }
        _ => panic!("fixture must exercise attention"),
    }
    let mut generated = Vec::new();
    for _ in 0..4 {
        assert_eq!(logits.len(), model.cfg.n_vocab);
        assert!(logits.iter().all(|value| value.is_finite()));
        let id = ops::argmax(&logits) as u32;
        generated.push(id);
        logits = model.step(&mut state, id);
    }
    assert_eq!(tok.decode(&generated).unwrap(), "AAAA");
    assert_eq!(state.pos, prompt.len() + 4);
}

#[test]
fn fixture_round_trips_unicode_through_byte_tokenizer() {
    let g = Gguf::open(&fixture()).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    for text in ["hello world\n", "£ café", "日本語", "😀"] {
        assert_eq!(tok.decode(&tok.encode(text).unwrap()).unwrap(), text);
    }
}

#[test]
fn truncated_tensor_data_is_rejected() {
    let bytes = std::fs::read(fixture()).unwrap();
    let path = std::env::temp_dir().join(format!(
        "forge9070-{}-truncated.gguf", std::process::id()
    ));
    std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
    let result = Gguf::open(&path);
    assert!(result.is_err(), "truncated tensor must not be accepted");
    drop(result);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn synthesized_weights_are_owned_and_survive_clearing_lora_overrides() {
    use rusting_strata::{gguf::Value, weights::Weights};
    let mut g = Gguf::open(&fixture()).unwrap();
    g.meta.insert("qwen3.ssm.group_count".into(), Value::U32(1));
    g.meta.insert("qwen3.ssm.time_step_rank".into(), Value::U32(2));
    let mut combined = g.tensors["blk.0.attn_q.weight"].clone();
    combined.name = "blk.0.ssm_ba.weight".into();
    combined.dims = vec![32, 4];
    g.tensors.insert(combined.name.clone(), combined);
    let mut weights = Weights::new(&g).unwrap();
    assert_eq!(weights.owned_bytes(), 32 * 4 * 4);
    let beta = weights.layer(0, "ssm_beta.weight").unwrap().to_f32();
    let alpha = weights.layer(0, "ssm_alpha.weight").unwrap().to_f32();
    assert_eq!(beta.len(), 64);
    assert_eq!(alpha.len(), 64);
    weights.clear_baked();
    assert_eq!(weights.owned_bytes(), 512);
    assert_eq!(weights.layer(0, "ssm_beta.weight").unwrap().to_f32(), beta);
    assert_eq!(weights.layer(0, "ssm_alpha.weight").unwrap().to_f32(), alpha);
}
