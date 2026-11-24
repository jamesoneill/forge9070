//! Explicit real-checkpoint same-input trajectory comparison. Never runs as part
//! of normal tests, and never downloads a model.
use super::*;

#[test]
#[ignore = "requires an idle hardware GPU and NEXTORA_DIRECT_MODEL local checkpoint"]
fn hardware_direct_experts_forced_trajectory_matches_packed() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let path=std::env::var_os("NEXTORA_DIRECT_MODEL").expect("set NEXTORA_DIRECT_MODEL to an existing local GGUF");
    let steps:usize=std::env::var("NEXTORA_DIRECT_STEPS").ok().map(|s|s.parse().unwrap()).unwrap_or(128);
    assert!((1..=128).contains(&steps));
    let mut source=Gguf::open(path).unwrap();
    source.set_expert_cache_capacity(24*1024*1024*1024).unwrap();
    let source=Arc::new(source);
    let warm=source.warm_expert_cache().unwrap();
    eprintln!("shared host expert warmup: {warm:?}");
    let fast_math=std::env::var("NEXTORA_DIRECT_FAST_MATH").map(|v|v!="0"&&v!="false").unwrap_or(true);
    let options=|direct| DeviceOptions {
        direct_experts:Some(direct),fast_math,
        memory_budget_bytes:14000*1024*1024,
        ..Default::default()
    };
    let tokenizer=crate::tokenizer::Tokenizer::from_gguf(&source).unwrap();
    let prompt=std::env::var("NEXTORA_DIRECT_PROMPT").unwrap_or_else(|_|"Write a Python function that returns the prime numbers up to n.".into());
    let prompt=tokenizer.encode(&prompt).unwrap();
    let mut forced=Vec::with_capacity(steps);
    let mut truth=Vec::<Vec<f32>>::with_capacity(steps);
    {
        let model=GpuModel::load(Arc::clone(&source),40960,options(false),&|_,_,_,_,_|{}).unwrap();
        assert!(!model.device.direct_experts,"explicit false must retain the legacy pipeline");
        let mut sequence=model.new_sequence().unwrap();
        let mut output=model.prefill(&mut sequence,&prompt).unwrap();
        for _ in 0..steps {
            let next=crate::runtime::greedy(&output.logits).unwrap();
            forced.push(next);truth.push(output.logits);
            output=model.step(&mut sequence,next).unwrap();
        }
        assert!(model.packed_bytes.load(Ordering::Relaxed)>0);
        eprintln!("packed mode: {} steps, stats={}",steps,model.memory_stats());
    }
    let mut max_abs=0.0f64;
    let mut max_scaled=0.0f64;
    let mut square=0.0f64;
    let mut total=0usize;
    let mut top_matches=0usize;
    let mut exact_logits=0usize;
    let mut first=None;
    {
        let model=GpuModel::load(Arc::clone(&source),40960,options(true),&|_,_,_,_,_|{}).unwrap();
        assert!(model.device.direct_experts,"force-direct cannot silently fall back");
        let mut sequence=model.new_sequence().unwrap();
        let mut output=model.prefill(&mut sequence,&prompt).unwrap();
        for (step,(&token,expected)) in forced.iter().zip(&truth).enumerate() {
            assert_eq!(output.logits.len(),expected.len());
            let mut step_abs=0.0f64;let mut step_scaled=0.0f64;let mut exact=0;
            for (&actual,&reference) in output.logits.iter().zip(expected) {
                assert!(actual.is_finite()&&reference.is_finite());
                let delta=(actual as f64-reference as f64).abs();
                step_abs=step_abs.max(delta);step_scaled=step_scaled.max(delta/(1.0+reference.abs() as f64));
                square+=delta*delta;total+=1;exact+=usize::from(actual.to_bits()==reference.to_bits());
            }
            let actual=crate::runtime::greedy(&output.logits).unwrap();
            top_matches+=usize::from(actual==token);
            if first.is_none()&&actual!=token {first=Some(step);}
            eprintln!("direct trajectory step={step} max_abs={step_abs:.9} scaled={step_scaled:.9} exact={exact}/{} packed_top={token} direct_top={actual}",expected.len());
            exact_logits+=exact;
            max_abs=max_abs.max(step_abs);max_scaled=max_scaled.max(step_scaled);
            output=model.step(&mut sequence,token).unwrap();
        }
        if std::env::var("NEXTORA_DIRECT_PACKED_CONTROL").as_deref()==Ok("1") {
            assert!(model.packed_bytes.load(Ordering::Relaxed)>0);
            assert_eq!(model.direct_source_bytes.load(Ordering::Relaxed),0);
        } else {
            assert_eq!(model.packed_bytes.load(Ordering::Relaxed),0);
            assert!(model.direct_source_bytes.load(Ordering::Relaxed)>0);
        }
        eprintln!("direct mode stats={}",model.memory_stats());
    }
    eprintln!("DIRECT_FORCED_TRAJECTORY {}",json!({"steps":steps,"logits":total,"bit_exact_logits":exact_logits,"max_abs":max_abs,"max_scaled":max_scaled,"rms":(square/total as f64).sqrt(),"top_token_matches":top_matches,"first_top_divergence":first}));
    // Explicitly diagnose numerical equivalence rather than blessing a speed
    // optimization whose same-input token choices changed.
    assert_eq!(top_matches,steps,"same-input direct expert trajectory changed greedy choices; inspect full logit deltas");
    assert_eq!(exact_logits,total,"direct expert path must preserve every logit bit on the reference trajectory");
}

/// Timing is separate from the logit comparison: no large logit copies or
/// diagnostics inside the measured decode loop. Source experts are warmed once
/// and reused across an A/B/A comparison to expose order/temperature effects.
#[test]
#[ignore = "requires an idle hardware GPU and NEXTORA_DIRECT_MODEL local checkpoint"]
fn hardware_direct_experts_paired_throughput() {
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(), Ok("1"));
    let path=std::env::var_os("NEXTORA_DIRECT_MODEL").expect("existing local GGUF required");
    let mut source=Gguf::open(path).unwrap();
    source.set_expert_cache_capacity(24*1024*1024*1024).unwrap();
    let source=Arc::new(source);
    eprintln!("shared host expert warmup: {:?}",source.warm_expert_cache().unwrap());
    let tokenizer=crate::tokenizer::Tokenizer::from_gguf(&source).unwrap();
    let text=std::env::var("NEXTORA_DIRECT_PROMPT").unwrap_or_else(|_|"fn square(x: i32) -> i32 {\n    // Return the square of x.\n".into());
    let prompt=tokenizer.encode(&text).unwrap();
    let mut expected=None;
    let mut results=Vec::new();
    for (run,direct) in [false,true,false].into_iter().enumerate() {
        let model=GpuModel::load(Arc::clone(&source),40960,DeviceOptions {
            direct_experts:Some(direct),fast_math:true,
            memory_budget_bytes:14000*1024*1024,..Default::default()
        },&|_,_,_,_,_|{}).unwrap();
        let mut samples=Vec::new();
        for repetition in 0..5 {
            let mut sequence=model.new_sequence().unwrap();
            let prefill_start=std::time::Instant::now();
            let mut output=model.prefill(&mut sequence,&prompt).unwrap();
            let prefill=prefill_start.elapsed().as_secs_f64();
            let before=model.memory_stats();
            let start=std::time::Instant::now();
            let mut checksum=0xcbf29ce484222325u64;
            for _ in 0..128 {
                let token=crate::runtime::greedy(&output.logits).unwrap();
                checksum=(checksum^token as u64).wrapping_mul(0x100000001b3);
                model.step_into(&mut sequence,token,&mut output).unwrap();
            }
            let seconds=start.elapsed().as_secs_f64();
            let after=model.memory_stats();
            if let Some(previous)=expected {assert_eq!(checksum,previous,"paired benchmark changed greedy trajectory");} else {expected=Some(checksum);}
            let sample=json!({"repetition":repetition,"warmup":repetition<2,"steps":128,"seconds":seconds,"tok_s":128.0/seconds,"prefill_seconds":prefill,"checksum":checksum,"before":before,"after":after});
            eprintln!("DIRECT_PAIRED_SAMPLE {}",json!({"run":run,"direct":direct,"sample":sample}));
            if repetition>=2 {samples.push(sample);}
        }
        if direct {assert_eq!(model.packed_bytes.load(Ordering::Relaxed),0);} else {assert!(model.packed_bytes.load(Ordering::Relaxed)>0);}
        results.push(json!({"run":run,"direct":direct,"samples":samples}));
    }
    eprintln!("DIRECT_PAIRED_THROUGHPUT {}",json!({"prompt_tokens":prompt.len(),"context":40960,"fast_math":true,"runs":results}));
}
