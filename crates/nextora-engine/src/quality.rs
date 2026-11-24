//! Model-level deterministic quality gate for a source and a candidate checkpoint.
//! This checks actual next-token logit distributions and reference-token
//! teacher-forcing on identical raw prompts. It is not a coding benchmark,
//! blind test, or proof of real-world usefulness.
use crate::{
    runtime::{Backend, BackendArgs, RuntimeModel},
};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde_json::{Value,json};
use std::{path::PathBuf, time::Instant};

#[derive(Args, Debug)]
pub struct QualityArgs {
    /// Original pretrained checkpoint, complete single- or multi-shard GGUF.
    #[arg(long)]
    pub reference: PathBuf,
    /// Candidate quantized checkpoint with the same tokenizer and architecture.
    #[arg(long)]
    pub candidate: PathBuf,
    /// Fixed raw prompt; repeat this option for up to 16 independent cases.
    #[arg(long = "prompt", required = true)]
    pub prompts: Vec<String>,
    /// Maximum scored next-token distributions per prompt.
    #[arg(long, default_value_t = 16)]
    pub steps: usize,
    /// Context for each sequence, including the fixed prompt and generated steps.
    #[arg(long, default_value_t = 4096)]
    pub ctx: usize,
    /// Candidate device; use cpu for quantization-only comparisons or vulkan for parity.
    #[arg(long, value_enum, default_value = "cpu")]
    pub candidate_backend: Backend,
    /// Vulkan adapter index (only with --candidate-backend vulkan).
    #[arg(long)]
    pub candidate_device: Option<usize>,
    /// Explicit candidate GPU allocation cap in MiB (0 means automatic).
    #[arg(long, default_value_t = 0)]
    pub candidate_vram_mib: usize,
    /// Reserved host memory allowance in MiB.
    #[arg(long, default_value_t = 1024)]
    pub ram_margin_mib: usize,
    /// Upper limit on mean KL(reference softmax || candidate softmax).
    #[arg(long, default_value_t = 0.1)]
    pub max_mean_kl: f64,
    /// Required fraction of reference/candidate top-1 tokens in agreement.
    #[arg(long, default_value_t = 0.8)]
    pub min_top1_agreement: f64,
    /// Write complete measured report including failed gates (never overwrite a model shard).
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct LogitStats {
    reference_top: u32,
    candidate_top: u32,
    mean_abs_delta: f64,
    max_abs_delta: f64,
    kl_reference_to_candidate: f64,
}

fn check_logits(a:&[f32],b:&[f32])->Result<LogitStats>{
    ensure!(!a.is_empty() && a.len()==b.len(), "reference and candidate vocabulary logits disagree");
    ensure!(a.iter().chain(b).all(|v|v.is_finite()),
        "non-finite model logits; cannot accept quality comparison");
    let mut a_top=0usize;
    let mut b_top=0usize;
    let mut max_abs=0f64;
    let mut sum_abs=0f64;
    let mut a_max=f64::NEG_INFINITY;
    let mut b_max=f64::NEG_INFINITY;
    for (i,(&x,&y)) in a.iter().zip(b).enumerate(){
        if x>a[a_top] {a_top=i;}
        if y>b[b_top] {b_top=i;}
        let delta=(f64::from(x)-f64::from(y)).abs();
        max_abs=max_abs.max(delta);
        sum_abs+=delta;
        a_max=a_max.max(f64::from(x));
        b_max=b_max.max(f64::from(y));
    }
    let a_total:f64=a.iter().map(|&v|(f64::from(v)-a_max).exp()).sum();
    let b_total:f64=b.iter().map(|&v|(f64::from(v)-b_max).exp()).sum();
    ensure!(a_total>0. && b_total>0. && a_total.is_finite() && b_total.is_finite(),
        "invalid softmax normalizer");
    let log_a=a_max+a_total.ln();
    let log_b=b_max+b_total.ln();
    let kl=a.iter().zip(b).map(|(&x,&y)|{
        let logp=f64::from(x)-log_a;
        logp.exp()*(logp-f64::from(y)+log_b)
    }).sum::<f64>().max(0.);
    ensure!(kl.is_finite(),"non-finite softmax KL");
    Ok(LogitStats {
        reference_top:u32::try_from(a_top)?,
        candidate_top:u32::try_from(b_top)?,
        mean_abs_delta:sum_abs/a.len() as f64,
        max_abs_delta:max_abs,
        kl_reference_to_candidate:kl,
    })
}

fn backend_options(backend:Backend, args:&QualityArgs)->BackendArgs{
    BackendArgs{
        cpu:false, backend:Some(backend),
        device:if backend==Backend::Vulkan {args.candidate_device} else {None},
        allow_software_vulkan:false,
        vram_mib:if backend==Backend::Vulkan {args.candidate_vram_mib} else {0},
        expert_ram_mib:Some(0),
        ram_margin_mib:args.ram_margin_mib,
        expert_warmup:false,
        state_slots:1,
    }
}

fn validate_models(reference:&RuntimeModel,candidate:&RuntimeModel)->Result<()>{
    ensure!(
        serde_json::to_value(reference.config())?==serde_json::to_value(candidate.config())?,
        "model geometry or architecture differs; compare matching checkpoints only"
    );
    ensure!(
        reference.tokenizer.vocab_size()==candidate.tokenizer.vocab_size(),
        "tokenizer vocabulary sizes differ"
    );
    let ref_tokenizer=reference.source.meta.iter().filter(|(key,_)|key.starts_with("tokenizer."));
    let candidate_tokenizer=candidate.source.meta.iter().filter(|(key,_)|key.starts_with("tokenizer."));
    ensure!(
        ref_tokenizer.eq(candidate_tokenizer),
        "tokenizer metadata differs; token IDs cannot be compared reliably"
    );
    ensure!(
        reference.source.tensors.len()==candidate.source.tensors.len(),
        "weight tensor inventories differ"
    );
    for (name, info) in &reference.source.tensors {
        let other=candidate.source.tensors.get(name)
            .with_context(|| format!("candidate missing tensor {name}"))?;
        ensure!(info.dims==other.dims,"candidate tensor geometry differs at {name}");
    }
    Ok(())
}

pub fn compare(args:&QualityArgs)->Result<Value>{
    ensure!((1..=16).contains(&args.prompts.len()),"quality comparison requires 1..=16 prompts");
    ensure!((1..=128).contains(&args.steps),"--steps must be in 1..=128");
    ensure!((16..=131072).contains(&args.ctx),"--ctx must be in 16..=131072");
    ensure!(args.ctx>args.steps,"context must exceed reference-step count");
    ensure!(args.prompts.iter().all(|p|!p.is_empty() && p.len()<=16384),
        "prompts must be nonempty and at most 16 KiB");
    ensure!(args.max_mean_kl.is_finite() && (0.0..=10.0).contains(&args.max_mean_kl),
        "max-mean-kl must be finite in 0..=10");
    ensure!(args.min_top1_agreement.is_finite() && (0.0..=1.0).contains(&args.min_top1_agreement),
        "min-top1-agreement must be finite in 0..=1");
    ensure!(matches!(args.candidate_backend,Backend::Cpu|Backend::Vulkan),
        "quality candidate backend must be cpu or vulkan (no auto or diagnostic hybrid)");
    ensure!(args.candidate_backend==Backend::Vulkan
        || (args.candidate_device.is_none() && args.candidate_vram_mib==0),
        "candidate Vulkan device/cap requires --candidate-backend vulkan");
    ensure!(
        args.reference!=args.candidate,
        "use different source and candidate paths for a meaningful quality assessment"
    );

    let started=Instant::now();
    let reference=RuntimeModel::load(&args.reference,args.ctx,&backend_options(Backend::Cpu,args))?;
    let candidate=RuntimeModel::load(&args.candidate,args.ctx,&backend_options(args.candidate_backend,args))?;
    validate_models(&reference,&candidate)?;
    let mut total=0usize;
    let mut agree=0usize;
    let mut sum_kl=0f64;
    let mut sum_abs=0f64;
    let mut worst_kl=0f64;
    let mut worst_abs=0f64;
    let mut cases=Vec::new();
    for (i,prompt) in args.prompts.iter().enumerate(){
        let tokens=reference.tokenizer.encode(prompt)?;
        ensure!(!tokens.is_empty(),"empty tokenized prompt");
        ensure!(tokens==candidate.tokenizer.encode(prompt)?,
            "prompt tokenization differs between source and candidate");
        ensure!(tokens.len().checked_add(args.steps).is_some_and(|n|n<=args.ctx),
            "prompt {} plus steps exceeds context",i+1);
        let mut rstate=reference.new_sequence()?;
        let mut cstate=candidate.new_sequence()?;
        let mut rstep=reference.prefill(&mut rstate,&tokens)?;
        let mut cstep=candidate.prefill(&mut cstate,&tokens)?;
        let mut count=0usize;
        let mut agreement=0usize;
        let mut kl_sum=0f64;
        let mut max_kl=0f64;
        for step in 0..args.steps {
            let stats=check_logits(&rstep.logits,&cstep.logits)?;
            count+=1;
            total+=1;
            if stats.reference_top==stats.candidate_top {agreement+=1;agree+=1;}
            sum_kl+=stats.kl_reference_to_candidate;
            kl_sum+=stats.kl_reference_to_candidate;
            max_kl=max_kl.max(stats.kl_reference_to_candidate);
            worst_kl=worst_kl.max(stats.kl_reference_to_candidate);
            worst_abs=worst_abs.max(stats.max_abs_delta);
            sum_abs+=stats.mean_abs_delta;
            if step+1==args.steps || reference.tokenizer.is_eos(stats.reference_top) {break;}
            // Use the reference token in BOTH sequences. If candidate chooses
            // another token, free generation would diverge and confound the
            // per-step distribution comparison.
            reference.step_into(&mut rstate,stats.reference_top,&mut rstep)?;
            candidate.step_into(&mut cstate,stats.reference_top,&mut cstep)?;
        }
        cases.push(json!({
            "case_index":i+1,
            "prompt_tokens":tokens.len(),
            "scored_positions":count,
            "top1_agreement":agreement as f64/count as f64,
            "mean_kl_reference_to_candidate":kl_sum/count as f64,
            "worst_kl_reference_to_candidate":max_kl,
            "prompt_content_exported":false,
        }));
    }
    ensure!(total>0,"no quality positions scored");
    let mean_kl=sum_kl/total as f64;
    let top1=agree as f64/total as f64;
    let passed=mean_kl<=args.max_mean_kl && top1>=args.min_top1_agreement;
    Ok(json!({
        "schema":"nextora.quality.v1",
        "engine":crate::ENGINE_ID,
        "version":crate::ENGINE_VERSION,
        "passed":passed,
        "reference_backend":"cpu",
        "candidate_backend":match args.candidate_backend {Backend::Cpu=>"cpu",Backend::Vulkan=>"vulkan",_=>"invalid"},
        "reference_model_mapped_bytes":reference.source.mapped_bytes(),
        "candidate_model_mapped_bytes":candidate.source.mapped_bytes(),
        "prompts":args.prompts.len(),
        "scored_positions":total,
        "top1_agreement":top1,
        "mean_kl_reference_to_candidate":mean_kl,
        "worst_kl_reference_to_candidate":worst_kl,
        "mean_absolute_logit_delta":sum_abs/total as f64,
        "maximum_absolute_logit_delta":worst_abs,
        "max_mean_kl_allowed":args.max_mean_kl,
        "min_top1_agreement_required":args.min_top1_agreement,
        "cases":cases,
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "model_paths_exported":false,
        "prompt_text_exported":false,
        "input_tokenizer_metadata_equal":true,
        "model_tensor_shapes_equal":true,
        "verification_scope":"controlled raw-prompt teacher-forced logit comparison; not independent model/task correctness, coding-benchmark accuracy, training calibration or a hardware speed benchmark",
        "hardware_performance_verified":false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_logits_are_zero_kl_and_identical_top1(){
        let a=check_logits(&[0.,1.,2.,-1.],&[0.,1.,2.,-1.]).unwrap();
        assert_eq!(a.kl_reference_to_candidate,0.);
        assert_eq!(a.mean_abs_delta,0.);
        assert_eq!(a.reference_top,a.candidate_top);
    }
    #[test]
    fn different_logits_show_positive_divergence(){
        let a=check_logits(&[3.,0.,-2.],&[-2.,0.,3.]).unwrap();
        assert!(a.kl_reference_to_candidate>0.1);
        assert_ne!(a.reference_top,a.candidate_top);
        assert!(a.max_abs_delta>0.);
    }
    #[test]
    fn unsafe_logits_fail_closed(){
        assert!(check_logits(&[f32::NAN],&[0.]).is_err());
        assert!(check_logits(&[0.,1.],&[0.]).is_err());
        assert!(check_logits(&[],&[]).is_err());
    }
}
