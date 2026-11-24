//! Size-targeted *offline* GGUF mixed precision using per-tensor reconstruction error.
//! This is a deterministic error-per-byte heuristic, NOT an importance-matrix,
//! task-quality or globally optimal quantisation algorithm.
use super::{align, eligible, encode_row};
use crate::{gguf::{Gguf, Tensor}, quant::QuantType};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value,json};
use std::collections::BTreeMap;

const MAX_MATRIX_WIDTH: usize = 1_048_576;
const MAX_CHOICE_TENSORS: usize = 4096;

#[derive(Clone, Debug)]
struct Choice {
    ty: QuantType,
    bytes: u64,
    error_sse: f64,
    nrmse: f64,
}
struct Matrix {
    name: String,
    choices: Vec<Choice>,
    selected: usize,
}
#[derive(Debug)]
pub(super) struct Plan {
    pub(super) formats: BTreeMap<String, QuantType>,
    pub(super) estimated_file_bytes: u64,
    pub(super) goal_bytes: u64,
    pub(super) report: Value,
}

fn rows_and_size(tensor: &Tensor<'_>, ty: QuantType) -> Result<(usize, u64)> {
    let width=tensor.width();
    let rows=tensor.rows();
    let stride=u64::try_from(ty.row_bytes(width)?)?;
    let bytes=u64::try_from(rows)?.checked_mul(stride).context("candidate matrix size overflow")?;
    Ok((rows,bytes))
}

/// Read each source row once, apply an exact independent runtime decode of the
/// candidate encoding, and accumulate squared error in F64 without retaining
/// any tensor-sized arrays. An unsuitable scale is an ineligible candidate.
fn measure(tensor:&Tensor<'_>, target:QuantType) -> Result<Option<Choice>> {
    let width=tensor.width();
    ensure!(width>0 && width<=MAX_MATRIX_WIDTH, "offline quantizer matrix width exceeds limit");
    let (rows, bytes)=rows_and_size(tensor,target)?;
    let mut original=vec![0f32;width];
    let mut reconstructed=vec![0f32;width];
    let mut encoded=Vec::new();
    let mut source_energy=0f64;
    let mut diff_energy=0f64;
    for row in 0..rows {
        tensor.read_row(row,&mut original)?;
        // Failure to represent a scale means we leave this particular matrix
        // in its original form (not that we silently accept corrupted weights).
        if encode_row(&original,target,&mut encoded).is_err() {
            return Ok(None);
        }
        crate::quant::decode_row(target,&encoded,&mut reconstructed)?;
        for (a,b) in original.iter().zip(&reconstructed) {
            let a=f64::from(*a);
            let delta=a-f64::from(*b);
            source_energy+=a*a;
            diff_energy+=delta*delta;
        }
    }
    let nrmse=if source_energy==0. {
        if diff_energy==0. {0.} else {f64::INFINITY}
    } else {(diff_energy/source_energy).sqrt()};
    ensure!(nrmse.is_finite() && diff_energy.is_finite(),
        "non-finite matrix error during quantization assessment");
    Ok(Some(Choice{ty:target,bytes,error_sse:diff_energy,nrmse}))
}

fn estimated_file_bytes(source:&Gguf, selected:&BTreeMap<String,QuantType>) -> Result<u64> {
    let (_, metadata)=source.preserved_metadata(None)?;
    let mut head=u64::try_from(24usize.checked_add(metadata.len()).context("header overflow")?)?;
    let mut data=0u64;
    for (name, tensor) in &source.tensors {
        let descriptor_len=8usize.checked_add(name.len())
            .and_then(|n|n.checked_add(4))
            .and_then(|n|n.checked_add(tensor.dims.len().checked_mul(8)?))
            .and_then(|n|n.checked_add(4+8)).context("descriptor overflow")?;
        head=head.checked_add(u64::try_from(descriptor_len)?).context("header overflow")?;
        data=align(data,source.alignment as u64)?;
        let ty=selected.get(name).copied().unwrap_or(tensor.ty);
        let (_,bytes)=rows_and_size(&source.tensor(name)?,ty)?;
        data=data.checked_add(bytes).context("candidate data size overflow")?;
    }
    align(head,source.alignment as u64)?
        .checked_add(data).context("estimated file length overflow")
}

fn current_formats(matrices:&[Matrix])->BTreeMap<String,QuantType>{
    matrices.iter().map(|m|(m.name.clone(),m.choices[m.selected].ty)).collect()
}
fn reduce(matrices:&mut [Matrix], original_bytes:u64, goal:u64) -> Result<(u64,usize)> {
    ensure!(goal>0,"requested file-size limit must be positive");
    // Each choice is smaller than the source, but reductions are ordered by
    // measured incremental squared-error cost per saved byte. This greedy
    // heuristic is deterministic; it does NOT solve an optimal knapsack.
    let mut bytes=original_bytes;
    let mut transitions=0usize;
    while bytes>goal {
        let mut best:Option<(usize,usize,f64,u64)>=None;
        for (i,m) in matrices.iter().enumerate() {
            let active=&m.choices[m.selected];
            for (j,choice) in m.choices.iter().enumerate() {
                if choice.bytes>=active.bytes {continue;}
                let saving=active.bytes-choice.bytes;
                let incremental_error=(choice.error_sse-active.error_sse).max(0.);
                let ratio=incremental_error/saving as f64;
                match best {
                    None=>best=Some((i,j,ratio,saving)),
                    Some((_,_,score,prev_saving)) if ratio<score || (ratio==score && saving>prev_saving)=>
                        best=Some((i,j,ratio,saving)),
                    _=>{}
                }
            }
        }
        let Some((i,j,_,saving))=best else {
            bail!("file-size goal cannot be reached with Q4_0/Q8_0 matrices passing the per-tensor reconstruction threshold; output was not created");
        };
        matrices[i].selected=j;
        bytes=bytes.checked_sub(saving).context("budget transition underflow")?;
        transitions+=1;
        ensure!(transitions<=matrices.len().saturating_mul(2), "quantization planner transition limit exceeded");
    }
    Ok((bytes,transitions))
}

pub(super) fn plan(source:&Gguf, goal_bytes:u64, max_nrmse:f64) -> Result<Plan> {
    ensure!(source.shard_paths().len()==1, "size-target planner only supports single-shard GGUF");
    ensure!(goal_bytes>0 && max_nrmse.is_finite() && max_nrmse>0. && max_nrmse<=1.,
        "invalid target size or reconstruction limit");
    let mut matrices=Vec::new();
    for (name,meta) in &source.tensors {
        if !eligible(name,&meta.dims,meta.ty,QuantType::Q4_0) {continue;}
        ensure!(matrices.len()<MAX_CHOICE_TENSORS,"too many matrices for bounded size-target planner");
        let tensor=source.tensor(name)?;
        let original_bytes=u64::try_from(meta.byte_len)?;
        let mut choices=vec![Choice{ty:meta.ty,bytes:original_bytes,error_sse:0.,nrmse:0.}];
        for ty in [QuantType::Q8_0,QuantType::Q4_0] {
            if let Some(candidate)=measure(&tensor,ty)? {
                if candidate.bytes<original_bytes && candidate.nrmse<=max_nrmse {
                    choices.push(candidate);
                }
            }
        }
        matrices.push(Matrix{name:name.clone(),choices,selected:0});
    }
    ensure!(!matrices.is_empty(),"no eligible floating-point matrices for size-target quantization");
    let baseline=estimated_file_bytes(source,&BTreeMap::new())?;
    ensure!(goal_bytes<baseline, "source file is already within the target size; no quantization is necessary");
    let (estimated_payload_bytes,transitions)=reduce(&mut matrices,baseline,goal_bytes)?;
    let formats=current_formats(&matrices);
    let exact_size=estimated_file_bytes(source,&formats)?;
    ensure!(exact_size<=goal_bytes, "aligned GGUF output would exceed requested file-size budget");
    let mut changed=0usize;
    let mut q4=0usize;
    let mut q8=0usize;
    let mut projected_errors=Vec::new();
    let mut accumulated_sse=0f64;
    for m in &matrices {
        let selection=&m.choices[m.selected];
        if m.selected==0 {continue;}
        changed+=1;
        accumulated_sse+=selection.error_sse;
        match selection.ty {QuantType::Q4_0=>q4+=1,QuantType::Q8_0=>q8+=1,_=>{}}
        projected_errors.push(json!({
            "tensor":m.name,
            "format":selection.ty.name(),
            "predicted_nrmse":selection.nrmse,
            "squared_error":selection.error_sse,
            "verified_in_output":false
        }));
    }
    ensure!(changed>0,"file-size planner did not select any conversion");
    Ok(Plan {
        formats,
        goal_bytes,
        estimated_file_bytes:exact_size,
        report:json!({
            "algorithm":"greedy_per_tensor_squared_error_per_saved_byte",
            "globally_optimal":false,
            "importance_matrix_used":false,
            "source_gguf_bytes":baseline,
            "requested_max_gguf_bytes":goal_bytes,
            "estimated_gguf_bytes":exact_size,
            "size_estimate_before_alignment_check":estimated_payload_bytes,
            "quantization_transitions":transitions,
            "converted_tensor_count":changed,
            "q4_0_tensors":q4,
            "q8_0_tensors":q8,
            "reconstruction_error_sse_sum":accumulated_sse,
            "tensor_choices":projected_errors,
            "quality_scope":"measured tensor-row reconstruction only; not coding, model logits, perplexity or other task accuracy",
            "no_existing_file_overwritten":true
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(ty:QuantType,bytes:u64,error:f64)->Choice {
        Choice{ty,bytes,error_sse:error,nrmse:error.sqrt()}
    }
    #[test]
    fn chooses_lower_error_cost_per_saved_byte_and_respects_goal() {
        let mut m=vec![
            Matrix{name:"a".into(),choices:vec![c(QuantType::F16,1000,0.),c(QuantType::Q8_0,600,2.),c(QuantType::Q4_0,300,50.)],selected:0},
            Matrix{name:"b".into(),choices:vec![c(QuantType::F16,1000,0.),c(QuantType::Q8_0,600,1.),c(QuantType::Q4_0,300,90.)],selected:0}
        ];
        let (bytes,transitions)=reduce(&mut m,2000,1600).unwrap();
        assert_eq!(bytes,1600);
        assert_eq!(transitions,1);
        assert_eq!(m[1].choices[m[1].selected].ty,QuantType::Q8_0);
    }
    #[test]
    fn impossible_budget_fails_without_approximating_quality_gate() {
        let mut m=vec![Matrix{name:"a".into(),
            choices:vec![c(QuantType::F16,1000,0.),c(QuantType::Q8_0,600,1.)],selected:0}];
        assert!(reduce(&mut m,1000,100).is_err());
    }
}
