//! Read-only model-fit and quantization size planning.
//!
//! No model conversion, implicit downloading, remote calls, GPU allocation,
//! quality claim, or automatic change to the runtime configuration is performed.
//! In particular GGUF file size, live VRAM use and output quality are distinct.
use crate::{gguf::{Gguf, TensorInfo}, quant::QuantType};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde_json::{Value, json};
use std::{path::PathBuf, collections::BTreeMap};

const MIB: u64 = 1024 * 1024;
const MATRIX_OPTIONS: [QuantType; 9] = [
    QuantType::F16, QuantType::Q8_0, QuantType::Q6_K, QuantType::Q5_K,
    QuantType::Q4_K, QuantType::Q4_0, QuantType::IQ4_XS,
    QuantType::IQ3_S, QuantType::IQ2_S,
];

#[derive(Args, Debug)]
pub struct ModelFitArgs {
    /// Local first GGUF shard of a complete model, no weights are loaded onto a GPU.
    #[arg(long)]
    pub model: PathBuf,
    /// Proposed context size per active request, for host memory planning.
    #[arg(long, default_value_t = 4096)]
    pub ctx: usize,
    /// Proposed active sequence count, 1..=4.
    #[arg(long, default_value_t = 1)]
    pub requests: usize,
    /// If known, GPU's nominal VRAM capacity in MiB (NOT available/free VRAM).
    #[arg(long)]
    pub target_vram_mib: Option<u64>,
    /// Optional desired GGUF file-size limit in MiB; distinct from VRAM.
    #[arg(long)]
    pub target_file_mib: Option<u64>,
    /// Reserve this percentage of nominal GPU heap for driver / other allocations.
    #[arg(long, default_value_t = 20)]
    pub gpu_headroom_percent: u8,
    /// OS/other-process RAM margin for host plan.
    #[arg(long, default_value_t = 4096)]
    pub ram_margin_mib: usize,
    /// Save a report; neither input GGUF nor its shards are overwritten.
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

fn eligible_for_size_estimate(info: &TensorInfo) -> bool {
    // Only dense floating-point weight matrices are *hypothetical* offline
    // re-quantization candidates. Existing quantized data has no available
    // full-precision original to compare against, so leave it unchanged.
    info.dims.len() >= 2
        && matches!(info.ty, QuantType::F32 | QuantType::F16 | QuantType::BF16)
        && info.name.ends_with(".weight")
        && !info.name.contains("token_embd")
        && !info.name.contains("output.weight")
        && !info.name.contains("norm")
        && !info.name.contains("rope")
        && !info.name.contains("pos_embd")
        && !info.name.contains(".scale")
}

fn projected_matrix_bytes(tensor: &TensorInfo, target: QuantType) -> Result<Option<u64>> {
    if !eligible_for_size_estimate(tensor) {
        return Ok(None);
    }
    let width = usize::try_from(tensor.dims[0]).context("model row width exceeds address space")?;
    if width == 0 || width % target.block_elements() != 0 {
        return Ok(None);
    }
    let count = tensor.dims[1..].iter().try_fold(1u64, |n, d| n.checked_mul(*d))
        .context("matrix row count overflow")?;
    let stride = u64::try_from(target.row_bytes(width)?).context("matrix row bytes overflow")?;
    Ok(Some(count.checked_mul(stride).context("candidate matrix size overflow")?))
}

fn estimate_candidates(tensors: &BTreeMap<String, TensorInfo>, model_bytes: u64, payload_bytes: u64, alignment: usize, desired: Option<u64>) -> Result<Value> {
    ensure!(model_bytes >= payload_bytes, "mapped size smaller than tensor payload");
    let fixed_overhead = model_bytes - payload_bytes;
    let mut candidates = vec![];
    let mut matrix_count=0u64;
    let mut mutable_payload=0u64;
    for tensor in tensors.values() {
        if eligible_for_size_estimate(tensor) {
            matrix_count += 1;
            mutable_payload=mutable_payload.checked_add(u64::try_from(tensor.byte_len)?)
                .context("matrix payload overflow")?;
        }
    }
    let static_payload = payload_bytes.checked_sub(mutable_payload).context("payload underflow")?;
    let padding_allowance = (tensors.len() as u64)
        .checked_mul(u64::try_from(alignment)?)
        .context("alignment allowance overflow")?;
    for target in MATRIX_OPTIONS {
        let mut hypothetical_payload=static_payload;
        let mut eligible=0u64;
        let mut unconverted=0u64;
        for tensor in tensors.values() {
            match projected_matrix_bytes(tensor,target)? {
                Some(bytes) => {
                    hypothetical_payload=hypothetical_payload.checked_add(bytes).context("candidate payload overflow")?;
                    eligible += 1;
                }
                None if eligible_for_size_estimate(tensor) => {
                    hypothetical_payload=hypothetical_payload.checked_add(u64::try_from(tensor.byte_len)?).context("candidate payload overflow")?;
                    unconverted += 1;
                }
                None => {}
            }
        }
        let estimate = hypothetical_payload.checked_add(fixed_overhead)
            .context("estimated output size overflow")?;
        let ceiling = estimate.checked_add(padding_allowance).context("output padding allowance overflow")?;
        candidates.push(json!({
            "format":target.name(),
            "estimated_gguf_bytes":estimate,
            "conservative_size_ceiling_bytes":ceiling,
            "candidate_matrix_count":eligible,
            "unchanged_ineligible_matrix_count":unconverted,
            "meets_file_size_ceiling": desired.map(|n| ceiling <= n),
            "converted_weights_written":false,
            "quality_verified":false
        }));
    }
    // A file-size target is not a VRAM fit check: mixed architectures, lazy
    // tensors, KV/cache, staging and offload are not determined by this value.
    let viable: Vec<&Value> = candidates.iter()
        .filter(|v|v["candidate_matrix_count"].as_u64().unwrap_or(0)>0
            && v["meets_file_size_ceiling"].as_bool() == Some(true))
        .collect();
    let closest_to_target=viable.into_iter()
        .max_by_key(|v| v["conservative_size_ceiling_bytes"].as_u64().unwrap_or(0))
        .and_then(|v|v["format"].as_str()).map(str::to_owned);
    Ok(json!({
        "current_mapped_file_bytes": model_bytes,
        "current_encoded_tensor_payload_bytes":payload_bytes,
        "fixed_metadata_alignment_and_shard_overhead_bytes":fixed_overhead,
        "hypothetically_requantizable_float_matrix_count":matrix_count,
        "hypothetically_requantizable_float_matrix_bytes":mutable_payload,
        "nonconvertible_payload_bytes":static_payload,
        "padding_uncertainty_allowance_bytes":padding_allowance,
        "target_file_bytes":desired,
        "largest_estimated_candidate_within_target": closest_to_target,
        "candidates":candidates,
        "warning":"These are layout-only estimates from existing tensor dimensions. No GGUF is created. No importance matrix, quantization-error evaluation or CPU/GPU correctness check has been performed. Re-quantizing an already quantized tensor is not proposed. Actual GGUF headers, padding and quantization rules may differ."
    }))
}

fn target_gpu_budget(gpu_mib: Option<u64>, headroom_percent:u8) -> Result<Value> {
    let cap = gpu_mib.map(|x|x.checked_mul(MIB).context("target VRAM bytes overflow")).transpose()?;
    let safe = cap.map(|n| n / 100 * u64::from(100-headroom_percent));
    Ok(json!({
        "nominal_capacity_bytes": cap,
        "reserve_percent":headroom_percent,
        "advisory_nextora_owned_allocation_ceiling_bytes":safe,
        "live_free_vram_known":false,
        "model_gpu_fit_verified":false,
        "scope":"headroom from nominal capacity; not live free VRAM and not the amount available to this process"
    }))
}

pub fn evaluate(args: &ModelFitArgs) -> Result<Value> {
    ensure!((128..=1048576).contains(&args.ctx), "ctx must be 128..=1048576");
    ensure!((1..=4).contains(&args.requests), "requests must be 1..=4");
    ensure!((5..=50).contains(&args.gpu_headroom_percent), "GPU headroom percentage must be in 5..=50");
    ensure!(args.target_vram_mib.is_none_or(|x| x>0), "target VRAM must be positive");
    ensure!(args.target_file_mib.is_none_or(|x| x>0), "target GGUF file size must be positive");
    let model=Gguf::open(&args.model)
        .context("parse a complete supported GGUF shard set; unsupported quantization formats are not silently converted")?;
    let payload=u64::try_from(model.tensor_bytes())?;
    let mapped=u64::try_from(model.mapped_bytes())?;
    let desired=args.target_file_mib.map(|n|n.checked_mul(MIB).context("target file size overflow")).transpose()?;
    let arch=model.meta.get("general.architecture").and_then(|v|v.as_str()).unwrap_or("unknown");
    let native_arch=matches!(arch, "qwen3"|"qwen3moe"|"qwen35"|"qwen4exp");
    let theoretical=estimate_candidates(&model.tensors,mapped,payload,model.alignment,desired)?;
    let hardware=crate::runtime::hardware_report();
    let eligible=hardware["adapters"].as_array().into_iter().flatten()
        .filter(|a|a["compatible"].as_bool()==Some(true) && a["software"].as_bool()==Some(false))
        .filter_map(|a|a["largest_usable_device_local_heap_bytes"].as_u64())
        .max();
    let nominal_mib=args.target_vram_mib.or(eligible.map(|n|n/MIB));
    // Host planner is an arithmetic bound for the current architecture, not
    // the same as GPU residency. Return failures as evidence rather than
    // turning any failure into a "needs more RAM" shopping recommendation.
    let host_plan = crate::memory::plan(&model,args.ctx,args.requests,true,Some(0),args.ram_margin_mib)
        .map(|p|json!({"status":"planned","details":p}))
        .unwrap_or_else(|e|json!({"status":"not_available","reason":format!("{e:#}")}));
    let gpu_plan = crate::memory::plan(&model,args.ctx,args.requests,false,Some(0),args.ram_margin_mib)
        .map(|p|json!({"status":"planned","details":p}))
        .unwrap_or_else(|e|json!({"status":"not_available","reason":format!("{e:#}")}));
    let gpu_budget=target_gpu_budget(nominal_mib,args.gpu_headroom_percent)?;
    let classification=if !native_arch {
        "unsupported_native_architecture"
    } else if gpu_budget["nominal_capacity_bytes"].is_null() {
        "capacity_unknown_benchmark_required"
    } else {
        "layout_inspected_execution_unverified"
    };
    Ok(json!({
        "schema":"nextora.model_fit.v1",
        "engine":crate::ENGINE_ID,
        "version":crate::ENGINE_VERSION,
        "model_file_name":args.model.file_name().map(|n|n.to_string_lossy().to_string()),
        "model_path_exported":false,
        "architecture":arch,
        "native_text_architecture_supported":native_arch,
        "status":classification,
        "requested_context_tokens":args.ctx,
        "active_request_slots":args.requests,
        "gpu":gpu_budget,
        "host_cpu_state_reserve_plan":host_plan,
        "host_gpu_state_reserve_plan":gpu_plan,
        "hypothetical_file_quantizations":theoretical,
        "recommendations":[
            "Do not buy hardware or claim that a GGUF will fit GPU memory based on mapped-file bytes alone",
            "For 16 GiB consumer GPUs, reserve VRAM for activations, KV/recurrent state and OS/driver activity; measure actual allocations on the target GPU",
            "Use the existing Nextora bench and Insight dashboard to compare context and concurrency at fixed model revision and driver",
            "When a file-size limit is needed, quantize offline using a validated encoder and importance matrix; verify model quality after conversion",
            "Looped Inkling models need loop scheduling, per-pass attention/recurrent state and independent CPU/Vulkan parity before native support can be claimed"
        ],
        "limitations":[
            "This command does not quantize weights, download files, load a model, allocate VRAM or generate text",
            "Quantization candidates are tensor-layout size estimates, not optimal error-minimising recipes or measured quality",
            "Expert-cache sizing and GPU state placement are runtime/architecture-dependent",
            "Vulkan heap size is nominal and cannot establish presently free memory, driver compatibility or stable compute",
            "Live model quality, latency and tokens/second require tests on the target device"
        ]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(name:&str, ty:QuantType, dims:&[u64]) -> TensorInfo {
        let width=dims[0] as usize;
        let rows=dims[1..].iter().product::<u64>() as usize;
        TensorInfo {name:name.into(),dims:dims.to_vec(),ty,offset:0,
            byte_len:ty.row_bytes(width).unwrap()*rows,
            file_offset:0,shard_index:0}
    }
    #[test]
    fn matrix_size_candidates_are_exact_for_block_geometry() {
        let t=tensor("blk.0.ffn_up.weight",QuantType::F16,&[256,256]);
        assert_eq!(projected_matrix_bytes(&t,QuantType::Q4_K).unwrap(),Some(144*256));
        assert_eq!(projected_matrix_bytes(&t,QuantType::Q8_0).unwrap(),Some(34*8*256));
        assert_eq!(projected_matrix_bytes(&t,QuantType::IQ3_S).unwrap(),Some(110*256));
    }
    #[test]
    fn ineligible_or_prequantized_weights_are_kept_unchanged() {
        let output=tensor("output.weight",QuantType::F16,&[256,16]);
        let prequant=tensor("blk.0.ffn_up.weight",QuantType::Q4_K,&[256,256]);
        assert_eq!(projected_matrix_bytes(&output,QuantType::Q4_K).unwrap(),None);
        assert_eq!(projected_matrix_bytes(&prequant,QuantType::Q4_K).unwrap(),None);
    }
    #[test]
    fn nondivisible_rows_do_not_get_fake_quantization_sizes() {
        let t=tensor("blk.0.ffn_up.weight",QuantType::F16,&[96,2]);
        assert_eq!(projected_matrix_bytes(&t,QuantType::Q4_K).unwrap(),None);
    }
    #[test]
    fn target_file_budget_is_not_reported_as_gpu_fit_or_quality() {
        let mut tensors=BTreeMap::new();
        let t=tensor("blk.0.ffn_up.weight",QuantType::F16,&[256,256]);
        tensors.insert(t.name.clone(),t);
        let estimates=estimate_candidates(&tensors,132000,131072,32,Some(70000)).unwrap();
        assert!(estimates["largest_estimated_candidate_within_target"].as_str().is_some());
        assert!(estimates["candidates"].as_array().unwrap().iter().all(|r|r["quality_verified"]==false));
        let device=target_gpu_budget(Some(16*1024),20).unwrap();
        assert_eq!(device["model_gpu_fit_verified"],false);
        assert!(device["advisory_nextora_owned_allocation_ceiling_bytes"].as_u64().unwrap() < 16*1024*MIB);
    }
    #[test]
    fn overflow_never_creates_fictitious_size_or_gpu_budget() {
        assert!(target_gpu_budget(Some(u64::MAX),20).is_err());
    }
}
