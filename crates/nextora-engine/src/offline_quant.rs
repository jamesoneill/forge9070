//! Non-destructive, bounded-memory, single-shard GGUF offline quantization.
//! This is NOT the importance-aware variable-bit optimizer planned in issue #19.
//! Only validated dense floating-point matrix rows can be encoded to Q4_0 or Q8_0.
//! All other tensors retain their original bytes. The output is independently
//! parsed before being published, and no existing path is overwritten.
use crate::{gguf::Gguf, quant::QuantType};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use half::f16;
use serde_json::{Value, json};
#[path = "quant_budget.rs"]
mod budget;
use std::{
    fs::{self, OpenOptions, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Target {
    Q4_0,
    Q8_0,
}
impl Target {
    fn quant(self) -> QuantType {
        match self { Self::Q4_0 => QuantType::Q4_0, Self::Q8_0 => QuantType::Q8_0 }
    }
    fn gguf_file_type(self) -> u32 {
        // ggml ftype enum: MOSTLY_Q4_0=2, MOSTLY_Q8_0=7
        match self { Self::Q4_0 => 2, Self::Q8_0 => 7 }
    }
}

#[derive(Args, Debug)]
pub struct OfflineQuantArgs {
    /// Existing, complete single-file GGUF containing F16/F32/BF16 weights.
    #[arg(long)]
    pub model: PathBuf,
    /// Output path must not already exist (including symlinks).
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, value_enum)]
    pub target: Option<Target>,
    /// Alternative to --target: select Q4_0/Q8_0 per tensor to meet a maximum output file size.
    /// The planner checks per-tensor reconstruction error; this is NOT a model-quality guarantee.
    #[arg(long)]
    pub target_size_mib: Option<u64>,
    /// Maximum normalized RMS reconstruction error per converted tensor.
    #[arg(long, default_value_t = 0.25)]
    pub max_nrmse: f64,
}

fn eligible(name: &str, dims: &[u64], source: QuantType, target: QuantType) -> bool {
    matches!(source, QuantType::F16 | QuantType::F32 | QuantType::BF16)
        && dims.len() >= 2
        && dims[0] > 0
        && usize::try_from(dims[0]).ok().is_some_and(|n| n % target.block_elements() == 0)
        && name.ends_with(".weight")
        && !name.contains("token_embd")
        && !name.contains("output.weight")
        && !name.contains("norm")
        && !name.contains("rope")
        && !name.contains("pos_embd")
        && !name.contains(".scale")
}

fn align(n:u64, width:u64)->Result<u64>{
    ensure!(width>0, "zero alignment");
    let remaining=n%width;
    if remaining==0 {Ok(n)} else { n.checked_add(width-remaining).context("alignment overflow") }
}

fn write_string(out: &mut Vec<u8>, value:&str)->Result<()> {
    let length=u64::try_from(value.len())?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

#[derive(Clone)]
struct Descriptor {
    name:String,
    dims:Vec<u64>,
    format:QuantType,
    offset:u64,
    bytes:u64,
    converted:bool,
}

/// GGUF row encoders with the public Q4_0 signed-maximum and Q8_0 scale
/// convention. A quantized row is decoded using *our existing runtime decoder*
/// before a tensor can pass its finite reconstruction-error gate.
fn encode_row(input:&[f32], target:QuantType, output:&mut Vec<u8>)->Result<()> {
    ensure!(matches!(target, QuantType::Q4_0|QuantType::Q8_0), "unsupported encoder");
    ensure!(!input.is_empty() && input.len()%32==0, "quantization requires complete 32-element blocks");
    ensure!(input.iter().all(|v|v.is_finite()), "non-finite source weights");
    output.clear();
    output.try_reserve_exact(target.row_bytes(input.len())?)?;
    for values in input.chunks_exact(32) {
        match target {
            QuantType::Q8_0 => {
                let maximum=values.iter().copied().map(f32::abs).fold(0f32,f32::max);
                let scale=if maximum==0. {0.} else {maximum/127.};
                let rounded=f16::from_f32(scale).to_f32();
                ensure!(maximum==0. || rounded>0. && rounded.is_finite(), "Q8_0 scale underflow/overflow");
                output.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
                for &v in values {
                    let q=if maximum==0. {0i8}
                        else {((v/rounded).round() as i32).clamp(-128,127) as i8};
                    output.push(q as u8);
                }
            }
            QuantType::Q4_0 => {
                let signed_max=values.iter().copied()
                    .max_by(|a,b|a.abs().total_cmp(&b.abs()))
                    .context("empty Q4 block")?;
                let scale=if signed_max==0. {0.} else {-signed_max/8.};
                let rounded=f16::from_f32(scale).to_f32();
                ensure!(signed_max==0. || rounded!=0. && rounded.is_finite(), "Q4_0 scale underflow/overflow");
                output.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
                for i in 0..16 {
                    let nibble=|v:f32| ->u8 {
                        if signed_max==0. {8} else {((v/rounded+8.).round() as i32).clamp(0,15) as u8}
                    };
                    output.push(nibble(values[i]) | (nibble(values[i+16])<<4));
                }
            }
            _ => unreachable!()
        }
    }
    ensure!(output.len()==target.row_bytes(input.len())?, "quantized row length disagrees");
    Ok(())
}

fn copy_exact(out:&mut impl Write, bytes:&[u8])->Result<()> {
    // The source is memory mapped; this bounded slice writes without making
    // another whole-tensor allocation.
    out.write_all(bytes)?;
    Ok(())
}
fn write_padding(out:&mut impl Write, count:u64)->Result<()> {
    let pad=[0u8;4096];
    let mut n=count;
    while n>0 {
        let size=usize::try_from(n.min(pad.len() as u64))?;
        out.write_all(&pad[..size])?;
        n-=size as u64;
    }
    Ok(())
}
fn same_existing(input:&Path, output:&Path) -> Result<bool> {
    if !output.exists() {return Ok(false);}
    Ok(same_file::is_same_file(input,output)?)
}
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self){let _=fs::remove_file(&self.0);}
}
fn temporary_path(output:&Path)->Result<(Temporary,File)> {
    let parent=output.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    ensure!(parent.is_dir(), "output parent directory does not exist");
    let base=output.file_name().context("output must name a file")?.to_string_lossy();
    let tick=SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    for i in 0..32 {
        let candidate=parent.join(format!(".{base}.nextora-quant-{}-{tick}-{i}.tmp",std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file)=>return Ok((Temporary(candidate),file)),
            Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists => continue,
            Err(error)=>return Err(error.into()),
        }
    }
    bail!("unable to allocate a unique quantizer temporary file")
}

pub fn quantize(args:&OfflineQuantArgs)->Result<Value> {
    ensure!(args.max_nrmse.is_finite() && args.max_nrmse>0. && args.max_nrmse<=1.,
        "max-nrmse must be a finite value in (0,1]");
    ensure!(args.target.is_some() != args.target_size_mib.is_some(),
        "select exactly one of --target or --target-size-mib");
    ensure!(args.target_size_mib.is_none_or(|m|m>0),
        "--target-size-mib must be positive");

    ensure!(!args.output.exists(), "output already exists; overwriting is forbidden");
    // A dangling symlink also counts as an existing destination.
    ensure!(fs::symlink_metadata(&args.output).is_err(), "output path already exists");
    ensure!(!same_existing(&args.model,&args.output)?, "output must not overwrite input");
    let source=Gguf::open(&args.model)?;
    ensure!(!crate::prism::declared(&source.meta), "offline quantization of folded Prism checkpoints is unsupported");
    ensure!(source.shard_paths().len()==1,
        "offline quantizer currently requires a single-file GGUF; split checkpoints are not supported");
    let static_target=args.target.map(Target::quant);
    let target_bytes=args.target_size_mib.map(|m|m.checked_mul(1024*1024)
        .context("target-size-mib exceeds supported file size")).transpose()?;
    let selected_plan=if let Some(bytes)=target_bytes {
        Some(budget::plan(&source,bytes,args.max_nrmse)?)
    } else {None};
    let file_type=if let Some(plan)=&selected_plan {
        let q4=plan.formats.values().filter(|&&ty|ty==QuantType::Q4_0).count();
        let q8=plan.formats.values().filter(|&&ty|ty==QuantType::Q8_0).count();
        if q4>=q8 {2u32} else {7u32}
    } else {
        args.target.context("explicit format is missing")?.gguf_file_type()
    };
    let original_type=source.meta.get("general.file_type");
    if let Some(value)=original_type {
        ensure!(matches!(value,crate::gguf::Value::U32(_)),
            "general.file_type must be uint32 to change quantization safely");
    }
    let (metadata_count,metadata)=source.preserved_metadata(Some(file_type))?;
    let mut header=Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&source.version.to_le_bytes());
    header.extend_from_slice(&(source.tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&metadata_count.to_le_bytes());
    header.extend_from_slice(&metadata);
    let mut descriptors=Vec::with_capacity(source.tensors.len());
    let mut current=0u64;
    let mut converted_count=0usize;
    for (name,tensor) in &source.tensors {
        let format=if let Some(plan)=&selected_plan {
            plan.formats.get(name).copied().unwrap_or(tensor.ty)
        } else {
            let goal=static_target.context("quantization format not selected")?;
            if eligible(name,&tensor.dims,tensor.ty,goal) {goal} else {tensor.ty}
        };
        let converted=format!=tensor.ty;
        let rows=tensor.dims.iter().skip(1).try_fold(1u64,|n,d|n.checked_mul(*d)).context("row count overflow")?;
        let stride=u64::try_from(format.row_bytes(usize::try_from(tensor.dims[0])?)?)?;
        let bytes=rows.checked_mul(stride).context("output tensor size overflow")?;
        current=align(current,source.alignment as u64)?;
        descriptors.push(Descriptor{name:name.clone(),dims:tensor.dims.clone(),format,offset:current,bytes,converted});
        current=current.checked_add(bytes).context("output model size overflow")?;
        if converted {converted_count+=1;}
        write_string(&mut header,name)?;
        header.extend_from_slice(&u32::try_from(tensor.dims.len())?.to_le_bytes());
        for &dim in &tensor.dims {header.extend_from_slice(&dim.to_le_bytes());}
        header.extend_from_slice(&format.id().to_le_bytes());
        header.extend_from_slice(&descriptors.last().unwrap().offset.to_le_bytes());
    }
    ensure!(converted_count>0,"no eligible floating-point matrices to convert; already quantized sources cannot be requantized");
    let start_data=align(u64::try_from(header.len())?,source.alignment as u64)?;
    let expected_length=start_data.checked_add(current).context("output size overflow")?;
    if let Some(plan)=&selected_plan {
        ensure!(expected_length==plan.estimated_file_bytes,
            "quantization plan and output GGUF descriptor size disagree");
        ensure!(expected_length<=plan.goal_bytes,
            "quantized GGUF would exceed requested output file-size budget");
    }
    let (temp,file)=temporary_path(&args.output)?;
    let mut output=BufWriter::with_capacity(1024*1024,file);
    output.write_all(&header)?;
    write_padding(&mut output,start_data-u64::try_from(header.len())?)?;
    let mut cursor=0u64;
    let mut error_metrics=vec![];
    for d in &descriptors {
        ensure!(d.offset>=cursor,"output tensor offsets are not increasing");
        write_padding(&mut output,d.offset-cursor)?;
        cursor=d.offset;
        let tensor=source.tensor(&d.name)?;
        if !d.converted {
            ensure!(d.bytes==u64::try_from(tensor.bytes().len())?,"unchanged tensor byte count differs");
            copy_exact(&mut output,tensor.bytes())?;
        } else {
            let width=usize::try_from(d.dims[0])?;
            ensure!(width<=1_048_576,"quantizer row width exceeds 1048576 element limit");
            let rows=usize::try_from(d.dims[1..].iter().try_fold(1u64,|n,v|n.checked_mul(*v)).context("rows overflow")?)?;
            let mut values=vec![0f32;width];
            let mut decoded=vec![0f32;width];
            let mut packed=Vec::new();
            let mut original_sum=0f64;
            let mut error_sum=0f64;
            let mut peak_error=0f32;
            for row in 0..rows {
                tensor.read_row(row,&mut values)?;
                encode_row(&values,d.format,&mut packed)?;
                crate::quant::decode_row(d.format,&packed,&mut decoded)?;
                for (&a,&b) in values.iter().zip(&decoded) {
                    let delta=(a-b).abs();
                    peak_error=peak_error.max(delta);
                    error_sum+=(delta as f64).powi(2);
                    original_sum+=(a as f64).powi(2);
                }
                output.write_all(&packed)?;
            }
            let normalised=if original_sum==0. {
                if error_sum==0. {0.} else {f64::INFINITY}
            } else {(error_sum/original_sum).sqrt()};
            ensure!(normalised.is_finite() && normalised<=args.max_nrmse,
                "tensor {} reconstruction NRMSE {:.6} exceeded limit {:.6}; output discarded",
                d.name,normalised,args.max_nrmse);
            error_metrics.push(json!({
                "name":d.name, "source_format":tensor.quant_type().name(),
                "destination_format":d.format.name(), "rows":rows, "nrmse":normalised,
                "max_abs_error":peak_error, "accepted":true
            }));
        }
        cursor=cursor.checked_add(d.bytes).context("output offset overflow")?;
    }
    output.flush()?;
    let file=output.into_inner().map_err(|e|e.into_error())?;
    file.sync_all()?;
    drop(file);
    let measured=fs::metadata(&temp.0)?.len();
    ensure!(measured==expected_length,"written GGUF length {} differs from expected {}",measured,expected_length);
    let verify=Gguf::open(&temp.0).context("round-trip parse converted GGUF")?;
    ensure!(verify.tensors.len()==source.tensors.len(),"output tensor count differs");
    for d in &descriptors {
        let v=verify.tensors.get(&d.name).context("converted GGUF omitted tensor")?;
        ensure!(v.dims==d.dims && v.ty==d.format && v.byte_len as u64==d.bytes,
            "GGUF tensor geometry or format differs after round-trip");
    }
    ensure!(verify.version==source.version && verify.meta.len()==source.meta.len(),"GGUF metadata changed unexpectedly");
    drop(verify);
    drop(source);
    // Hard-link publish is create-new: it fails if the destination already
    // exists, and never replaces a user's model or a symlink.
    fs::hard_link(&temp.0,&args.output).context("publish validated output without overwriting")?;
    Ok(json!({
        "schema":"nextora.quantize.v1",
        "target_format":static_target.map(|ty|ty.name()).unwrap_or("MIXED_Q4_0_Q8_0"),
        "size_target_bytes":target_bytes,
        "size_target_met":target_bytes.map(|maximum|measured<=maximum),
        "mixed_precision_plan":selected_plan.as_ref().map(|p|p.report.clone()),
        "source_file_name":args.model.file_name().map(|s|s.to_string_lossy().to_string()),
        "output_file_name":args.output.file_name().map(|s|s.to_string_lossy().to_string()),
        "output_size_bytes":measured,"converted_tensors":converted_count,
        "retained_tensor_count":descriptors.len()-converted_count,
        "quality_gate":"per-tensor normalized RMS reconstruction error; not a model-level quality or coding task evaluation",
        "max_nrmse_allowed":args.max_nrmse,"tensor_errors":error_metrics,
        "validated_gguf_roundtrip":true,"source_overwritten":false,
        "model_quality_verified":false,"hardware_performance_verified":false,
        "limitations":[
            "Single-shard GGUF only; metadata types and unchanged tensors retained verbatim",
            "No importance matrix, calibration, globally optimal bit allocation or model-level quality parity",
            "Successful tensor reconstruction checks do not establish a supported architecture, valid tokenizer or stable execution",
            "CPU-only offline conversion; no driver changes, GPU allocation or external calls"
        ]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_zero_blocks_encode_and_decode_without_non_finite_values() {
        for target in [QuantType::Q4_0,QuantType::Q8_0] {
            let mut packed=vec![];
            encode_row(&[0.;32],target,&mut packed).unwrap();
            let mut decoded=[f32::NAN;32];
            crate::quant::decode_row(target,&packed,&mut decoded).unwrap();
            assert_eq!(decoded,[0.;32]);
        }
    }
    #[test]
    fn sample_rows_encode_and_roundtrip_with_bounded_error() {
        let row=(0..64).map(|n|(n as f32-30.)/40.).collect::<Vec<_>>();
        for target in [QuantType::Q4_0,QuantType::Q8_0] {
            let mut packed=vec![];
            encode_row(&row,target,&mut packed).unwrap();
            let mut decoded=vec![0.;64];
            crate::quant::decode_row(target,&packed,&mut decoded).unwrap();
            let error=row.iter().zip(decoded.iter()).map(|(a,b)|(a-b).abs()).fold(0.,f32::max);
            assert!(error<if target==QuantType::Q8_0 {0.02} else {0.2});
        }
    }

    #[test]
    fn writes_parseable_new_gguf_and_never_overwrites_source() {
        use std::io::Write;
        let base=std::env::temp_dir().join(format!("nextora-quant-test-{}-{}",
            std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&base).unwrap();
        let input=base.join("input.gguf");
        let output=base.join("output.gguf");
        let mut data=Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        data.extend_from_slice(&2u64.to_le_bytes()); // metadata count
        write_string(&mut data,"general.architecture").unwrap();
        data.extend_from_slice(&8u32.to_le_bytes()); // string type
        write_string(&mut data,"qwen3").unwrap();
        write_string(&mut data,"general.file_type").unwrap();
        data.extend_from_slice(&4u32.to_le_bytes()); // u32
        data.extend_from_slice(&1u32.to_le_bytes()); // F16
        write_string(&mut data,"blk.0.ffn_up.weight").unwrap();
        data.extend_from_slice(&2u32.to_le_bytes()); // 2D
        data.extend_from_slice(&256u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // F16
        data.extend_from_slice(&0u64.to_le_bytes()); // data offset
        let offset=align(data.len() as u64,32).unwrap() as usize;
        data.resize(offset,0);
        for i in 0..256 {
            data.extend_from_slice(&f16::from_f32((i as f32-128.)/128.).to_le_bytes());
        }
        let mut source=File::create(&input).unwrap();
        source.write_all(&data).unwrap();source.sync_all().unwrap();drop(source);
        let initial=fs::read(&input).unwrap();
        for target in [Target::Q4_0, Target::Q8_0] {
            let args=OfflineQuantArgs {model:input.clone(),output:output.clone(),target:Some(target),target_size_mib:None,max_nrmse:0.25};
            let report=quantize(&args).unwrap();
            assert_eq!(report["converted_tensors"],1);
            assert_eq!(report["validated_gguf_roundtrip"],true);
            let parsed=Gguf::open(&output).unwrap();
            assert_eq!(parsed.tensors["blk.0.ffn_up.weight"].ty,target.quant());
            assert_eq!(parsed.meta["general.file_type"].as_u64(),Some(target.gguf_file_type() as u64));
            assert_eq!(fs::read(&input).unwrap(),initial);
            let original_output=fs::read(&output).unwrap();
            assert!(quantize(&args).is_err());
            assert_eq!(fs::read(&output).unwrap(),original_output);
            drop(parsed);
            fs::remove_file(&output).unwrap();
        }
        fs::remove_file(input).unwrap();
        fs::remove_dir(base).unwrap();
    }


    #[test]
    fn target_file_size_chooses_safe_quantization_and_fails_impossible_budget() {
        use std::io::Write;
        let base=std::env::temp_dir().join(format!("nextora-budget-test-{}-{}",
            std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&base).unwrap();
        let input=base.join("source.gguf");
        let output=base.join("one-mib.gguf");
        let mut data=Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes());
        write_string(&mut data,"general.architecture").unwrap();
        data.extend_from_slice(&8u32.to_le_bytes());
        write_string(&mut data,"qwen3").unwrap();
        write_string(&mut data,"general.file_type").unwrap();
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        write_string(&mut data,"blk.0.ffn_up.weight").unwrap();
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&256u64.to_le_bytes());
        data.extend_from_slice(&4096u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let offset=align(data.len() as u64,32).unwrap() as usize;
        data.resize(offset,0);
        let mut f=File::create(&input).unwrap();
        f.write_all(&data).unwrap();
        for _ in 0..4096 {
            for i in 0..256 {
                f.write_all(&f16::from_f32((i as f32-128.)/128.).to_le_bytes()).unwrap();
            }
        }
        f.sync_all().unwrap();
        drop(f);
        let original_length=fs::metadata(&input).unwrap().len();
        let args=OfflineQuantArgs {
            model:input.clone(),output:output.clone(),
            target:None,target_size_mib:Some(1),max_nrmse:0.25
        };
        let report=quantize(&args).unwrap();
        assert_eq!(report["size_target_met"],true);
        assert!(report["output_size_bytes"].as_u64().unwrap()<=1024*1024);
        assert_eq!(report["mixed_precision_plan"]["importance_matrix_used"],false);
        assert_eq!(report["model_quality_verified"],false);
        assert!(fs::metadata(&input).unwrap().len()==original_length);
        let actual=Gguf::open(&output).unwrap();
        assert_eq!(actual.tensors["blk.0.ffn_up.weight"].ty,QuantType::Q4_0);
        drop(actual);
        fs::remove_file(&output).unwrap();
        let mut impossible=args;
        impossible.max_nrmse=0.000001;
        assert!(quantize(&impossible).is_err());
        assert!(!output.exists());
        fs::remove_file(input).unwrap();
        fs::remove_dir(base).unwrap();
    }

    #[test]
    fn unaligned_or_nonfinite_source_is_rejected() {
        assert!(encode_row(&[f32::NAN;32],QuantType::Q4_0,&mut vec![]).is_err());
        assert!(encode_row(&[0.;31],QuantType::Q8_0,&mut vec![]).is_err());
    }
}
