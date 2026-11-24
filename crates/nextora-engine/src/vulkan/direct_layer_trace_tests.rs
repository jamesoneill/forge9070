//! Ignored same-input diagnostics. No production hooks, model downloads or
//! speed claims: operation readbacks intentionally serialize execution.
use super::*;
use super::super::context::{DiagnosticPipelineMode, with_diagnostic_pipeline};

#[derive(Clone)]
struct Point { key:String, meta:Value, raw:Vec<u8>, integer:bool }
struct Slice { name:&'static str, buffer:Arc<Buffer>, regions:Vec<(usize,usize)>, integer:bool }
fn one(name:&'static str,buffer:&Arc<Buffer>,offset:usize,count:usize,integer:bool)->Slice {
    Slice{name,buffer:Arc::clone(buffer),regions:vec![(offset,count)],integer}
}
fn outputs(op:&GraphOp,position:usize)->Vec<Slice> {
    let p=&op.operation.push.0;let b=&op.operation.buffers;
    let n=|i:usize|p[i+1]as usize;
    match p[0] {
        0=>vec![one("linear",&b[3],p[12]as usize,n(1)*if p[15]&1!=0 {op.operation.groups[2]as usize}else{1},false)],
        1=>vec![one("rms_norm",&b[3],0,n(0)*n(1),false)],
        2=>vec![one("rope",&b[3],0,n(0)*n(1),false)],
        3=>{assert_eq!(p[15]&64,0,"this diagnostic traces F32 KV only");vec![one("key_row",&b[3],position*n(0),n(0),false),one("value_row",&b[4],position*n(0),n(0),false)]},
        4=>vec![Slice{name:"attention_scores",buffer:Arc::clone(&b[3]),regions:(0..n(1)).map(|head|(head*n(4),position+1)).collect(),integer:false}],
        5=>vec![one("attention",&b[3],0,n(1)*n(3),false)],
        6=>vec![one("swiglu",&b[3],p[12]as usize,n(0),false)],
        7=>vec![one("residual",&b[3],0,n(0),false)],
        8=>vec![one("router_probabilities",&b[3],0,n(0),false),one("selected_experts_and_weights",&b[5],0,n(1)*2,false)],
        9=>vec![one("embedding",&b[3],0,n(0),false)],
        10=>vec![one("weighted_expert_merge",&b[3],0,n(0),false)],
        16=>vec![one("q8_words",&b[2],p[11]as usize,n(0).div_ceil(32)*8,true),one("q8_scales",&b[4],p[13]as usize,n(0).div_ceil(32),false)],
        value=>panic!("unmapped diagnostic opcode {value}"),
    }
}
fn label(model:&GpuModel,op:&GraphOp)->String {
    if let Some((name,_))=model.weights.iter().find(|(_,weight)|Arc::ptr_eq(weight,&op.operation.buffers[0])) {return name.clone()}
    match op.operation.push.0[0] {2=>"rope".into(),3=>"kv_append".into(),4=>"attention_scores".into(),5=>"attention_values".into(),6=>"swiglu".into(),7=>"residual".into(),8=>"router_topk".into(),9=>"token_embedding".into(),10=>"expert_merge".into(),16=>"activation_quantization".into(),_=>"expert_linear".into()}
}
fn record_one(model:&GpuModel,commands:&mut Commands,op:&GraphOp,token:u32,position:usize,
    already_begun:bool,host:&Arc<Buffer>,key:&str,points:&mut Vec<Point>)->Result<()> {
    if !already_begun {commands.begin()?;}
    GpuModel::record_range(commands,std::slice::from_ref(op),token,position);
    commands.compute_to_transfer();
    let slices=outputs(op,position);let mut offset=0usize;let mut spans=Vec::new();
    for slice in slices {
        let begin=offset;
        for (start,count) in &slice.regions {
            let bytes=count.checked_mul(4).context("trace range overflow")?;
            ensure!((start+count)*4<=slice.buffer.size as usize,"trace source range outside allocation");
            ensure!(offset+bytes+4<=host.size as usize,"trace readback capacity exceeded");
            unsafe {model.device.raw.cmd_copy_buffer(commands.raw,slice.buffer.raw,host.raw,
                &[vk::BufferCopy{src_offset:(start*4)as u64,dst_offset:offset as u64,size:bytes as u64}]);}
            offset+=bytes;
        }
        spans.push((slice.name,begin,offset-begin,slice.integer));
    }
    unsafe {model.device.raw.cmd_copy_buffer(commands.raw,op.operation.buffers[6].raw,host.raw,
        &[vk::BufferCopy{src_offset:0,dst_offset:offset as u64,size:4}]);}
    commands.transfer_to_host();commands.submit()?;
    let mut raw=vec![0u8;offset+4];host.read(0,&mut raw)?;
    ensure!(u32::from_ne_bytes(raw[offset..offset+4].try_into().unwrap())==0,"non-finite GPU intermediate at {key}");
    let p=&op.operation.push.0;
    for (name,begin,length,integer) in spans {
        let bytes=raw[begin..begin+length].to_vec();
        if !integer {ensure!(bytes.chunks_exact(4).all(|v|f32::from_ne_bytes(v.try_into().unwrap()).is_finite()),"non-finite trace output at {key}/{name}");}
        points.push(Point{key:format!("{key}/{name}"),meta:json!({"position":position,"op":p[0],"stage":label(model,op),
            "shape_n0_n6":&p[1..8],"offsets":&p[10..14],"flags":p[15],"base_groups":op.operation.groups,
            "array_pipeline":op.operation.array_weights,"elements":length/4,"kind":if integer{"u32"}else{"f32"}}),raw:bytes,integer});
    }
    Ok(())
}
fn static_range(model:&GpuModel,s:&mut GpuSequence,start:usize,end:usize,token:u32,position:usize,
    host:&Arc<Buffer>,detailed:bool,points:&mut Vec<Point>)->Result<()> {
    if detailed {
        for index in start..end {record_one(model,&mut s.commands,&s.forward[index],token,position,false,host,
            &format!("token{position}/forward{index:04}"),points)?;}
    } else {s.commands.begin()?;GpuModel::record_range(&s.commands,&s.forward[start..end],token,position);s.commands.submit()?;}
    Ok(())
}
fn routing_points(position:usize,layer:usize,sparse:&SparseSequence,points:&mut Vec<Point>) {
    for (name,values) in [("normalized_ffn_input",sparse.cpu_input.as_slice()),("selected_experts_and_weights",sparse.cpu_routes.as_slice())] {
        points.push(Point{key:format!("token{position}/layer{layer:02}/{name}"),meta:json!({"position":position,"layer":layer,"stage":name,"kind":"f32","elements":values.len()}),raw:bytes_of_f32(values).to_vec(),integer:false});
    }
}
fn trace(model:&GpuModel,tokens:&[u32],all_operations:bool)->Result<(Vec<Point>,Value)> {
    ensure!(model.config.expert_count>0,"diagnostic requires ordinary MoE");
    let mut sequence=model.new_sequence()?;
    let mut sparse=sequence.sparse.take().context("missing sparse diagnostic state")?;
    let maximum=model.config.vocab_size.max(model.config.embedding_length*model.config.experts_per_token)
        .max(model.config.expert_feed_forward_length*model.config.experts_per_token);
    let host=model.device.host(maximum*4+4096)?;
    let mut points=Vec::new();
    for (position,&token) in tokens.iter().enumerate() {
        let detailed=position==0||all_operations;
        model.ensure_capacity(&mut sequence,position+1)?;
        model.prepare_rope_tables(&mut sequence,1)?;
        sequence.commands.begin()?;
        unsafe {model.device.raw.cmd_fill_buffer(sequence.commands.raw,sequence.status.raw,0,4,0);}
        model.record_rope_upload(&sequence.commands,&sequence,1);
        sequence.commands.transfer_to_compute();sequence.commands.submit()?;
        let first=sparse.routes[0];
        static_range(model,&mut sequence,0,first+1,token,position,&host,detailed,&mut points)?;
        sequence.commands.begin()?;model.copy_routing(&sequence.commands,&sequence,&sparse);sequence.commands.submit()?;
        model.read_routing(&mut sparse)?;
        for layer in 0..sparse.routes.len() {
            routing_points(position,layer,&sparse,&mut points);
            // Match a21-token production prefill's admission policy, irrespective
            // of extra diagnostic dispatch fences/projected first-token logits.
            let graph=model.experts(&sequence,&mut sparse,layer,tokens.len()==1)?;
            if detailed {
                for (index,op) in graph.iter().enumerate() {record_one(model,&mut sparse.commands,op,token,position,index==0,&host,
                    &format!("token{position}/layer{layer:02}/expert{index:02}"),&mut points)?;}
            } else {GpuModel::record_range(&sparse.commands,&graph,token,position);sparse.commands.submit()?;}
            drop(graph);sparse.packed_leases.clear();
            if let Some(prepared)=sparse.pending_upload.take(){model.cache.as_ref().unwrap().commit_prepared(prepared)?;}
            let begin=sparse.routes[layer]+1;
            let end=sparse.routes.get(layer+1).map(|next|next+1).unwrap_or(sequence.forward.len());
            static_range(model,&mut sequence,begin,end,token,position,&host,detailed,&mut points)?;
            if layer+1<sparse.routes.len(){sequence.commands.begin()?;model.copy_routing(&sequence.commands,&sequence,&sparse);sequence.commands.submit()?;model.read_routing(&mut sparse)?;}
        }
        if position==0||position+1==tokens.len(){
            for (index,op) in sequence.project.iter().enumerate(){record_one(model,&mut sequence.commands,op,token,position,false,&host,
                &format!("token{position}/project{index:02}"),&mut points)?;}
        }
        sequence.position+=1;
    }
    sequence.sparse=Some(sparse);
    let stats=model.memory_stats();
    ensure!(stats["cpu_expert_fallbacks"].as_u64()==Some(0),"CPU expert fallback would confound pipeline diagnostic: {stats}");
    Ok((points,stats))
}
fn comparison(reference:&[Point],actual:&[Point])->Value {
    let mut first_bits=None;let mut first_scaled=None;let mut rows=Vec::new();
    for (index,(expected,actual)) in reference.iter().zip(actual).enumerate(){
        if expected.key!=actual.key||expected.raw.len()!=actual.raw.len()||expected.integer!=actual.integer {
            return json!({"structural_divergence":index,"expected_key":expected.key,"actual_key":actual.key});
        }
        let mut different=0;let mut maximum=0.0f64;let mut scaled=0.0f64;let mut square=0.0f64;let mut first=None;
        for (element,(a,b)) in expected.raw.chunks_exact(4).zip(actual.raw.chunks_exact(4)).enumerate(){
            let x=u32::from_ne_bytes(a.try_into().unwrap());let y=u32::from_ne_bytes(b.try_into().unwrap());
            if x!=y {different+=1;if first.is_none(){first=Some(json!({"index":element,"reference_bits":format!("{x:08x}"),"actual_bits":format!("{y:08x}"),
                "reference":if expected.integer{json!(x)}else{json!(f32::from_bits(x))},"actual":if expected.integer{json!(y)}else{json!(f32::from_bits(y))}}));}}
            if !expected.integer {let x=f32::from_bits(x)as f64;let y=f32::from_bits(y)as f64;let delta=(x-y).abs();maximum=maximum.max(delta);scaled=scaled.max(delta/(1.0+x.abs()));square+=delta*delta;}
        }
        let row=json!({"index":index,"key":expected.key,"reference":expected.meta,"actual":actual.meta,
            "different_bits":different,"elements":expected.raw.len()/4,"max_abs":maximum,"max_scaled":scaled,
            "rms":(square/(expected.raw.len()/4).max(1)as f64).sqrt(),"first_different":first});
        if different>0&&first_bits.is_none(){first_bits=Some(row.clone());}
        if (scaled>1e-6||expected.integer&&different>0)&&first_scaled.is_none(){first_scaled=Some(row.clone());}
        rows.push(row);
    }
    json!({"reference_points":reference.len(),"actual_points":actual.len(),"same_point_count":reference.len()==actual.len(),
        "first_bit_difference":first_bits,"first_scaled_over_1e_6_or_integer_difference":first_scaled,"points":rows})
}

#[test]
#[ignore="explicit local checkpoint and idle hardware GPU required"]
fn hardware_direct_layer_intermediates(){
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(),Ok("1"));
    let path=std::env::var_os("NEXTORA_DIRECT_MODEL").expect("NEXTORA_DIRECT_MODEL existing local GGUF required");
    let mut source=Gguf::open(path).unwrap();source.set_expert_cache_capacity(24*1024*1024*1024).unwrap();let source=Arc::new(source);
    eprintln!("shared expert warmup: {:?}",source.warm_expert_cache().unwrap());
    let tokenizer=crate::tokenizer::Tokenizer::from_gguf(&source).unwrap();
    let prompt=std::env::var("NEXTORA_DIRECT_TRACE_PROMPT").unwrap_or_else(|_|"fn square(x: i32) -> i32 {\n    // Return the square of x.\n".into());
    let tokens=if let Ok(ids)=std::env::var("NEXTORA_DIRECT_TRACE_TOKEN_IDS"){serde_json::from_str::<Vec<u32>>(&ids).unwrap()}else{tokenizer.encode(&prompt).unwrap()};
    assert!(!tokens.is_empty()&&tokens.len()<=64,"trace is bounded to64 prompt tokens");
    let fast=std::env::var("NEXTORA_DIRECT_FAST_MATH").map(|value|value!="0"&&value!="false").unwrap_or(true);
    let all=std::env::var("NEXTORA_DIRECT_TRACE_ALL_OPS").as_deref()==Ok("1");
    let modes=[("legacy_packed",DiagnosticPipelineMode::Production,false),("all_array_packed",DiagnosticPipelineMode::AllArrayPacked,true),
        ("all_array_direct",DiagnosticPipelineMode::AllArrayDirect,true),("dual_direct",DiagnosticPipelineMode::Production,true)];
    let mut baseline:Option<Vec<Point>>=None;let mut reports=Vec::new();
    for (name,mode,direct) in modes {
        let (points,stats)=with_diagnostic_pipeline(mode,||{
            let model=GpuModel::load(Arc::clone(&source),40960,DeviceOptions{direct_experts:Some(direct),fast_math:fast,
                memory_budget_bytes:14000*1024*1024,..Default::default()},&|_,_,_,_,_|{}).unwrap();
            eprintln!("tracing {name} at position0 and {} common prompt tokens",tokens.len());
            trace(&model,&tokens,all).unwrap()
        });
        if let Some(reference)=&baseline {let compared=comparison(reference,&points);eprintln!("TRACE_FIRST {name} {}",json!({"first":compared["first_bit_difference"],"first_significant":compared["first_scaled_over_1e_6_or_integer_difference"]}));reports.push(json!({"mode":name,"memory":stats,"comparison":compared}));}
        else {reports.push(json!({"mode":name,"memory":stats,"points":points.len()}));baseline=Some(points);}
    }
    let report=json!({"checkpoint":source.path,"prompt":prompt,"token_ids":tokens,"fast_math":fast,
        "readbacks_serialize_execution":true,"full_operation_positions":if all{json!("all")}else{json!([0])},
        "layer_inputs_and_routes_every_token":true,"all_selected_experts_retained":true,"modes":reports});
    if let Some(path)=std::env::var_os("NEXTORA_DIRECT_TRACE_REPORT"){std::fs::write(path,serde_json::to_vec_pretty(&report).unwrap()).unwrap();}
    eprintln!("DIRECT_LAYER_TRACE_DONE {}",json!({"tokens":tokens.len(),"modes":4}));
}

#[test]
#[ignore="explicit software Vulkan diagnostic smoke test"]
fn software_direct_layer_trace_capture_smoke(){
    assert_eq!(std::env::var("NEXTORA_VULKAN_TEST").as_deref(),Ok("1"));
    let path=std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf");
    let mut source=Gguf::open(path).unwrap();source.set_expert_cache_capacity(1024*1024).unwrap();let source=Arc::new(source);
    source.warm_expert_cache().unwrap();
    let mut baseline:Option<Vec<Point>>=None;
    for (mode,direct) in [(DiagnosticPipelineMode::Production,false),(DiagnosticPipelineMode::AllArrayPacked,true),
        (DiagnosticPipelineMode::AllArrayDirect,true),(DiagnosticPipelineMode::Production,true)] {
        let (points,_)=with_diagnostic_pipeline(mode,||{
            let model=GpuModel::load(Arc::clone(&source),32,DeviceOptions{allow_software:true,direct_experts:Some(direct),
                memory_budget_bytes:16*1024*1024,..Default::default()},&|_,_,_,_,_|{}).unwrap();
            trace(&model,&[2,19],true).unwrap()
        });
        assert!(points.iter().any(|point|point.key.contains("embedding")));
        assert!(points.iter().any(|point|point.key.contains("key_row")));
        assert!(points.iter().any(|point|point.key.contains("selected_experts_and_weights")));
        assert!(points.iter().any(|point|point.key.starts_with("token1/project")));
        if let Some(reference)=&baseline {let compared=comparison(reference,&points);assert_eq!(compared["same_point_count"],true,"{compared}");}
        else{baseline=Some(points);}
    }
}
