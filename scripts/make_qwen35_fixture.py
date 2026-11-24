#!/usr/bin/env python3
"""Tiny untrained Qwen35 GGUF plus independent scalar reference outputs.

The fixture has three DeltaNet layers and one gated-attention layer, partial
RoPE, tiled value-head order, dense SwiGLU, and an optional validated MTP block.
It uses no downloaded weights or Python packages. It proves equations and
state behavior, not pretrained quality or hardware speed.
"""
import argparse, json, math, struct
from pathlib import Path
from make_tiny_fixture import u32,u64,string,aligned,byte_token

def build(mtp=False):
    n,ff,heads,kv,hd,rot=8,16,2,1,8,2
    nk,nv,dk,dv,kernel=2,4,2,2,4
    inner=nv*dv;channels=2*nk*dk+inner
    tokens=[byte_token(i) for i in range(256)]+['<|endoftext|>','<|im_start|>','<|im_end|>','<think>','</think>']
    metadata=[];weights={};descriptions=[]
    def meta(k,t,v):metadata.append(string(k)+u32(t)+v)
    def mi(k,v):meta(k,4,u32(v))
    def mf(k,v):meta(k,6,struct.pack('<f',v))
    def ms(k,v):meta(k,8,string(v))
    def array(k,t,values):meta(k,9,u32(t)+u64(len(values))+b''.join(string(v) if t==8 else u32(v) for v in values))
    ms('general.architecture','qwen35');ms('general.type','model');ms('general.name','Nextora untrained mixed Qwen35 equation fixture')
    for suffix,value in {'block_count':5 if mtp else 4,'context_length':512,'embedding_length':n,'feed_forward_length':ff,'attention.head_count':heads,'attention.head_count_kv':kv,'attention.key_length':hd,'attention.value_length':hd,'rope.dimension_count':rot,'full_attention_interval':4,'ssm.group_count':nk,'ssm.time_step_rank':nv,'ssm.state_size':dk,'ssm.inner_size':inner,'ssm.conv_kernel':kernel}.items():mi('qwen35.'+suffix,value)
    if mtp:mi('qwen35.nextn_predict_layers',1)
    mf('qwen35.rope.freq_base',10000.0);mf('qwen35.attention.layer_norm_rms_epsilon',1e-6)
    array('qwen35.rope.dimension_sections',4,[1,0,0,0])
    ms('tokenizer.ggml.model','gpt2');ms('tokenizer.ggml.pre','qwen2');array('tokenizer.ggml.tokens',8,tokens);array('tokenizer.ggml.merges',8,[]);array('tokenizer.ggml.token_type',4,[1]*256+[3]*5);mi('tokenizer.ggml.eos_token_id',256)
    def tensor(name,dims,values):
        assert math.prod(dims)==len(values)
        raw=struct.pack('<'+'f'*len(values),*values);values=list(struct.unpack('<'+'f'*len(values),raw))
        weights[name]=(dims,values);descriptions.append((name,dims,raw))
    def matrix(name,cols,rows,seed,amplitude=.1):
        tensor(name,[cols,rows],[amplitude*(((r*13+c*7+seed*11)%23)-11)/11 for r in range(rows) for c in range(cols)])
    def norm(name,width):tensor(name,[width],[1.0+.01*j for j in range(width)])
    tensor('token_embd.weight',[n,len(tokens)],[.1+((t*7+c*3)%17)*.025 for t in range(len(tokens)) for c in range(n)])
    norm('output_norm.weight',n);matrix('output.weight',n,len(tokens),91,.13)
    for layer in range(5 if mtp else 4):
        prefix=f'blk.{layer}.';draft=layer==4
        if not draft:norm(prefix+'attn_norm.weight',n)
        norm(prefix+'post_attention_norm.weight',n)
        matrix(prefix+'ffn_gate.weight',n,ff,layer+1,.08);matrix(prefix+'ffn_up.weight',n,ff,layer+2,.09);matrix(prefix+'ffn_down.weight',ff,n,layer+3,.08)
        if layer>=3:
            matrix(prefix+'attn_q.weight',n,2*heads*hd,layer+4,.13);matrix(prefix+'attn_k.weight',n,kv*hd,layer+5,.12);matrix(prefix+'attn_v.weight',n,kv*hd,layer+6,.11);matrix(prefix+'attn_output.weight',heads*hd,n,layer+7,.12)
            norm(prefix+'attn_q_norm.weight',hd);norm(prefix+'attn_k_norm.weight',hd)
        else:
            matrix(prefix+'attn_qkv.weight',n,channels,layer+4,.15);matrix(prefix+'attn_gate.weight',n,inner,layer+5,.2)
            matrix(prefix+'ssm_beta.weight',n,nv,layer+6,.11);matrix(prefix+'ssm_alpha.weight',n,nv,layer+7,.09);matrix(prefix+'ssm_out.weight',inner,n,layer+8,.15)
            tensor(prefix+'ssm_a',[nv],[-.4-.1*j for j in range(nv)]);tensor(prefix+'ssm_dt.bias',[nv],[-.2+.1*j for j in range(nv)])
            tensor(prefix+'ssm_conv1d.weight',[kernel,channels],[v*(1+.01*c) for c in range(channels) for v in [.03,.07,.1,.6]])
            norm(prefix+'ssm_norm.weight',dv)
        if draft:
            matrix(prefix+'nextn.eh_proj.weight',n*2,n,99)
            for suffix in ['nextn.enorm.weight','nextn.hnorm.weight','nextn.shared_head_norm.weight']:norm(prefix+suffix,n)
    payload=b'';descriptors=b''
    for name,dims,raw in descriptions:
        descriptors+=string(name)+u32(len(dims))+b''.join(u64(d) for d in dims)+u32(0)+u64(len(payload));payload+=aligned(raw)
    raw=aligned(b'GGUF'+u32(3)+u64(len(descriptions))+u64(len(metadata))+b''.join(metadata)+descriptors)+payload
    # A separate row-major scalar interpreter creates the golden outputs. The
    # source weights have already been rounded to the fixture's F32 storage.
    def values(name):return weights[name][1]
    def mv(name,x):
        dims,w=weights[name];return [sum(w[r*dims[0]+c]*x[c] for c in range(dims[0])) for r in range(dims[1])]
    def rms(x,w):
        inv=1/math.sqrt(sum(v*v for v in x)/len(x)+1e-6);return [v*inv*s for v,s in zip(x,w)]
    def sigmoid(x):return 1/(1+math.exp(-x))
    def silu(x):return x*sigmoid(x)
    states=[{'conv':[0.]*(channels*(kernel-1)),'matrix':[0.]*(nv*dk*dv)} for _ in range(3)]
    keys=[];vals=[];golden=[]
    for position,token in enumerate([3,8,19,77]):
        residual=values('token_embd.weight')[token*n:(token+1)*n]
        for layer in range(4):
            p=f'blk.{layer}.';x=rms(residual,values(p+'attn_norm.weight'))
            if layer<3:
                state=states[layer];qkv=mv(p+'attn_qkv.weight',x);z=mv(p+'attn_gate.weight',x);beta=mv(p+'ssm_beta.weight',x);alpha=mv(p+'ssm_alpha.weight',x)
                conv=values(p+'ssm_conv1d.weight');active=[]
                for c in range(channels):active.append(silu(sum(state['conv'][tap*channels+c]*conv[c*kernel+tap] for tap in range(kernel-1))+qkv[c]*conv[c*kernel+kernel-1]))
                state['conv']=state['conv'][channels:]+qkv
                q=active[:nk*dk];k=active[nk*dk:2*nk*dk];v=active[2*nk*dk:]
                for vector in [q,k]:
                    for h in range(nk):
                        denom=math.sqrt(sum(t*t for t in vector[h*dk:(h+1)*dk])+1e-6)
                        vector[h*dk:(h+1)*dk]=[t/denom for t in vector[h*dk:(h+1)*dk]]
                recurrent=[]
                for h in range(nv):
                    kh=h%nk;decay=math.exp(values(p+'ssm_a')[h]*math.log1p(math.exp(alpha[h]+values(p+'ssm_dt.bias')[h])))
                    y=[]
                    for j in range(dv):
                        offset=(h*dv+j)*dk;old=[s*decay for s in state['matrix'][offset:offset+dk]]
                        delta=(v[h*dv+j]-sum(old[i]*k[kh*dk+i] for i in range(dk)))*sigmoid(beta[h])
                        new=[old[i]+delta*k[kh*dk+i] for i in range(dk)];state['matrix'][offset:offset+dk]=new
                        y.append(sum(new[i]*q[kh*dk+i] for i in range(dk))/math.sqrt(dk))
                    normed=rms(y,values(p+'ssm_norm.weight'));recurrent.extend(normed[j]*silu(z[h*dv+j]) for j in range(dv))
                update=mv(p+'ssm_out.weight',recurrent)
            else:
                qg=mv(p+'attn_q.weight',x);q=[]
                for h in range(heads):q.extend(rms(qg[2*h*hd:2*h*hd+hd],values(p+'attn_q_norm.weight')))
                k=rms(mv(p+'attn_k.weight',x),values(p+'attn_k_norm.weight'));v=mv(p+'attn_v.weight',x)
                for vector,count in [(q,heads),(k,kv)]:
                    for h in range(count):
                        a,b=vector[h*hd],vector[h*hd+1];co,si=math.cos(position),math.sin(position)
                        vector[h*hd]=a*co-b*si;vector[h*hd+1]=b*co+a*si
                keys.extend(k);vals.extend(v);attention=[]
                for h in range(heads):
                    scores=[sum(q[h*hd+j]*keys[t*kv*hd+j] for j in range(hd))/math.sqrt(hd) for t in range(position+1)]
                    probs=[math.exp(s-max(scores)) for s in scores];denom=sum(probs)
                    attention.extend(sum(probs[t]*vals[t*kv*hd+j] for t in range(position+1))/denom*sigmoid(qg[2*h*hd+hd+j]) for j in range(hd))
                update=mv(p+'attn_output.weight',attention)
            residual=[a+b for a,b in zip(residual,update)]
            x=rms(residual,values(p+'post_attention_norm.weight'));gate=mv(p+'ffn_gate.weight',x);up=mv(p+'ffn_up.weight',x)
            update=mv(p+'ffn_down.weight',[silu(a)*b for a,b in zip(gate,up)]);residual=[a+b for a,b in zip(residual,update)]
        hidden=rms(residual,values('output_norm.weight'));logits=mv('output.weight',hidden)
        golden.append({'hidden':hidden,'logits':logits[:16]})
    return raw,{'prompt':[3,8,19,77],'steps':golden,'recurrent':states,'keys':keys,'values':vals}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--mtp',action='store_true');args=parser.parse_args()
    data,reference=build(args.mtp);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(data);args.output.with_suffix('.reference.json').write_text(json.dumps(reference,indent=2)+'\n');print(args.output,len(data))
if __name__=='__main__':main()
