// Independent Q4_64 attention-cache storage. Each 128-value block is 17 u32:
// two F16 scales in word zero, then sixteen little-endian signed-nibble words.
// This is a runtime KV layout, independent of every GGUF weight encoding.
#ifndef NEXTORA_COMPACT_KV_GLSL
#define NEXTORA_COMPACT_KV_GLSL
layout(set=0,binding=2,std430) buffer CompactSecond { uint kv_second[]; };
layout(set=0,binding=3,std430) buffer CompactOutput { uint kv_output[]; };
layout(set=0,binding=4,std430) buffer CompactState { uint kv_state[]; };
shared float kv_scale[4];
shared uint kv_scale_packed[4];
shared float kv_probabilities[1024];

uint kv_row_words(uint width) { return ((width+127u)/128u)*17u; }
float kv_finite(float value) {
    checked(value);
    return isnan(value)||isinf(value)?0.0:value;
}
float kv_scale_from_bits(uint packed) {
    uint exponent=packed>>10u,mantissa=packed&1023u;
    return exponent==0u?float(mantissa)*5.9604644775390625e-8:
        uintBitsToFloat(((exponent+112u)<<23u)|(mantissa<<13u));
}
uint kv_positive_half_rne(float value) {
    uint bits=floatBitsToUint(value);
    int exponent=int((bits>>23u)&255u)-112;
    uint mantissa=bits&8388607u;
    if(exponent<=0) {
        if(exponent< -10)return 0u;
        mantissa|=8388608u;
        uint shift=uint(14-exponent);
        uint rounded=mantissa>>shift;
        uint remainder=mantissa&((1u<<shift)-1u),halfway=1u<<(shift-1u);
        return rounded+uint(remainder>halfway||(remainder==halfway&&(rounded&1u)!=0u));
    }
    uint rounded=(uint(exponent)<<10u)|(mantissa>>13u);
    uint remainder=mantissa&8191u;
    return rounded+uint(remainder>4096u||(remainder==4096u&&(rounded&1u)!=0u));
}
uint kv_round_scale(float maximum) {
    if(maximum==0.0)return 0u;
    // Retain a representable nonzero scale for tiny finite groups. Oversized
    // scales report an error rather than silently accepting clipped history.
    float scale=max(maximum/7.0,5.9604644775390625e-8);
    if(maximum>458528.0){atomicOr(status[0],1u);scale=65504.0;}
    // packHalf2x16 rounding differs between Vulkan implementations. Specify
    // nearest-even in integer arithmetic, then refine the divide against exact
    // F16 midpoints times seven (at most fifteen significant binary digits).
    uint packed=max(1u,min(31743u,kv_positive_half_rne(scale)));
    float current=kv_scale_from_bits(packed);
    float lower=(kv_scale_from_bits(packed-1u)+current)*3.5;
    bool odd=(packed&1u)!=0u;
    if(packed>1u&&(maximum<lower||(maximum==lower&&odd)))packed--;
    else if(packed<31743u) {
        float upper=(current+kv_scale_from_bits(packed+1u))*3.5;
        if(maximum>upper||(maximum==upper&&odd))packed++;
    }
    return packed;
}
float kv_max64(float value) {
    uint lane=gl_LocalInvocationID.x;
    reduction[lane]=value;
    barrier();
    for(uint stride=32u;stride>0u;stride>>=1u) {
        if((lane%64u)<stride)reduction[lane]=max(reduction[lane],reduction[lane+stride]);
        barrier();
    }
    return reduction[(lane/64u)*64u];
}
int kv_quantize(float value,float scale) {
    if(scale==0.0)return 0;
    float magnitude=abs(value);
    int code=int(clamp(roundEven(magnitude/scale),0.0,7.0));
    // A device may approximate division. F16 scales times these small half-
    // integers are exact F32 boundaries, so refine a near-tie independently of
    // that approximation and retain nearest-even decisions on every device.
    float lower=(float(code)-0.5)*scale,upper=(float(code)+0.5)*scale;
    bool odd=(code&1)!=0;
    if(magnitude<lower||(magnitude==lower&&odd))code=max(code-1,0);
    else if(magnitude>upper||(magnitude==upper&&odd))code=min(code+1,7);
    return value<0.0?-code:code;
}
void compact_kv_store() {
    uint lane=gl_LocalInvocationID.x, block=flat_group();
    if(block>=(p.n0+127u)/128u)return;
    uint source=block*128u+lane;
    float key=source<p.n0?kv_finite(a[source]):0.0;
    float value=source<p.n0?kv_finite(b[source]):0.0;
    float key_max=kv_max64(abs(key));
    if(lane==0u||lane==64u) {
        uint index=lane/64u;
        kv_scale_packed[index]=kv_round_scale(key_max);
        kv_scale[index]=kv_scale_from_bits(kv_scale_packed[index]);
    }
    barrier();
    float value_max=kv_max64(abs(value));
    if(lane==0u||lane==64u) {
        uint index=2u+lane/64u;
        kv_scale_packed[index]=kv_round_scale(value_max);
        kv_scale[index]=kv_scale_from_bits(kv_scale_packed[index]);
    }
    barrier();
    uint target=p.n1*kv_row_words(p.n0)+block*17u;
    if(lane==0u) {
        kv_output[target]=kv_scale_packed[0]|(kv_scale_packed[1]<<16u);
        kv_state[target]=kv_scale_packed[2]|(kv_scale_packed[3]<<16u);
    }
    if(lane<16u) {
        float key_scale=kv_scale[lane/8u], value_scale=kv_scale[2u+lane/8u];
        uint key_word=0u,value_word=0u;
        for(uint component=0u;component<8u;component++) {
            uint index=block*128u+lane*8u+component;
            float k=index<p.n0?kv_finite(a[index]):0.0;
            float v=index<p.n0?kv_finite(b[index]):0.0;
            int qk=kv_quantize(k,key_scale);
            int qv=kv_quantize(v,value_scale);
            key_word|=(uint(qk)&15u)<<(component*4u);
            value_word|=(uint(qv)&15u)<<(component*4u);
        }
        kv_output[target+1u+lane]=key_word;
        kv_state[target+1u+lane]=value_word;
    }
}
float compact_key(uint token,uint coordinate,uint width) {
    uint block=token*kv_row_words(width)+(coordinate/128u)*17u;
    uint within=coordinate%128u;
    vec2 scales=unpackHalf2x16(kv_second[block]);
    uint nibble=(kv_second[block+1u+within/8u]>>((within%8u)*4u))&15u;
    int code=int(nibble);if(code>=8)code-=16;
    return float(code)*(within<64u?scales.x:scales.y);
}
float compact_value(uint token,uint coordinate,uint width) {
    uint block=token*kv_row_words(width)+(coordinate/128u)*17u;
    uint within=coordinate%128u;
    vec2 scales=unpackHalf2x16(kv_state[block]);
    uint nibble=(kv_state[block+1u+within/8u]>>((within%8u)*4u))&15u;
    int code=int(nibble);if(code>=8)code-=16;
    return float(code)*(within<64u?scales.x:scales.y);
}
void compact_attention_partial() {
    uint lane=gl_LocalInvocationID.x,chunk=gl_WorkGroupID.x,head=gl_WorkGroupID.y;
    uint start=chunk*1024u;
    if(start>=p.n0||head>=p.n1)return;
    uint kv_head=head/(p.n1/p.n2);
    float local_max=-3.402823466e38;
    for(uint i=lane;i<1024u;i+=128u) {
        uint token=start+i;
        float score=-3.402823466e38;
        if(token<p.n0) {
            float dot=0.0;
            for(uint component=0u;component<p.n3;component++) {
                dot+=a[head*p.n3+component]*compact_key(token,kv_head*p.n3+component,p.n5);
            }
            score=checked(dot*inversesqrt(float(p.n3)));
        }
        kv_probabilities[i]=score;
        local_max=max(local_max,score);
    }
    float maximum=max_group(local_max);
    barrier();
    float local_sum=0.0;
    for(uint i=lane;i<1024u;i+=128u) {
        float probability=start+i<p.n0?exp(kv_probabilities[i]-maximum):0.0;
        kv_probabilities[i]=probability;
        local_sum+=probability;
    }
    float denominator=sum_group(local_sum);
    barrier();
    uint output_base=(head*p.n4+chunk)*(p.n3+2u);
    if(lane==0u){c[output_base]=checked(maximum);c[output_base+1u]=checked(denominator);}
    uint count=min(1024u,p.n0-start);
    for(uint component=lane;component<p.n3;component+=128u) {
        float total=0.0;
        for(uint i=0u;i<count;i++) {
            total+=kv_probabilities[i]*compact_value(start+i,kv_head*p.n3+component,p.n5);
        }
        c[output_base+2u+component]=checked(total);
    }
}
void compact_attention_merge() {
    uint lane=gl_LocalInvocationID.x,head=flat_group();
    if(head>=p.n1)return;
    float local_max=-3.402823466e38;
    for(uint chunk=lane;chunk<p.n0;chunk+=128u) {
        local_max=max(local_max,a[(head*p.n4+chunk)*(p.n3+2u)]);
    }
    float maximum=max_group(local_max);
    barrier();
    float local_sum=0.0;
    for(uint chunk=lane;chunk<p.n0;chunk+=128u) {
        uint input_base=(head*p.n4+chunk)*(p.n3+2u);
        local_sum+=exp(a[input_base]-maximum)*a[input_base+1u];
    }
    float denominator=sum_group(local_sum);
    barrier();
    for(uint component=lane;component<p.n3;component+=128u) {
        float total=0.0;
        for(uint chunk=0u;chunk<p.n0;chunk++) {
            uint input_base=(head*p.n4+chunk)*(p.n3+2u);
            total+=exp(a[input_base]-maximum)*a[input_base+2u+component];
        }
        c[head*p.n3+component]=checked(total/denominator);
    }
}
#endif
