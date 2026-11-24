// Optional Q8 activation / packed integer-dot execution. Model weights retain
// their GGUF format; each 32-value activation group has an independent F32 scale.
#ifndef NEXTORA_FAST_DOT_GLSL
#define NEXTORA_FAST_DOT_GLSL

// Alias the second storage binding as integer words so arbitrary Q8 bit
// patterns are stored exactly, including bit patterns representing float NaNs.
layout(set=0,binding=2,std430) buffer PackedActivations { uint q8_words[]; };

// OpSDotKHR with packed 4x8-bit operands. Using the standard SPIR-V intrinsic
// keeps the build compatible with glslang versions predating GLSL's dot API.
spirv_instruction(extensions=["SPV_KHR_integer_dot_product"], capabilities=[6019,6018], id=4450)
int q8_signed_dot_instruction(int lhs, int rhs, spirv_literal int format);
int q8_signed_dot(int lhs, int rhs) { return q8_signed_dot_instruction(lhs,rhs,0); }

void quantize_activation_group() {
    uint group=flat_group();
    if(group>=(p.n0+31u)/32u)return;
    uint lane=gl_LocalInvocationID.x;
    uint start=group*32u;
    float magnitude=0.0;
    if(lane<32u && start+lane<p.n0) {
        float value=checked(a[p.off0+start+lane]);
        if(!isnan(value)&&!isinf(value))magnitude=abs(value);
    }
    float maximum=max_group(magnitude);
    float scale=maximum/127.0;
    if(lane==0u)d[p.off3+group]=scale;
    if(lane<8u) {
        uint packed=0u;
        for(uint component=0u;component<4u;component++) {
            uint column=start+lane*4u+component;
            float value=column<p.n0?a[p.off0+column]:0.0;
            if(isnan(value)||isinf(value))value=0.0;
            int quantized=scale>0.0?int(clamp(roundEven(value/scale),-127.0,127.0)):0;
            packed|=(uint(quantized)&255u)<<(component*8u);
        }
        q8_words[p.off1+group*8u+lane]=packed;
    }
}

float fast_q4_k_partial(uint base, uint width, uint input_offset, uint lane) {
    float partial=0.0;
    for(uint column=lane*4u;column<width;column+=512u) {
        uint block=base+(column/256u)*144u;
        uint within=column%256u, group=within/32u, component=within%32u;
        uint scale,minimum;
        if(group<4u) {
            scale=byte_at(block+4u+group)&63u;
            minimum=byte_at(block+8u+group)&63u;
        } else {
            scale=(byte_at(block+8u+group)&15u)|((byte_at(block+group)>>6u)<<4u);
            minimum=(byte_at(block+8u+group)>>4u)|((byte_at(block+4u+group)>>6u)<<4u);
        }
        uint codes=(forge_word_at(block+16u+(group/2u)*32u+component)>>(4u*(group%2u)))&0x0f0f0f0fu;
        int activation=int(q8_words[(input_offset+column)/4u]);
        int products=q8_signed_dot(int(codes),activation);
        int activation_sum=q8_signed_dot(0x01010101,activation);
        vec2 global=unpackHalf2x16(forge_word_at(block));
        float activation_scale=d[(input_offset+column)/32u];
        partial+=activation_scale*((global.x*float(scale))*float(products)-(global.y*float(minimum))*float(activation_sum));
    }
    return partial;
}

float fast_q6_k_partial(uint base, uint width, uint input_offset, uint lane) {
    float partial=0.0;
    for(uint column=lane*4u;column<width;column+=512u) {
        uint block=base+(column/256u)*210u;
        uint within=column%256u, section=within/128u;
        uint quadrant=(within%128u)/32u, component=within%32u;
        uint low=(forge_word_at(block+section*64u+(quadrant%2u)*32u+component)>>(4u*(quadrant/2u)))&0x0f0f0f0fu;
        uint high=(forge_word_at(block+128u+section*32u+component)>>(2u*quadrant))&0x03030303u;
        uint codes=low|(high<<4u);
        int activation=int(q8_words[(input_offset+column)/4u]);
        int products=q8_signed_dot(int(codes),activation)-32*q8_signed_dot(0x01010101,activation);
        int scale=int(byte_at(block+192u+section*8u+quadrant*2u+component/16u));
        if(scale>=128)scale-=256;
        // scripts/build-direct-shaders.py preserves the legacy modules and
        // lowers this expression for the direct module to two rounded products
        // (activation_scale*products, then signed scale) and a final fused
        // accumulation with the encoded half scale. That order matches the
        // original AMD kernel; GLSL precise would propagate to unrelated math.
        partial+=(half_at(block+208u)*float(scale))*d[(input_offset+column)/32u]*float(products);
    }
    return partial;
}
#endif
