// Independent packed block decoding for the mixed GSQ-RCO checkpoint formats.
// Byte layout reference: GGML ggml-common.h / ggml-quants.c. MIT codebook data
// is attributed in ../GGML-NOTICE.md. No weight expansion or int64 requirement.
#ifndef NEXTORA_QUANT_GLSL
#define NEXTORA_QUANT_GLSL
#include "quant_tables.glsl"

uint forge_word_at(uint offset) {
    return byte_at(offset) | (byte_at(offset+1u)<<8u)
        | (byte_at(offset+2u)<<16u) | (byte_at(offset+3u)<<24u);
}
uint forge_grid_byte(uint code, uint lane) { return (code>>(lane*8u))&255u; }

float extended_value_at(uint base, uint index, uint ty) {
    if(ty==42u) { // Q2_0: 64 values, 2-byte scale and sixteen 2-bit bytes.
        uint block=base+(index/64u)*18u;
        uint i=index%64u;
        int code=int((byte_at(block+2u+i/4u)>>(2u*(i%4u)))&3u)-1;
        return half_at(block)*float(code);
    }
    if(ty==20u) { // IQ4_NL: ordered low/high 16-value halves.
        uint block=base+(index/32u)*18u;
        uint i=index%32u;
        uint code=(byte_at(block+2u+i%16u)>>(4u*(i/16u)))&15u;
        return half_at(block)*float(forge_kvalues_iq4nl[code]);
    }
    if(ty==12u || ty==13u) { // Q4_K / Q5_K affine blocks.
        bool five=ty==13u;
        uint block=base+(index/256u)*(five?176u:144u);
        uint i=index%256u, group=i/32u, lane=i%32u;
        uint scale, minimum;
        if(group<4u) {
            scale=byte_at(block+4u+group)&63u;
            minimum=byte_at(block+8u+group)&63u;
        } else {
            scale=(byte_at(block+8u+group)&15u)|((byte_at(block+group)>>6u)<<4u);
            minimum=(byte_at(block+8u+group)>>4u)|((byte_at(block+4u+group)>>6u)<<4u);
        }
        uint code=(byte_at(block+(five?48u:16u)+(group/2u)*32u+lane)>>(4u*(group%2u)))&15u;
        if(five) code|=((byte_at(block+16u+lane)>>group)&1u)<<4u;
        return (half_at(block)*float(scale))*float(code)-half_at(block+2u)*float(minimum);
    }
    if(ty==14u) { // Q6_K signed sub-block scales.
        uint block=base+(index/256u)*210u;
        uint i=index%256u, section=i/128u, quadrant=(i%128u)/32u, lane=i%32u;
        uint low=(byte_at(block+section*64u+(quadrant%2u)*32u+lane)>>(4u*(quadrant/2u)))&15u;
        uint high=(byte_at(block+128u+section*32u+lane)>>(2u*quadrant))&3u;
        int code=int(low|(high<<4u))-32;
        int scale=int(byte_at(block+192u+section*8u+quadrant*2u+lane/16u));
        if(scale>=128) scale-=256;
        return (half_at(block+208u)*float(scale))*float(code);
    }
    if(ty==23u) { // IQ4_XS signed 6-bit scales and nonlinear 4-bit values.
        uint block=base+(index/256u)*136u;
        uint i=index%256u, group=i/32u, lane=i%32u;
        uint low=(byte_at(block+4u+group/2u)>>(4u*(group%2u)))&15u;
        uint high=(half_word(block+2u)>>(2u*group))&3u;
        int scale=int(low|(high<<4u))-32;
        uint code=(byte_at(block+8u+group*16u+lane%16u)>>(4u*(lane/16u)))&15u;
        return (half_at(block)*float(scale))*float(forge_kvalues_iq4nl[code]);
    }
    if(ty==22u) { // IQ2_S 10-bit eight-value grid and explicit signs.
        uint block=base+(index/256u)*82u;
        uint i=index%256u, group=i/32u, octet=(i%32u)/8u, lane=i%8u;
        uint scale_code=(byte_at(block+74u+group)>>(4u*(octet/2u)))&15u;
        float scale=half_at(block)*(0.5+float(scale_code))*0.25;
        uint key=byte_at(block+2u+group*4u+octet)
            | (((byte_at(block+66u+group)>>(2u*octet))&3u)<<8u);
        uvec2 grid=forge_iq2s_grid[key];
        uint value=forge_grid_byte(lane<4u?grid.x:grid.y,lane%4u);
        uint signs=byte_at(block+34u+group*4u+octet);
        return scale*float(value)*(((signs>>lane)&1u)==0u?1.0:-1.0);
    }
    if(ty==18u) { // IQ3_XXS paired four-value grids and compressed signs.
        uint block=base+(index/256u)*98u;
        uint i=index%256u, group=i/32u, octet=(i%32u)/8u, lane=i%8u;
        uint codes=forge_word_at(block+66u+group*4u);
        float scale=half_at(block)*(0.5+float(codes>>28u))*0.5;
        uint signs=forge_ksigns_iq2xs[(codes>>(7u*octet))&127u];
        uint key=byte_at(block+2u+group*8u+octet*2u+lane/4u);
        uint value=forge_grid_byte(forge_iq3xxs_grid[key],lane%4u);
        return scale*float(value)*(((signs>>lane)&1u)==0u?1.0:-1.0);
    }
    if(ty==21u) { // IQ3_S 9-bit four-value grids and explicit signs.
        uint block=base+(index/256u)*110u;
        uint i=index%256u, group=i/32u, octet=(i%32u)/8u, lane=i%8u;
        uint scale_code=(byte_at(block+106u+group/2u)>>(4u*(group%2u)))&15u;
        float scale=half_at(block)*float(1u+2u*scale_code);
        uint packed_index=octet*2u+lane/4u;
        uint key=byte_at(block+2u+group*8u+packed_index)
            | (((byte_at(block+66u+group)>>packed_index)&1u)<<8u);
        uint value=forge_grid_byte(forge_iq3s_grid[key],lane%4u);
        uint signs=byte_at(block+74u+group*4u+octet);
        return scale*float(value)*(((signs>>lane)&1u)==0u?1.0:-1.0);
    }
    return invalid_value();
}
#endif
