// Fast Direct Q4_K/Q6_K GEMV, one workgroup per output row.
// Each lane visits the same columns as its two original 128-lane counterparts.
// Fold the two partials in their original order before reducing in shared memory.
float wave_partial(uint base, uint width, uint offset, uint lane) {
    return p.n2 == 12u ? fast_q4_k_partial(base,width,offset,lane)
                       : fast_q6_k_partial(base,width,offset,lane);
}
void main() {
    uint lane = gl_LocalInvocationID.x;
    uint row = flat_group();
    if (row >= p.n1) return;
    uint slot=gl_WorkGroupID.z;
    uint expert=0u;
    if((p.flags&1u)!=0u) {
        float chosen=e[slot*2u];
        if(isnan(chosen)||chosen<0.0||chosen>=float(p.n4)) {
            if(lane==0u)c[p.off2+slot*p.n1+row]=invalid_value();
            return;
        }
        expert=uint(chosen);
    }
    uint base=p.off0+(expert*p.n1+row)*p.n3;
#ifdef NEXTORA_DIRECT_EXPERTS
    if((p.flags&512u)!=0u)base=p.off0+row*p.n3;
#endif
    uint input_offset=p.off1+(((p.flags&2u)!=0u)?slot*p.n0:0u);
    float sum = wave_partial(base,p.n0,input_offset,lane)
              + wave_partial(base,p.n0,input_offset,lane+64u);
    if (EXPERT_LANES == 32u) {
        sum += wave_partial(base,p.n0,input_offset,lane+32u)
             + wave_partial(base,p.n0,input_offset,lane+96u);
    }
    reduction[lane] = sum;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) reduction[lane] += reduction[lane+stride];
        barrier();
    }
    sum = reduction[0];
    if (lane == 0u) c[p.off2+slot*p.n1+row] = checked(sum);
}
