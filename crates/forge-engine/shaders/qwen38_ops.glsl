// Independent Forge Qwen3.8 operations, MIT.
// Included after checked(), sum_group(), max_group() and flat_group().
// All dimensions/allocations are validated by the Rust model plan. Float buffer
// indices are element indices; w[] stores little-endian F32 words for the small
// decoded normalization/convolution weights or the previous recurrent matrix.
// The binding/dimension contract is documented beside each operation.

float qwen_sigmoid(float x) {
    return x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}
float qwen_silu(float x) { return x * qwen_sigmoid(x); }
float qwen_softplus(float x) { return x > 20.0 ? x : log(1.0 + exp(x)); }
uint qwen_score_bits(float score) {
    if (isnan(score) || isinf(score) || score < 0.0) {
        invalid_value();
        return 0u;
    }
    return score == 0.0 ? 0u : floatBitsToUint(score);
}

void qwen_operation() {
    uint lane = gl_LocalInvocationID.x;
    uint group = flat_group();
    uint element = group * 128u + lane;

    if (p.op == 100u) { // Replicate embedding: a[n] -> c[streams*n].
        if (element >= p.n0 * p.n1) return;
        c[element] = checked(a[element % p.n0]);

    } else if (p.op == 101u) { // HC low rank: n0=rank, n1=streams; a -> c.
        if (element >= p.n0) return;
        c[element] = checked(qwen_silu(a[element] / float(p.n1)));

    } else if (p.op == 102u) { // HC read: a=normalized streams, b=up gates, c=mixed.
        if (element >= p.n0) return; // n0=model width, n1=streams.
        float total = 0.0;
        for (uint stream = 0u; stream < p.n1; ++stream) {
            uint i = stream * p.n0 + element;
            total += a[i] * qwen_sigmoid(b[i]) / float(p.n1);
        }
        c[element] = checked(total);

    } else if (p.op == 103u) { // HC write: a=streams, b=update, e=inject, c=streams.
        if (element >= p.n0 * p.n1) return;
        uint stream = element / p.n0;
        float gate = 2.0 * qwen_sigmoid(e[stream] / float(p.n1));
        c[element] = checked(a[element] + gate * b[element % p.n0]);

    } else if (p.op == 104u) { // Extract Q: a=[query,gate] per head; c=query.
        if (element >= p.n0 * p.n1) return; // n0=head width, n1=query heads.
        c[element] = checked(a[(element / p.n0) * (2u * p.n0) + element % p.n0]);

    } else if (p.op == 105u) { // Gate attention: a=attention, b=packed Q/gate, c=result.
        if (element >= p.n0 * p.n1) return;
        uint gate = (element / p.n0) * (2u * p.n0) + p.n0 + element % p.n0;
        c[element] = checked(a[element] * qwen_sigmoid(b[gate]));

    } else if (p.op == 106u) { // Shared expert: a=routed, b=shared, e[0]=gate; c=sum.
        if (element >= p.n0) return; // flags bit0 means an ungated shared expert.
        float scale = (p.flags & 1u) != 0u ? 1.0 : qwen_sigmoid(e[0]);
        c[element] = checked(a[element] + scale * b[element]);

    } else if (p.op == 107u) { // Element copy with offsets; used for raw indexer append.
        if (element >= p.n0) return;
        c[p.off2 + element] = checked(a[p.off0 + element]);

    } else if (p.op == 110u) { // Delta convolution: n0=channels, n1=kernel taps.
        // w=[channel][tap], a=current raw QKV, b=old history (oldest first),
        // c=SiLU convolution, d=new history. b and d may alias: a channel reads
        // every old tap before shifting its own elements toward lower addresses.
        if (element >= p.n0) return;
        float total = 0.0;
        for (uint tap = 0u; tap + 1u < p.n1; ++tap) {
            total += b[tap * p.n0 + element] * uintBitsToFloat(w[element * p.n1 + tap]);
        }
        total += a[element] * uintBitsToFloat(w[element * p.n1 + p.n1 - 1u]);
        c[element] = checked(qwen_silu(total));
        for (uint tap = 0u; tap + 2u < p.n1; ++tap) {
            d[tap * p.n0 + element] = b[(tap + 1u) * p.n0 + element];
        }
        if (p.n1 > 1u) d[(p.n1 - 2u) * p.n0 + element] = checked(a[element]);

    } else if (p.op == 111u) { // Delta Q/K L2: n0=key width, n1=2*key heads; f0=epsilon.
        // a and c MUST alias the full activated QKV buffer; the V suffix is kept.
        if (group >= p.n1) return;
        uint base = group * p.n0;
        float partial = 0.0;
        for (uint i = lane; i < p.n0; i += 128u) partial += a[base + i] * a[base + i];
        float norm = sqrt(checked(sum_group(partial)));
        float inverse = 1.0 / max(norm, p.f0);
        barrier();
        for (uint i = lane; i < p.n0; i += 128u) c[base + i] = checked(a[base + i] * inverse);

    } else if (p.op == 112u) { // Delta gates: n0=value heads; w=[dt bias | A].
        // a=beta projection, b=alpha projection, c=[sigmoid beta | exponential decay].
        if (element >= p.n0) return;
        float beta = qwen_sigmoid(a[element]);
        float exponent = qwen_softplus(b[element] + uintBitsToFloat(w[element]))
                       * uintBitsToFloat(w[p.n0 + element]);
        c[element] = checked(beta);
        c[p.n0 + element] = checked(exp(exponent));

    } else if (p.op == 113u) { // Delta recurrence: n0=dk,n1=nk,n2=nv,n3=dv.
        // One workgroup owns one [value head, value coordinate] matrix row.
        // w=old state, a=normalized QKV, b=packed gates, c=new state, d=raw output.
        // w/c may alias: all prediction reads finish before any row update and
        // each lane writes only its own columns. No other workgroup reads this row.
        uint row = group;
        if (row >= p.n2 * p.n3) return;
        uint head = row / p.n3;
        uint coordinate = row % p.n3;
        uint key_head = head % p.n1;
        uint query_base = key_head * p.n0;
        uint key_base = p.n1 * p.n0 + query_base;
        uint state_base = row * p.n0;
        float decay = b[p.n2 + head];
        float prediction = 0.0;
        for (uint i = lane; i < p.n0; i += 128u) {
            prediction += uintBitsToFloat(w[state_base + i]) * decay * a[key_base + i];
        }
        prediction = sum_group(prediction);
        barrier();
        float value = a[2u * p.n1 * p.n0 + head * p.n3 + coordinate];
        float delta = b[head] * (value - prediction);
        float output_sum = 0.0;
        for (uint i = lane; i < p.n0; i += 128u) {
            float updated = uintBitsToFloat(w[state_base + i]) * decay + delta * a[key_base + i];
            c[state_base + i] = checked(updated);
            output_sum += updated * a[query_base + i];
        }
        output_sum = sum_group(output_sum) * inversesqrt(float(p.n0));
        if (lane == 0u) d[row] = checked(output_sum);

    } else if (p.op == 114u) { // Fused Delta RMSNorm and sigmoid gate.
        // n0=dv,n1=value heads,f0=epsilon; w=shared norm weights,a=raw output,b=z,c=result.
        if (group >= p.n1) return;
        uint base = group * p.n0;
        float square = 0.0;
        for (uint i = lane; i < p.n0; i += 128u) square += a[base + i] * a[base + i];
        float inverse = inversesqrt(checked(sum_group(square)) / float(p.n0) + p.f0);
        barrier();
        for (uint i = lane; i < p.n0; i += 128u) {
            c[base + i] = checked(a[base + i] * inverse * uintBitsToFloat(w[i]) * qwen_sigmoid(b[base + i]));
        }

    } else if (p.op == 115u) { // Optional combined BA projection: n0=nv,n1=nk.
        // a=[group beta rows | group alpha rows], c=beta, d=alpha.
        if (element >= p.n0) return;
        uint per_group = p.n0 / p.n1;
        uint offset = (element / per_group) * 2u * per_group + element % per_group;
        c[element] = checked(a[offset]);
        d[element] = checked(a[offset + per_group]);

    } else if (p.op == 120u) { // PLE gate: n0=model width,n1=streams; group=stream.
        // a=normalized key, b=normalized query, e=shared value vector, c=gated streams.
        if (group >= p.n1) return;
        uint base = group * p.n0;
        float partial = 0.0;
        for (uint i = lane; i < p.n0; i += 128u) partial += a[base + i] * b[base + i];
        float score = checked(sum_group(partial) * inversesqrt(float(p.n0)));
        float magnitude = sqrt(max(abs(score), 1e-6));
        float gate = qwen_sigmoid(score > 0.0 ? magnitude : (score < 0.0 ? -magnitude : 0.0));
        barrier();
        for (uint i = lane; i < p.n0; i += 128u) c[base + i] = checked(e[i] * gate);

    } else if (p.op == 121u) { // PLE dilated four-tap convolution; n0=stream width.
        // w=[channel][4], a=normalized gated streams,b=old9-row history,c=new history,
        // d=in-place residual streams,e=gated streams. b/c may alias. The old
        // taps at ages9,6,3 are read before that channel's forward shift.
        if (element >= p.n0) return;
        float convolution = uintBitsToFloat(w[element * 4u + 3u]) * a[element];
        for (uint tap = 0u; tap < 3u; ++tap) {
            convolution += uintBitsToFloat(w[element * 4u + tap]) * b[tap * 3u * p.n0 + element];
        }
        d[element] = checked(d[element] + e[element] + qwen_silu(convolution));
        for (uint age = 0u; age < 8u; ++age) c[age * p.n0 + element] = b[(age + 1u) * p.n0 + element];
        c[8u * p.n0 + element] = checked(a[element]);

    } else if (p.op == 130u) { // Cache a newly completed QSA pooled key, one group.
        // n0=indexer width,n1=block index,n2=rotary prefix,n3=pool4; f0=epsilon,f1=base.
        // w=K norm weights,a=raw keys,b=width-sized scratch,c=pooled keys. b is distinct.
        if (group != 0u) return;
        float square = 0.0;
        for (uint i = lane; i < p.n0; i += 128u) {
            float mean = 0.0;
            for (uint token = 0u; token < p.n3; ++token) mean += a[(p.n1 * p.n3 + token) * p.n0 + i] / float(p.n3);
            b[i] = mean;
            square += mean * mean;
        }
        float inverse = inversesqrt(checked(sum_group(square)) / float(p.n0) + p.f0);
        memoryBarrierBuffer(); barrier();
        for (uint i = lane; i < p.n0; i += 128u) b[i] = checked(b[i] * inverse * uintBitsToFloat(w[i]));
        memoryBarrierBuffer(); barrier();
        uint half_width = p.n2 / 2u;
        uint base = p.n1 * p.n0;
        for (uint i = lane; i < p.n0; i += 128u) {
            if (i < half_width) {
                float angle = float(p.n1 * p.n3) * pow(p.f1, -float(i) / float(half_width));
                float cosine = cos(angle), sine = sin(angle);
                c[base + i] = checked(b[i] * cosine - b[i + half_width] * sine);
                c[base + i + half_width] = checked(b[i + half_width] * cosine + b[i] * sine);
            } else if (i >= p.n2) {
                c[base + i] = checked(b[i]);
            }
        }

    } else if (p.op == 131u) { // QSA score: n0=indexer width,n1=heads,n2=complete blocks.
        // a=normalized/rotated Q,b=cached normalized/rotated pooled K,c=block scores.
        if (group >= p.n2) return;
        float score = 0.0;
        for (uint head = 0u; head < p.n1; ++head) {
            float partial = 0.0;
            for (uint i = lane; i < p.n0; i += 128u) partial += a[head * p.n0 + i] * b[group * p.n0 + i];
            float dot_product = checked(sum_group(partial));
            score += max(dot_product, 0.0);
            barrier();
        }
        if (lane == 0u) c[group] = checked(score * inversesqrt(float(p.n0)));

    } else if (p.op == 132u) { // Exact GPU top-k blocks and compact attended token list.
        // a=block scores,c=block mask,d=[count, token IDs...]. n0=complete blocks,
        // n1=kept blocks,n2=current token count,n3=cell capacity (budget+3), WG1.
        // Nonnegative F32 score bits preserve order. Parallel radix selection
        // takes 31 bounded passes, instead of quadratic rank tests or O(N*k)
        // insertion sorting. Equal scores choose lower block IDs deterministically.
        if (group != 0u) return;
        if (p.n0 <= p.n1) {
            if (p.n2 > p.n3) { if (lane == 0u) invalid_value(); return; }
            for (uint block = lane; block < p.n0; block += 128u) c[block] = 1.0;
            for (uint token = lane; token < p.n2; token += 128u) d[1u + token] = float(token);
            if (lane == 0u) d[0] = float(p.n2);
            return;
        }
        uint prefix = 0u, mask = 0u, remaining = p.n1;
        for (int bit = 30; bit >= 0; --bit) {
            uint current_bit = 1u << uint(bit);
            float count = 0.0;
            for (uint block = lane; block < p.n0; block += 128u) {
                float score = checked(a[block]);
                uint bits = qwen_score_bits(score);
                if ((bits & mask) == prefix && (bits & current_bit) != 0u) count += 1.0;
            }
            uint ones = uint(sum_group(count));
            barrier();
            if (ones >= remaining) prefix |= current_bit;
            else remaining -= ones;
            mask |= current_bit;
        }
        if (lane == 0u) {
            uint ties = 0u, last = 0u;
            for (uint block = 0u; block < p.n0; ++block) {
                if (qwen_score_bits(a[block]) == prefix) {
                    ++ties;
                    if (ties == remaining) { last = block; break; }
                }
            }
            reduction[0] = float(last);
        }
        barrier();
        uint last_tie = uint(reduction[0]);
        float cutoff = uintBitsToFloat(prefix);
        for (uint block = lane; block < p.n0; block += 128u) {
            float score = a[block];
            c[block] = score > cutoff || (score == cutoff && block <= last_tie) ? 1.0 : 0.0;
        }
        memoryBarrierBuffer(); barrier();
        if (lane == 0u) {
            uint count = 0u;
            for (uint block = 0u; block < p.n0; ++block) if (c[block] != 0.0) {
                if (count + 4u > p.n3) { invalid_value(); d[0] = 0.0; return; }
                for (uint offset = 0u; offset < 4u; ++offset) d[1u + count++] = float(block * 4u + offset);
            }
            for (uint token = p.n0 * 4u; token < p.n2; ++token) {
                if (count >= p.n3) { invalid_value(); d[0] = 0.0; return; }
                d[1u + count++] = float(token);
            }
            d[0] = float(count);
        }

    } else if (p.op == 133u) { // Sparse attention scores over compacted QSA cells.
        // a=Q,b=KV keys,c=scores,e=cell list. n0=cell capacity,n1=Q heads,n2=KV heads,
        // n3=head width,n4=score row stride,n5=initialized KV token count.
        // Dispatch x=cell capacity,y=Q heads; inactive cell workgroups return.
        uint slot = gl_WorkGroupID.x, head = gl_WorkGroupID.y;
        float stored_count = e[0];
        if (isnan(stored_count) || isinf(stored_count) || stored_count < 0.0 || stored_count > float(p.n0)) {
            if (lane == 0u) invalid_value();
            return;
        }
        uint count = uint(stored_count);
        if (head >= p.n1 || slot >= count) return;
        if (count > p.n0 || isnan(e[slot + 1u]) || isinf(e[slot + 1u])
            || e[slot + 1u] < 0.0 || e[slot + 1u] >= float(p.n5)) {
            if (lane == 0u) c[head * p.n4 + slot] = invalid_value();
            return;
        }
        uint token = uint(e[slot + 1u]);
        uint key_head = head / (p.n1 / p.n2);
        float partial = 0.0;
        uint start = (token * p.n2 + key_head) * p.n3;
        for (uint i = lane; i < p.n3; i += 128u) partial += a[head * p.n3 + i] * b[start + i];
        float score = sum_group(partial) * inversesqrt(float(p.n3));
        if (lane == 0u) c[head * p.n4 + slot] = checked(score);

    } else if (p.op == 134u) { // Sparse softmax/value sum over the actual compact list.
        // a=scores,b=KV values,c=attention,e=cell list; dimensions as op133.
        // Dispatch one group per query head; complexity depends on QSA budget,
        // not the complete context length.
        uint head = group;
        if (head >= p.n1) return;
        float stored_count = e[0];
        if (isnan(stored_count) || isinf(stored_count) || stored_count < 0.0 || stored_count > float(p.n0)) {
            if (lane == 0u) invalid_value();
            return;
        }
        uint count = uint(stored_count);
        if (count == 0u || count > p.n0) { if (lane == 0u) invalid_value(); return; }
        float bad_cells = 0.0;
        for (uint slot = lane; slot < count; slot += 128u) {
            float token = e[slot + 1u];
            if (isnan(token) || isinf(token) || token < 0.0 || token >= float(p.n5)) bad_cells += 1.0;
        }
        bad_cells = sum_group(bad_cells);
        barrier();
        if (bad_cells != 0.0) { if (lane == 0u) invalid_value(); return; }
        float maximum = -3.402823466e38;
        for (uint slot = lane; slot < count; slot += 128u) maximum = max(maximum, a[head * p.n4 + slot]);
        maximum = max_group(maximum);
        barrier();
        float partial = 0.0;
        for (uint slot = lane; slot < count; slot += 128u) partial += exp(a[head * p.n4 + slot] - maximum);
        float denominator = checked(sum_group(partial));
        barrier();
        uint value_head = head / (p.n1 / p.n2);
        for (uint i = lane; i < p.n3; i += 128u) {
            float total = 0.0;
            for (uint slot = 0u; slot < count; ++slot) {
                uint token = uint(e[slot + 1u]);
                float probability = exp(a[head * p.n4 + slot] - maximum) / denominator;
                total += probability * b[(token * p.n2 + value_head) * p.n3 + i];
            }
            c[head * p.n3 + i] = checked(total);
        }

    } else {
        if (lane == 0u) invalid_value();
    }
}
