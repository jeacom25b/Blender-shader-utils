#define SHARED sha##red
SHARED uint reduction_buffer_[COMPUTE_TOTAL_INVOCATIONS_];

#define ROP_ADD_  0
#define ROP_MUL_  1
#define ROP_MAX_  2
#define ROP_MIN_  3
#define ROP_AND_  4
#define ROP_OR_   5
#define ROP_XOR_  6
#define ROP_FADD_ 7
#define ROP_FMUL_ 8
#define ROP_FMAX_ 9
#define ROP_FMIN_ 10

uint reduction_op_(uint a, uint b, const uint ROP){
    switch (ROP){
        case ROP_ADD_:
            return a + b;
        case ROP_MUL_:
            return a * b;
        case ROP_MAX_:
            return max(a, b);
        case ROP_MIN_:
            return min(a, b);
        case ROP_AND_:
            return a & b;
        case ROP_OR_:
            return a | b;
        case ROP_XOR_:
            return a ^ b;
        case ROP_FADD_:
            return floatBitsToUint(uintBitsToFloat(a) + uintBitsToFloat(b));
        case ROP_FMUL_:
            return floatBitsToUint(uintBitsToFloat(a) * uintBitsToFloat(b));
        case ROP_FMAX_:
            return floatBitsToUint(max(uintBitsToFloat(a), uintBitsToFloat(b)));
        case ROP_FMIN_:
            return floatBitsToUint(min(uintBitsToFloat(a), uintBitsToFloat(b)));
        default:
            return 0;
    }
}

uint parallel_reduction_impl_(uint x, const uint ROP){
    uint idx = gl_LocalInvocationIndex;
    reduction_buffer_[idx] = x;
    barrier();

    uint N = COMPUTE_TOTAL_INVOCATIONS_;
    uint pot = 1u << findMSB(N);

    if (idx < N - pot){
        reduction_buffer_[idx] = reduction_op_(reduction_buffer_[idx], reduction_buffer_[idx + pot], ROP);
    }
    barrier();

    uint stride = pot / 2;

    while (stride > 0){
        if (idx < stride){
            reduction_buffer_[idx] = reduction_op_(reduction_buffer_[idx], reduction_buffer_[idx + stride], ROP);
        }
        barrier();
        stride >>= 1;
    }
    return reduction_buffer_[0];
}

uint parallel_add(uint x){
    return parallel_reduction_impl_(x, ROP_ADD_);
}
uint parallel_mul(uint x){
    return parallel_reduction_impl_(x, ROP_MUL_);
}
uint parallel_max(uint x){
    return parallel_reduction_impl_(x, ROP_MAX_);
}
uint parallel_min(uint x){
    return parallel_reduction_impl_(x, ROP_MIN_);
}
uint parallel_and(uint x){
    return parallel_reduction_impl_(x, ROP_AND_);
}
uint parallel_or(uint x){
    return parallel_reduction_impl_(x, ROP_OR_);
}
uint parallel_xor(uint x){
    return parallel_reduction_impl_(x, ROP_XOR_);
}
float parallel_add(float x){
    return uintBitsToFloat(parallel_reduction_impl_(floatBitsToUint(x), ROP_FADD_));
}
float parallel_mul(float x){
    return uintBitsToFloat(parallel_reduction_impl_(floatBitsToUint(x), ROP_FMUL_));
}
float parallel_max(float x){
    return uintBitsToFloat(parallel_reduction_impl_(floatBitsToUint(x), ROP_FMAX_));
}
float parallel_min(float x){
    return uintBitsToFloat(parallel_reduction_impl_(floatBitsToUint(x), ROP_FMIN_));
}