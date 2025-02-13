#pragma once

// Arithmetic operator with left-hand scalar to vec4
#define ARITHMETIC_OPERATOR_SCALAR_VEC(op) \
    template<int R0, int R1, int R2, int R3> \
    __host__ __device__ __forceinline__ vec4 operator ##op(const float lhs, __vec_swizzle<float, 4, 4, R0, R1, R2, R3>& rhs) \
    { \
        return vec4(lhs ##op rhs[R0], lhs ##op rhs[R1], lhs ##op rhs[R2], lhs ##op rhs[R3]); \
    }