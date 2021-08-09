#pragma once

#include "CudaVecBase.cuh"

namespace Cuda
{
	template<>
	struct __align__(8) __vec_swizzle<float, 2, 2, 0, 1>
	{
		enum _attrs : size_t { kDims = 2 };
		using kType = float;
		union  
		{
			struct { float x, y; };
			struct { float i0, i1; };
			float data[2];

			__vec_swizzle<float, 2, 2, 0, 0> xx;
			__vec_swizzle<float, 2, 2, 1, 0> yx;
			__vec_swizzle<float, 2, 2, 1, 1> yy;
		};

        __vec_swizzle() = default;
        __vec_swizzle(const __vec_swizzle&) = default;
        __host__ __device__ __forceinline__ explicit __vec_swizzle(const float v) : x(v), y(v) {}
        __host__ __device__ __forceinline__ __vec_swizzle(const float& x_, const float& y_) : x(x_), y(y_) {}

        // Cast from other vec2 types
        template<typename OtherType, int OtherSize, int I0, int I1>
        __host__ __device__ __forceinline__ explicit __vec_swizzle(const __vec_swizzle<OtherType, OtherSize, 2, I0, I1>& v) :
            x(float(v.data[I0])), y(float(v.data[I1])) {}

        template<int L0, int L1, int R0, int R1>
        __host__ __device__ __forceinline__ void UnpackTo(float* otherData) const
        {
            otherData[L0] = data[0];
            otherData[L1] = data[1];
        }

        // Cast from swizzled types
        template<int ActualSize, int... In>
        __host__ __device__ __forceinline__ __vec_swizzle(const __vec_swizzle<float, ActualSize, 2, In...>& swizzled)
        {
            swizzled.UnpackTo<0, 1, In...>(data);
        }

        // Assign from swizzled types
        template<int RS, int R0, int R1>
        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
        {
            x = rhs.other[R0]; y = rhs.other[R1];
            return *this;
        }

        // Assign from arithmetic types
        template<typename OtherType, typename = typename std::enable_if<std::is_arithmetic<OtherType>::value>::type>
        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const OtherType& rhs)
        {
            x = float(rhs); y = float(rhs);
            return *this;
        }

        __host__ __device__ __vec_swizzle& operator=(const float& v) { x = v; y = v; return *this; }

        __host__ __device__ __forceinline__ const float& operator[](const unsigned int idx) const { return data[idx]; }
        __host__ __device__ __forceinline__ float& operator[](const unsigned int idx) { return data[idx]; }

        __host__ inline std::string format() const { return tfm::format("{%.10f, %.10f}", x, y); }
    };

    using vec2 = __vec_swizzle<float, 2, 2, 0, 1>;

    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 operator +(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] + rhs.data[R0], lhs.data[L1] + rhs.data[R1] };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 operator +(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const float& rhs)
    {
        return { lhs.data[L0] + rhs, lhs.data[L1] + rhs };
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 operator -(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] - rhs.data[R0], lhs.data[L1] - rhs.data[R1] };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 operator -(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const float& rhs)
    {
        return { lhs.data[L0] - rhs, lhs.data[L1] - rhs };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 operator -(const __vec_swizzle<float, LS, 2, L0, L1>& lhs)
    {
        return { -lhs.data[L0], -lhs.data[L1] };
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 operator *(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] * rhs.data[R0], lhs.data[L1] * rhs.data[R1] };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 operator *(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const float& rhs)
    {
        return { lhs.data[L0] * rhs, lhs.data[L1] * rhs };
    }
    template<int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 operator *(const float& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        return { lhs * rhs.data[R0], lhs * rhs.data[R1] };
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 operator /(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] / rhs.data[R0], lhs.data[L1] / rhs.data[R1] };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 operator /(const __vec_swizzle<float, LS, 2, L0, L1>& lhs, const float& rhs)
    {
        return { lhs.data[L0] / rhs, lhs.data[L1] / rhs };
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1>& operator +=(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] += rhs.data[R0]; lhs.data[L1] += rhs.data[R1];
        return lhs;
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1>& operator -=(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] -= rhs.data[R0]; lhs.data[L1] -= rhs.data[R1];
        return lhs;
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1>& operator *=(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] *= rhs.data[R0]; lhs.data[L1] *= rhs.data[R1];
        return lhs;
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1>& operator *=(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const float& rhs)
    {
        lhs.data[L0] *= rhs; lhs.data[L1] *= rhs;
        return lhs;
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1>& operator /=(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] /= rhs.data[R0]; lhs.data[L1] /= rhs.data[R1];
        return lhs;
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1>& operator /=(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const float& rhs)
    {
        lhs.data[L0] /= rhs; lhs.data[L1] /= rhs;
        return lhs;
    }

    // Vector functions
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ float dot(__vec_swizzle<float, LS, 2, L0, L1>& lhs, const __vec_swizzle<float, RS, 2, R0, R1>& rhs)
    {
        return lhs.data[L0] * rhs.data[R0] + lhs.data[L1] * rhs.data[R1];
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 perpendicular(__vec_swizzle<float, LS, 2, L0, L1>& lhs)
    {
        return { -lhs.data[L1], lhs.data[L0] };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ float length2(const __vec_swizzle<float, LS, 2, L0, L1>& v)
    {
        return v.data[L0] * v.data[L0] + v.data[L1] * v.data[L1];
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ float length(const __vec_swizzle<float, LS, 2, L0, L1>& v)
    {
        return sqrt(v.data[L0] * v.data[L0] + v.data[L1] * v.data[L1]);
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 2, L0, L1> normalize(const __vec_swizzle<float, LS, 2, L0, L1>& v)
    {
        return v / length(v);
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 fmod(__vec_swizzle<float, LS, 2, L0, L1>& a, const __vec_swizzle<float, RS, 2, R0, R1>& b)
    {
        return { fmodf(a.data[L0], b.data[R0]), fmodf(a.data[L1], b.data[R1]) };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 fmod(__vec_swizzle<float, LS, 2, L0, L1>& a, const float& b)
    {
        return { fmodf(a.data[L0], b), fmodf(a.data[L1], b) };
    }
    template<int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ __forceinline__ vec2 pow(__vec_swizzle<float, LS, 2, L0, L1>& a, const __vec_swizzle<float, RS, 2, R0, R1>& b)
    {
        return { powf(a.data[L0], b.data[R0]), powf(a.data[L1], b.data[R1]) };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 exp(__vec_swizzle<float, LS, 2, L0, L1>& a)
    {
        return { exp(a.data[L0]), exp(a.data[L1]) };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 log(__vec_swizzle<float, LS, 2, L0, L1>& a)
    {
        return { logf(a.data[L0]), logf(a.data[L1]) };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 log10(__vec_swizzle<float, LS, 2, L0, L1>& a)
    {
        return { log10f(a.data[L0]), log10f(a.data[L1]) };
    }
    template<int LS, int L0, int L1>
    __host__ __device__ __forceinline__ vec2 log2(__vec_swizzle<float, LS, 2, L0, L1>& a)
    {
        return { log2f(a.data[L0]), log2f(a.data[L1]) };
    }

    __host__ __device__ __forceinline__ vec2 clamp(const vec2& v, const vec2& a, const vec2& b) { return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y) }; }
    __host__ __device__ __forceinline__ vec2 saturate(const vec2& v, const vec2& a, const vec2& b) { return { clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f) }; }
    __host__ __device__ __forceinline__ vec2 abs(const vec2& a) { return { fabs(a.x), fabs(a.y) }; }
    __host__ __device__ __forceinline__ float sum(const vec2& a) { return a.x + a.y; }
    __host__ __device__ __forceinline__ vec2 ceil(const vec2& v) { return { ceilf(v.x), ceilf(v.y) }; }
    __host__ __device__ __forceinline__ vec2 floor(const vec2& v) { return { floorf(v.x), floorf(v.y) }; }
    __host__ __device__ __forceinline__ vec2 sign(const vec2& v) { return { sign(v.x), sign(v.y) }; }

    __host__ __device__ __forceinline__ float cwiseMax(const vec2& v) { return (v.x > v.y) ? v.x : v.y; }
    __host__ __device__ __forceinline__ float cwiseMin(const vec2& v) { return (v.x < v.y) ? v.x : v.y; }

    __host__ __device__ __forceinline__ bool operator==(const vec2& lhs, const vec2& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
    __host__ __device__ __forceinline__ bool operator!=(const vec2& lhs, const vec2& rhs) { return lhs.x != rhs.x || lhs.y != rhs.y; }

    __host__ __device__ __forceinline__ vec2 max(const vec2& a, const vec2& b) { return vec2(max(a.x, b.x), max(a.y, b.y)); }
    __host__ __device__ __forceinline__ vec2 min(const vec2& a, const vec2& b) { return vec2(min(a.x, b.x), min(a.y, b.y)); }

    // FIXME: Cuda intrinsics aren't working. Why is this?
    //__host__ __device__ __forceinline__ vec2 saturate(const vec2& v, const vec2& a, const vec2& b)	{ return { __saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }}

    template<typename T>
    __host__ __device__ __forceinline__ T cast(const vec2& v) { T r; r.x = v.x; r.y = v.y; return r; }

}