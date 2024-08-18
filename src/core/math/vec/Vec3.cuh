#pragma once

#include "VecBase.cuh"
#include "Vec2.cuh"

namespace Enso
{		
    // Full specialisation of __vec_swizzle for vec3
	template<>
	struct __vec_swizzle<float, 3, 3, 0, 1, 2>
	{
		enum _attrs : size_t { kDims = 3 };
		using kType = float;

		union
		{
			struct { float x, y, z; };
			struct { float i0, i1, i2; };
			float data[3];

			__vec_swizzle<float, 3, 3, 0, 0, 0> xxx; /*__vec_swizzle<float, 3, 3, 0, 0, 1> xxy;*/ /*__vec_swizzle<float, 3, 3, 0, 0, 2> xxz;*/
			/*__vec_swizzle<float, 3, 3, 0, 1, 0> xyx;*/ /*__vec_swizzle<float, 3, 3, 0, 1, 1> xyy;*/ 
			/*__vec_swizzle<float, 3, 3, 0, 2, 0> xzx;*/ /*__vec_swizzle<float, 3, 3, 0, 2, 1> xzy;*/ /*__vec_swizzle<float, 3, 3, 0, 2, 2> xzz;*/
			/*__vec_swizzle<float, 3, 3, 1, 0, 0> yxx;*/ /*__vec_swizzle<float, 3, 3, 1, 0, 1> yxy;*/ /*__vec_swizzle<float, 3, 3, 1, 0, 2> yxz;*/
			/*__vec_swizzle<float, 3, 3, 1, 1, 0> yyx;*/ /*__vec_swizzle<float, 3, 3, 1, 1, 1> yyy;*/ __vec_swizzle<float, 3, 3, 1, 1, 2> yyz;
			__vec_swizzle<float, 3, 3, 1, 2, 0> yzx; /*__vec_swizzle<float, 3, 3, 1, 2, 1> yzy;*/ /*__vec_swizzle<float, 3, 3, 1, 2, 2> yzz;*/
			/*__vec_swizzle<float, 3, 3, 2, 0, 0> zxx;*/ /*__vec_swizzle<float, 3, 3, 2, 0, 1> zxy;*/ /*__vec_swizzle<float, 3, 3, 2, 0, 2> zxz;*/
			__vec_swizzle<float, 3, 3, 2, 1, 0> zyx; /*__vec_swizzle<float, 3, 3, 2, 1, 1> zyy;*/ /*__vec_swizzle<float, 3, 3, 2, 1, 2> zyz;*/
			/*__vec_swizzle<float, 3, 3, 2, 2, 0> zzx;*/ /*__vec_swizzle<float, 3, 3, 2, 2, 1> zzy;*/ /*__vec_swizzle<float, 3, 3, 2, 2, 2> zzz;*/

			/*__vec_swizzle<float, 3, 2, 0, 0> xx;*/ __vec_swizzle<float, 3, 2, 0, 1> xy; __vec_swizzle<float, 3, 2, 0, 2> xz;
			__vec_swizzle<float, 3, 2, 1, 0> yx; /*__vec_swizzle<float, 3, 2, 1, 1> yy;*/ __vec_swizzle<float, 3, 2, 1, 2> yz;
			__vec_swizzle<float, 3, 2, 2, 0> zx; __vec_swizzle<float, 3, 2, 2, 1> zy; /*__vec_swizzle<float, 3, 2, 2, 2> zz;*/
		};

        __host__ __device__ __vec_swizzle() {}
        __host__ __device__ __forceinline__ explicit __vec_swizzle(const float v) : x(v), y(v), z(v) {}
        __host__ __device__ __forceinline__ __vec_swizzle(const float& x_, const float& y_, const float& z_) : x(x_), y(y_), z(z_) {}
        __host__ __device__ __forceinline__ __vec_swizzle(const vec2& v, const float& z_) : x(v.x), y(v.y), z(z_) {}
        __host__ __device__ __forceinline__ __vec_swizzle(const float& x_, const vec2& v) : x(x_), y(v.x), z(v.y) {}

        // Cast from other vec3 types
        template<typename OtherType, int OtherSize, int I0, int I1, int I2, int... In>
        __host__ __device__ __vec_swizzle(const __vec_swizzle<OtherType, OtherSize, 3, I0, I1, I2, In...>& v) :
            x(float(v.data[I0])), y(float(v.data[I1])), z(float(v.data[I2])) {}

        template<int L0, int L1, int L2, int R0, int R1, int R2>
        __host__ __device__ __forceinline__ void UnpackTo(float* otherData) const
        {
            otherData[L0] = data[R0];
            otherData[L1] = data[R1];
            otherData[L2] = data[R2];
        }

        // Cast from swizzled types
        /*template<int ActualSize, int... In>
        __host__ __device__ __forceinline__ __vec_swizzle(const __vec_swizzle<float, ActualSize, 3, In...>& swizzled)
        {
            swizzled.UnpackTo<0, 1, 2, In...>(data);
        }*/

        __host__ __device__ __forceinline__ __vec_swizzle& operator+=(const __vec_swizzle& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator-=(const __vec_swizzle& rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator*=(const __vec_swizzle& rhs) { x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator/=(const __vec_swizzle& rhs) { x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator+=(const float& rhs) { x += rhs; y += rhs; z += rhs; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator-=(const float& rhs) { x -= rhs; y -= rhs; z -= rhs; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator*=(const float& rhs) { x *= rhs; y *= rhs; z *= rhs; return *this; }
        __host__ __device__ __forceinline__ __vec_swizzle& operator/=(const float& rhs) { x /= rhs; y /= rhs; z /= rhs; return *this; }

        // Assign from swizzled types
        template<int RS, int R0, int R1, int R2>
        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
        {
            x = rhs.other[R0]; y = rhs.other[R1]; z = rhs.other[R2];
            return *this;
        }

        // Assign from arithmetic types
        template<typename OtherType, typename = typename std::enable_if<std::is_arithmetic<OtherType>::value>::type>
        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const OtherType& rhs)
        {
            x = float(rhs); y = float(rhs); z = float(rhs); 
            return *this;
        }

        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const float& v) { x = v; y = v; z = v; return *this; }

        __host__ __device__ __forceinline__ const float& operator[](const unsigned int idx) const { return data[idx]; }
        __host__ __device__ __forceinline__ float& operator[](const unsigned int idx) { return data[idx]; }

        __host__ inline std::string format() const { return tfm::format("{%.10f, %.10f, %.10f}", x, y, z); }
    };

    // Alias vec3 to the linear triple
    using vec3 = __vec_swizzle<float, 3, 3, 0, 1, 2>;

    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 operator +(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] + rhs.data[R0], lhs.data[L1] + rhs.data[R1], lhs.data[L2] + rhs.data[R2] };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 operator +(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const float& rhs)
    {
        return { lhs.data[L0] + rhs, lhs.data[L1] + rhs, lhs.data[L2] + rhs };
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 operator -(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] - rhs.data[R0], lhs.data[L1] - rhs.data[R1], lhs.data[L2] - rhs.data[R2] };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 operator -(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const float& rhs)
    {
        return { lhs.data[L0] - rhs, lhs.data[L1] - rhs, lhs.data[L2] - rhs };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 operator -(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs)
    {
        return { -lhs.data[L0], -lhs.data[L1], -lhs.data[L2] };
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 operator *(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] * rhs.data[R0], lhs.data[L1] * rhs.data[R1], lhs.data[L2] * rhs.data[R2] };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 operator *(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const float& rhs)
    {
        return { lhs.data[L0] * rhs, lhs.data[L1] * rhs, lhs.data[L2] * rhs };
    }
    template<int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 operator *(const float& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs * rhs.data[R0], lhs * rhs.data[R1], lhs * rhs.data[R2] };
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 operator /(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] / rhs.data[R0], lhs.data[L1] / rhs.data[R1], lhs.data[L2] / rhs.data[R2] };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 operator /(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const float& rhs)
    {
        return { lhs.data[L0] / rhs, lhs.data[L1] / rhs, lhs.data[L2] / rhs };
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2>& operator +=(__vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] += rhs.data[R0]; lhs.data[L1] += rhs.data[R1]; lhs.data[L2] += rhs.data[R2];
        return lhs;
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2>& operator -=(__vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] -= rhs.data[R0]; lhs.data[L1] -= rhs.data[R1]; lhs.data[L2] -= rhs.data[R2];
        return lhs;
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2>& operator *=(__vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] *= rhs.data[R0]; lhs.data[L1] *= rhs.data[R1]; lhs.data[L2] *= rhs.data[R2];
        return lhs;
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2>& operator *=(__vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const float& rhs)
    {
        lhs.data[L0] *= rhs; lhs.data[L1] *= rhs; lhs.data[L2] *= rhs;
        return lhs;
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2>& operator /=(__vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] /= rhs.data[R0]; lhs.data[L1] /= rhs.data[R1]; lhs.data[L2] /= rhs.data[R2];
        return lhs;
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2>& operator /=(__vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const float& rhs)
    {
        lhs.data[L0] /= rhs; lhs.data[L1] /= rhs; lhs.data[L2] /= rhs;
        return lhs;
    }

    // Vector functions
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ float dot(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return lhs.data[L0] * rhs.data[R0] + lhs.data[L1] * rhs.data[R1] + lhs.data[L2] * rhs.data[R2];
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 cross(const __vec_swizzle<float, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<float, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L1] * rhs.data[R2] - lhs.data[L2] * rhs.data[R1], lhs.data[L2] * rhs.data[R0] - lhs.data[L0] * rhs.data[R2], lhs.data[L0] * rhs.data[R1] - lhs.data[L1] * rhs.data[R0] };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ float length2(const __vec_swizzle<float, LS, 3, L0, L1, L2>& v)
    {
        return v.data[L0] * v.data[L0] + v.data[L1] * v.data[L1] + v.data[L2] * v.data[L2];
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ float length(const __vec_swizzle<float, LS, 3, L0, L1, L2>& v)
    {
        return sqrtf(v.data[L0] * v.data[L0] + v.data[L1] * v.data[L1] + v.data[L2] * v.data[L2]);
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __vec_swizzle<float, LS, 3, L0, L1, L2> normalize(const __vec_swizzle<float, LS, 3, L0, L1, L2>& v)
    {
        return v / length(v);
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 fmod(__vec_swizzle<float, LS, 3, L0, L1, L2>& a, const __vec_swizzle<float, RS, 3, R0, R1, R2>& b)
    {
        return { fmodf(a.data[L0], b.data[R0]), fmodf(a.data[L1], b.data[R1]), fmodf(a.data[L2], b.data[R2]) };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 fmod(__vec_swizzle<float, LS, 3, L0, L1, L2>& a, const float& b)
    {
        return { fmodf(a.data[L0], b), fmodf(a.data[L1], b), fmodf(a.data[L2], b) };
    }
    template<int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ vec3 pow(__vec_swizzle<float, LS, 3, L0, L1, L2>& a, const __vec_swizzle<float, RS, 3, R0, R1, R2>& b)
    {
        return { powf(a.data[L0], b.data[R0]), powf(a.data[L1], b.data[R1]), powf(a.data[L2], b.data[R2]) };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 exp(__vec_swizzle<float, LS, 3, L0, L1, L2>& a)
    {
        return { expf(a.data[L0]), expf(a.data[L1]), expf(a.data[L2]) };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 log(__vec_swizzle<float, LS, 3, L0, L1, L2>& a)
    {
        return { logf(a.data[L0]), logf(a.data[L1]), logf(a.data[L2]) };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 log10(__vec_swizzle<float, LS, 3, L0, L1, L2>& a)
    {
        return { mag10f(a.data[L0]), mag10f(a.data[L1]), mag10f(a.data[L2]) };
    }
    template<int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ vec3 log2(__vec_swizzle<float, LS, 3, L0, L1, L2>& a)
    {
        return { mog2f(a.data[L0]), mog2f(a.data[L1]), mog2f(a.data[L2]) };
    }

    __host__ __device__ __forceinline__ vec3 clamp(const vec3& v, const vec3& a, const vec3& b) { return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z) }; }
    __host__ __device__ __forceinline__ vec3 saturate(const vec3& v) { return { clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f), clamp(v.z, 0.0f, 1.0f) }; }
    __host__ __device__ __forceinline__ vec3 abs(const vec3& a) { return { fabs(a.x), fabs(a.y), fabs(a.z) }; }
    __host__ __device__ __forceinline__ float sum(const vec3& a) { return a.x + a.y + a.z; }
    __host__ __device__ __forceinline__ vec3 ceil(const vec3& v) { return { ceilf(v.x), ceilf(v.y), ceilf(v.z) }; }
    __host__ __device__ __forceinline__ vec3 floor(const vec3& v) { return { floorf(v.x), floorf(v.y), floorf(v.z) }; }
    __host__ __device__ __forceinline__ vec3 sign(const vec3& v) { return { sign(v.x), sign(v.y), sign(v.z) }; }
    __host__ __device__ __forceinline__ vec3 fract(const vec3& v) { return { fmodf(v.x, 1.0f), fmodf(v.y, 1.0f), fmodf(v.z, 1.0f) }; }
    __host__ __device__ __forceinline__ vec3 mod(const vec3& a, const vec3& b) { return { fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z) }; }

    __host__ __device__ __forceinline__ bool operator==(const vec3& lhs, const vec3& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; }
    __host__ __device__ __forceinline__ bool operator!=(const vec3& lhs, const vec3& rhs) { return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z; }

    __host__ __device__ __forceinline__ float cwiseMax(const vec3& v) { return (v.x > v.y) ? ((v.x > v.z) ? v.x : v.z) : ((v.y > v.z) ? v.y : v.z); }
    __host__ __device__ __forceinline__ float cwiseMin(const vec3& v) { return (v.x < v.y) ? ((v.x < v.z) ? v.x : v.z) : ((v.y < v.z) ? v.y : v.z); }
    __host__ __device__ __forceinline__ float cwiseExtremum(const vec3& v)
    {
        const float high = cwiseMax(v);
        const float low = cwiseMin(v);
        return (fabs(high) > fabs(low)) ? high : low;
    }

    __host__ __device__ __forceinline__ vec3 max(const vec3& a, const vec3& b) { return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
    __host__ __device__ __forceinline__ vec3 min(const vec3& a, const vec3& b) { return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }

    // FIXME: Cuda intrinsics aren't working. Why is this?
    //__host__ __device__ __forceinline__ vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return { __saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }}

    template<typename T>
    __host__ __device__ __forceinline__ T cast(const vec3& v) { T r; r.x = v.x; r.y = v.y; r.z = v.z; return r; }
}