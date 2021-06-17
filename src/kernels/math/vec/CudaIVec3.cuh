#pragma once

#include "CudaVecBase.cuh"

namespace Cuda
{
	template<typename Type>
	struct __vec_swizzle<Type, 3, 3, 0, 1, 2>
	{
		enum _attrs : size_t { kDims = 3 };
		using kType = Type;

		union
		{
			struct { Type x, y, z; };
			struct { Type i0, i1, i2; };
			Type data[3];

            /*__vec_swizzle<Type, 3, 3, 0, 0, 0> xxx;*/ /*__vec_swizzle<Type, 3, 3, 0, 0, 1> xxy;*/ /*__vec_swizzle<Type, 3, 3, 0, 0, 2> xxz;*/
            /*__vec_swizzle<Type, 3, 3, 0, 1, 0> xyx;*/ /*__vec_swizzle<Type, 3, 3, 0, 1, 1> xyy;*/
            /*__vec_swizzle<Type, 3, 3, 0, 2, 0> xzx;*/ /*__vec_swizzle<Type, 3, 3, 0, 2, 1> xzy;*/ /*__vec_swizzle<Type, 3, 3, 0, 2, 2> xzz;*/
            /*__vec_swizzle<Type, 3, 3, 1, 0, 0> yxx;*/ /*__vec_swizzle<Type, 3, 3, 1, 0, 1> yxy;*/ /*__vec_swizzle<Type, 3, 3, 1, 0, 2> yxz;*/
            /*__vec_swizzle<Type, 3, 3, 1, 1, 0> yyx;*/ /*__vec_swizzle<Type, 3, 3, 1, 1, 1> yyy;*/ /*__vec_swizzle<Type, 3, 3, 1, 1, 2> yyz;*/
            /*__vec_swizzle<Type, 3, 3, 1, 2, 0> yzx;*/ /*__vec_swizzle<Type, 3, 3, 1, 2, 1> yzy;*/ /*__vec_swizzle<Type, 3, 3, 1, 2, 2> yzz;*/
            /*__vec_swizzle<Type, 3, 3, 2, 0, 0> zxx;*/ /*__vec_swizzle<Type, 3, 3, 2, 0, 1> zxy;*/ /*__vec_swizzle<Type, 3, 3, 2, 0, 2> zxz;*/
            /*__vec_swizzle<Type, 3, 3, 2, 1, 0> zyx;*/ /*__vec_swizzle<Type, 3, 3, 2, 1, 1> zyy;*/ /*__vec_swizzle<Type, 3, 3, 2, 1, 2> zyz;*/
            /*__vec_swizzle<Type, 3, 3, 2, 2, 0> zzx;*/ /*__vec_swizzle<Type, 3, 3, 2, 2, 1> zzy;*/ /*__vec_swizzle<Type, 3, 3, 2, 2, 2> zzz;*/

            /*__vec_swizzle<Type, 3, 2, 0, 0> xx;*/ /*__vec_swizzle<Type, 3, 2, 0, 1> xy;*/ /*__vec_swizzle<Type, 3, 2, 0, 2> xz;*/
            /*__vec_swizzle<Type, 3, 2, 1, 0> yx;*/ /*__vec_swizzle<Type, 3, 2, 1, 1> yy;*/ /*__vec_swizzle<Type, 3, 2, 1, 2> yz;*/
            /*__vec_swizzle<Type, 3, 2, 2, 0> zx;*/ /*__vec_swizzle<Type, 3, 2, 2, 1> zy;*/ /*__vec_swizzle<Type, 3, 2, 2, 2> zz;*/
		};

        __host__ __device__ __forceinline__ __vec_swizzle() {}
        __host__ __device__ __forceinline__ __vec_swizzle(const __vec_swizzle&) = default;
        __host__ __device__ __forceinline__ explicit __vec_swizzle(const Type v) : x(v), y(v), z(v) {}
        __host__ __device__ __forceinline__ __vec_swizzle(const Type& x_, const Type& y_, const Type& z_) : x(x_), y(y_), z(z_) {}
        __host__ __device__ __forceinline__ __vec_swizzle(const __ivec2<Type>& v, const Type& z_) : x(v.x), y(v.y), z(z_) {}

        // Cast from other vec3 types
        template<typename OtherType, int OtherSize, int I0, int I1, int I2>
        __host__ __device__ explicit __vec_swizzle(const __vec_swizzle<OtherType, OtherSize, 3, I0, I1, I2>& v) :
            x(Type(v.data[I0])), y(Type(v.data[I1])), z(Type(v.data[I2])) {}

        template<int L0, int L1, int L2, int R0, int R1, int R2>
        __host__ __device__ __forceinline__ void UnpackTo(Type* otherData) const
        {
            otherData[L0] = data[0];
            otherData[L1] = data[1];
            otherData[L2] = data[2];
        }

        // Cast from swizzled types
        template<int ActualSize, int... In>
        __host__ __device__ __forceinline__ __vec_swizzle(const __vec_swizzle<Type, ActualSize, 3, In...>& swizzled)
        {
            swizzled.UnpackTo<0, 1, 2, In...>(data);
        }

        // Assign from swizzled types
        template<int RS, int R0, int R1, int R2>
        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
        {
            x = rhs.other[R0]; y = rhs.other[R1]; z = rhs.other[R2];
            return *this;
        }

        __host__ __device__ __forceinline__ __vec_swizzle& operator=(const Type& v) { x = v; y = v; z = v; return *this; }

        __host__ __device__ __forceinline__ const Type& operator[](const unsigned int idx) const { return data[idx]; }
        __host__ __device__ __forceinline__ Type& operator[](const unsigned int idx) { return data[idx]; }

        __host__ inline std::string format() const { return tfm::format("{%i, %i, %i}", x, y, z); }
    };

    // Alias vec3 to the linear triple
    template<typename Type> using __ivec3 = __vec_swizzle<Type, 3, 3, 0, 1, 2>;

    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator +(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] + rhs.data[R0], lhs.data[L1] + rhs.data[R1], lhs.data[L2] + rhs.data[R2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator +(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] + rhs, lhs.data[L1] + rhs, lhs.data[L2] + rhs };
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator -(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] - rhs.data[R0], lhs.data[L1] - rhs.data[R1], lhs.data[L2] - rhs.data[R2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator -(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] - rhs, lhs.data[L1] - rhs, lhs.data[L2] - rhs };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator -(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs)
    {
        return { -lhs.data[L0], -lhs.data[L1], -lhs.data[L2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator *(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] * rhs.data[R0], lhs.data[L1] * rhs.data[R1], lhs.data[L2] * rhs.data[R2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator *(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] * rhs, lhs.data[L1] * rhs, lhs.data[L2] * rhs };
    }
    template<typename Type, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator *(const Type& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs * rhs.data[L0], lhs * rhs.data[L1], lhs * rhs.data[L2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator /(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] / rhs.data[R0], lhs.data[L1] / rhs.data[R1], lhs.data[L2] / rhs.data[R2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator /(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] / rhs, lhs.data[L1] / rhs, lhs.data[L2] / rhs };
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<Type, LS, 3, L0, L1, L2>& operator +=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] += rhs.data[R0]; lhs.data[L1] += rhs.data[R1]; lhs.data[L2] += rhs.data[R2];
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<Type, LS, 3, L0, L1, L2>& operator -=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] -= rhs.data[R0]; lhs.data[L1] -= rhs.data[R1]; lhs.data[L2] -= rhs.data[R2];
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<Type, LS, 3, L0, L1, L2>& operator *=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] *= rhs.data[R0]; lhs.data[L1] *= rhs.data[R1]; lhs.data[L2] *= rhs.data[R2];
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __vec_swizzle<Type, LS, 3, L0, L1, L2>& operator *=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        lhs.data[L0] *= rhs; lhs.data[L1] *= rhs; lhs.data[L2] *= rhs;
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __vec_swizzle<Type, LS, 3, L0, L1, L2>& operator /=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] /= rhs.data[R0]; lhs.data[L1] /= rhs.data[R1]; lhs.data[L2] /= rhs.data[R2];
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __vec_swizzle<Type, LS, 3, L0, L1, L2>& operator /=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        lhs.data[L0] /= rhs; lhs.data[L1] /= rhs; lhs.data[L2] /= rhs;
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator %=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        lhs.data[L0] %= rhs; lhs.data[L1] %= rhs; lhs.data[L2] %= rhs;  return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator %=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        return { lhs.data[L0] % rhs.data[R0], lhs.data[L1] % rhs.data[R1], lhs.data[L2] % rhs.data[R2] };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator ^=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        lhs.data[L0] ^= rhs; lhs.data[L1] ^= rhs; lhs.data[L2] ^= rhs; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator ^=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] ^= rhs.data[R0]; lhs.data[L1] ^= rhs.data[R1]; lhs.data[L2] ^= rhs.data[R2]; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator &=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        lhs.data[L0] &= rhs; lhs.data[L1] &= rhs; lhs.data[L2] &= rhs; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator &=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] &= rhs.data[R0]; lhs.data[L1] &= rhs.data[R1]; lhs.data[L2] &= rhs.data[R2]; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator |=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        lhs.data[L0] |= rhs; lhs.data[L1] |= rhs; lhs.data[L2] |= rhs; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2, int RS, int R0, int R1, int R2>
    __host__ __device__ __forceinline__ __ivec3<Type>& operator |=(__vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const __vec_swizzle<Type, RS, 3, R0, R1, R2>& rhs)
    {
        lhs.data[L0] |= rhs.data[R0]; lhs.data[L1] |= rhs.data[R1]; lhs.data[L2] |= rhs.data[R2]; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator<<(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] << rhs, lhs.data[L1] << rhs, lhs.data[L2] << rhs };
    }
    template<typename Type, int LS, int L0, int L1, int L2>
    __host__ __device__ __forceinline__ __ivec3<Type> operator >>(const __vec_swizzle<Type, LS, 3, L0, L1, L2>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] >> rhs, lhs.data[L1] >> rhs, lhs.data[L2] >> rhs };
    }


    template<typename Type>
    __host__ __device__ __forceinline__ __ivec3<Type> clamp(const __ivec3<Type>& v, const __ivec3<Type>& a, const __ivec3<Type>& b)
    {
        return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z) };
    }
    template<typename Type>
    __host__ __device__ __forceinline__ __ivec3<Type> abs(const __ivec3<Type>& a)
    {
        return { abs(a.x), abs(a.y), abs(a.z) };
    }
    template<typename Type>
    __host__ __device__ __forceinline__ Type sum(const __ivec3<Type>& a)
    {
        return a.x + a.y + a.z;
    }
    template<typename Type>
    __host__ __device__ __forceinline__ __ivec3<Type> sign(const __ivec3<Type>& v)
    {
        return { sign(v.x), sign(v.y), sign(v.z) };
    }

    template<typename Type>
    __host__ __device__ __forceinline__ bool operator==(const __ivec3<Type>& lhs, const __ivec3<Type>& rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
    template<typename Type>
    __host__ __device__ __forceinline__ bool operator!=(const __ivec3<Type>& lhs, const __ivec3<Type>& rhs)
    {
        return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z;
    }

    template<typename Type>
    __host__ __device__ __forceinline__ Type cwiseMax(const __ivec3<Type>& v)
    {
        return (v.x > v.y) ? ((v.x > v.z) ? v.x : v.z) : ((v.y > v.z) ? v.y : v.z);
    }

    template<typename Type>
    __host__ __device__ __forceinline__ Type cwiseMin(const __ivec3<Type>& v)
    {
        return (v.x < v.y) ? ((v.x < v.z) ? v.x : v.z) : ((v.y < v.z) ? v.y : v.z);
    }

    // FIXME: Cuda intrinsics aren't working. Why is this?
    //__host__ __device__ __forceinline__ vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

    using ivec3 = __ivec3<int>;
    using uvec3 = __ivec3<uint>;
}