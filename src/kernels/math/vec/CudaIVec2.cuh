#pragma once

#include "CudaVecBase.cuh"

namespace Cuda
{
	template<typename Type>
	struct __align__(8) __vec_swizzle<Type, 2, 2, 0, 1>
	{
		enum _attrs : size_t { kDims = 2 };
		using kType = Type;
		union
		{
			struct { Type x, y; };
			struct { Type i0, i1; };
			Type data[2];

			__vec_swizzle<Type, 2, 2, 0, 0> xx;
			__vec_swizzle<Type, 2, 2, 1, 0> yx;
			__vec_swizzle<Type, 2, 2, 1, 1> yy;
		};

		__vec_swizzle() = default;
		__vec_swizzle(const __vec_swizzle&) = default;
		__host__ __device__ explicit __vec_swizzle(const Type v) : x(v), y(v) {}
		__host__ __device__ __vec_swizzle(const Type& x_, const Type& y_) : x(x_), y(y_) {}

		// Cast from other vec2 types
		template<typename OtherType, int I0, int I1>
		__host__ __device__ explicit __vec_swizzle(const __vec_swizzle<OtherType, 2, 2, I0, I1>& v) :
			x(Type(v.data[I0])), y(Type(v.data[I1])) {}

		template<int L0, int L1, int R0, int R1>
		__host__ __device__ inline void UnpackTo(Type* otherData) const
		{
			otherData[L0] = data[0];
			otherData[L1] = data[1];
		}

		// Cast from swizzled types
		template<int ActualSize, int... In>
		__host__ __device__ inline __vec_swizzle(const __vec_swizzle<Type, ActualSize, 2, In...>& swizzled)
		{
			swizzled.UnpackTo<0, 1, In...>(data);
		}

		// Assign from swizzled types
		template<int RS, int R0, int R1>
		__host__ __device__ inline __vec_swizzle& operator=(const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
		{
			x = rhs.other[R0]; y = rhs.other[R1];
			return *this;
		}

		__host__ __device__ __vec_swizzle& operator=(const Type& v) { x = v; y = v; return *this; }

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%i, %i}", x, y); }
	};

    // Alias vec3 to the linear triple
    template<typename Type> using __ivec2 = __vec_swizzle<Type, 2, 2, 0, 1>;

    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type> operator +(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] + rhs.data[R0], lhs.data[L1] + rhs.data[R1] };
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator +(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] + rhs, lhs.data[L1] + rhs};
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type> operator -(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] - rhs.data[R0], lhs.data[L1] - rhs.data[R1] };
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator -(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] - rhs, lhs.data[L1] - rhs };
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator -(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs)
    {
        return { -lhs.data[L0], -lhs.data[L1]};
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type> operator *(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] * rhs.data[R0], lhs.data[L1] * rhs.data[R1]};
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator *(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] * rhs, lhs.data[L1] * rhs };
    }
    template<typename Type, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type> operator *(const Type& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        return { lhs * rhs.data[L0], lhs * rhs.data[L1] };
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type> operator /(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] / rhs.data[R0], lhs.data[L1] / rhs.data[R1] };
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator /(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] / rhs, lhs.data[L1] / rhs };
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __vec_swizzle<Type, LS, 2, L0, L1>& operator +=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] += rhs.data[R0]; lhs.data[L1] += rhs.data[R1]; 
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __vec_swizzle<Type, LS, 2, L0, L1>& operator -=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] -= rhs.data[R0]; lhs.data[L1] -= rhs.data[R1]; 
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __vec_swizzle<Type, LS, 2, L0, L1>& operator *=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] *= rhs.data[R0]; lhs.data[L1] *= rhs.data[R1];
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __vec_swizzle<Type, LS, 2, L0, L1>& operator *=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        lhs.data[L0] *= rhs; lhs.data[L1] *= rhs; 
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __vec_swizzle<Type, LS, 2, L0, L1>& operator /=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] /= rhs.data[R0]; lhs.data[L1] /= rhs.data[R1];
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __vec_swizzle<Type, LS, 2, L0, L1>& operator /=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        lhs.data[L0] /= rhs; lhs.data[L1] /= rhs;
        return lhs;
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type>& operator %=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        lhs.data[L0] %= rhs; lhs.data[L1] %= rhs; lhs.data[L2] %= rhs;  return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type>& operator %=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        return { lhs.data[L0] % rhs.data[R0], lhs.data[L1] % rhs.data[R1]};
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type>& operator ^=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        lhs.data[L0] ^= rhs; lhs.data[L1] ^= rhs; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type>& operator ^=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] ^= rhs.data[R0]; lhs.data[L1] ^= rhs.data[R1]; return lhs;
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type>& operator &=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        lhs.data[L0] &= rhs; lhs.data[L1] &= rhs; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type>& operator &=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] &= rhs.data[R0]; lhs.data[L1] &= rhs.data[R1]; return lhs;
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type>& operator |=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        lhs.data[L0] |= rhs; lhs.data[L1] |= rhs; return lhs;
    }
    template<typename Type, int LS, int L0, int L1, int RS, int R0, int R1>
    __host__ __device__ inline __ivec2<Type>& operator |=(__vec_swizzle<Type, LS, 2, L0, L1>& lhs, const __vec_swizzle<Type, RS, 2, R0, R1>& rhs)
    {
        lhs.data[L0] |= rhs.data[R0]; lhs.data[L1] |= rhs.data[R1]; return lhs;
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator<<(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] << rhs, lhs.data[L1] << rhs};
    }
    template<typename Type, int LS, int L0, int L1>
    __host__ __device__ inline __ivec2<Type> operator >>(const __vec_swizzle<Type, LS, 2, L0, L1>& lhs, const Type& rhs)
    {
        return { lhs.data[L0] >> rhs, lhs.data[L1] >> rhs };
    }


    template<typename Type>
    __host__ __device__ inline __ivec2<Type> clamp(const __ivec2<Type>& v, const __ivec2<Type>& a, const __ivec2<Type>& b)
    {
        return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y) };
    }
    template<typename Type>
    __host__ __device__ inline __ivec2<Type> abs(const __ivec2<Type>& a)
    {
        return { abs(a.x), abs(a.y) };
    }
    template<typename Type>
    __host__ __device__ inline Type sum(const __ivec2<Type>& a)
    {
        return a.x + a.y;
    }
    template<typename Type>
    __host__ __device__ inline __ivec2<Type> sign(const __ivec2<Type>& v)
    {
        return { sign(v.x), sign(v.y) };
    }

    template<typename Type>
    __host__ __device__ inline bool operator==(const __ivec2<Type>& lhs, const __ivec2<Type>& rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
    template<typename Type>
    __host__ __device__ inline bool operator!=(const __ivec2<Type>& lhs, const __ivec2<Type>& rhs)
    {
        return lhs.x != rhs.x || lhs.y != rhs.y;
    }

    template<typename Type>
    __host__ __device__ inline Type cwiseMax(const __ivec2<Type>& v)
    {
        return (v.x > v.y) ? v.x : v.y;
    }

    template<typename Type>
    __host__ __device__ inline Type cwiseMin(const __ivec2<Type>& v)
    {
        return (v.x < v.y) ? v.x : v.y;
    }

    // FIXME: Cuda intrinsics aren't working. Why is this?
    //__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

    using ivec2 = __ivec2<int>;
    using uvec2 = __ivec2<uint>;
}