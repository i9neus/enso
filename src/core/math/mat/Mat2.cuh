#pragma once

#include "../vec/CudaVec2.cuh"

namespace Cuda
{
	template<typename Type>
	struct __mat2
	{
		enum _attrs : size_t { kDims = 2 };
		using VecType = __vec_swizzle<Type, 2, 2, 0, 1 >;

		union
		{
			struct { VecType x, y; };
			struct
			{
				Type i00, i01;
				Type i10, i11;
			};
			VecType data[2];
		};

		__host__ __device__ __forceinline__ __mat2() {}
		//__host__ __device__ __forceinline__ __mat2(const __mat2&) = default; // NOTE: Commented out to suppress nvcc compiler warnings
		__host__ __device__ __forceinline__ __mat2(const VecType& x_, const VecType& y_) : x(x_), y(y_) {}
		__host__ __device__ __forceinline__ __mat2(const float& i00_, const float& i01_, const float& i10_, const float& i11_) :
			i00(i00_), i01(i01_),
			i10(i10_), i11(i11_) {}

		__host__ __device__ __forceinline__ static __mat2 Indentity()
		{
			return __mat2(VecType(1.0f, 0.0f), VecType(0.0f, 1.0f));
		}

		__host__ __device__ __forceinline__  static __mat2 Null()
		{
			return __mat2(VecType(0.0f, 0.0f), VecType(0.0f, 0.0f));
		}

		__host__ __device__ __forceinline__ bool IsSymmetric() const
		{
			return i01 == i10;
		}

		__host__ __device__ __forceinline__ const VecType& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ __forceinline__ VecType& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format(const bool pretty = false) const
		{
			const char nl = pretty ? '\n' : ' ';
			return tfm::format("{%s,%c%s}", x.format(), nl, y.format());
		}
	};

	using mat2 = __mat2<float>;
	using imat2 = __mat2<int>;
	using umat2 = __mat2<uint>;

	template<typename Type>
	__host__ __device__ __forceinline__ __mat2<Type> operator *(const __mat2<Type>& a, const __mat2<Type>& b)
	{
		__mat2<Type> r;
		r.i00 = a.i00 * b.i00 + a.i01 * b.i10;
		r.i10 = a.i10 * b.i00 + a.i11 * b.i10;
		r.i01 = a.i00 * b.i01 + a.i01 * b.i11;
		r.i11 = a.i10 * b.i01 + a.i11 * b.i11;
		return r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ __mat2<Type>& operator *=(__mat2<Type>& a, const __mat2<Type>& b)
	{
		const __mat2<Type> r = a * b;
		return a = r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ typename __mat2<Type>::VecType operator *(const __mat2<Type>& a, const typename __mat2<Type>::VecType& b)
	{
		typename __mat2<Type>::VecType r;
		r.x = a.i00 * b.x + a.i01 * b.y;
		r.y = a.i10 * b.x + a.i11 * b.y;
		return r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ __mat2<Type> operator *(const __mat2<Type>& a, const Type& b)
	{
		return __mat2<Type>(a[0] * b, a[1] * b);
	}

	template<typename Type>
	__host__ __device__ __forceinline__ Type trace(const __mat2<Type>& m)
	{
		return m.i00 + m.i11;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ Type det(const __mat2<Type>& m)
	{
		return m.i00 * m.i11 - m.i01 * m.i10;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ __mat2<Type> transpose(const __mat2<Type>& m)
	{
		__mat2<Type> r;
		r.i00 = m.i00; r.i01 = m.i10;
		r.i10 = m.i01; r.i11 = m.i11; 
		return r;
	}

	template<typename Type, typename = typename std::enable_if<std::is_floating_point<Type>::value>>
	__host__ __device__ __forceinline__ __mat2<Type> inverse(const __mat2<Type>& m)
	{
		constexpr Type kInverseEpsilon = 1e-20f;

		const Type determinant = det(m);
		if (fabs(determinant) < kInverseEpsilon) { return __mat2<Type>::Null(); }
		const Type invDet = 1 / determinant;

		return { { m.i11 * invDet, -m.i01 * invDet }, { -m.i10 * invDet, m.i00 * invDet } };
	}

	template<typename Type>
	__host__ __device__ __forceinline__ bool operator ==(const __mat2<Type>& a, const __mat2<Type>& b)
	{
		return a.i00 == b.i00 && a.i01 == b.i01 && a.i10 == b.i10 && a.i11 == b.i11;
	}
	template<typename Type>
	__host__ __device__ __forceinline__ bool operator !=(const __mat2<Type>& a, const __mat2<Type>& b) { return !(a == b); }
}