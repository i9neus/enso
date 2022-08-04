#pragma once

#include "CudaMat2.cuh"
#include "../vec/CudaVec3.cuh"

namespace Cuda
{
	#define _det2(a, b, c, d) ((a) * (d) - (b) * (c))
	
	template<typename Type>
	struct __mat3
	{
		enum _attrs : size_t { kDims = 3 };
		using VecType = __vec_swizzle<Type, 3, 3, 0, 1, 2>;

		union
		{
			struct { VecType x, y, z; };
			struct
			{
				Type i00, i01, i02;
				Type i10, i11, i12;
				Type i20, i21, i22;
			};
			VecType data[3];
		};

		__host__ __device__ __forceinline__ __mat3() {}
		__host__ __device__ __forceinline__ __mat3(const __mat3&) = default;
		__host__ __device__ __forceinline__ __mat3(const VecType& x_, const VecType& y_, const VecType& z_) : x(x_), y(y_), z(z_) {}
		__host__ __device__ __forceinline__ __mat3(const float& i00_, const float& i01_, const float& i02_, 
											       const float& i10_, const float& i11_, const float& i12_, 
												   const float& i20_, const float& i21_, const float& i22_) : 
			i00(i00_), i01(i01_), i02(i02_), 
			i10(i10_), i11(i11_), i12(i12_),
			i20(i20_), i21(i21_), i22(i22_) {}

		__host__ __device__ __forceinline__ static __mat3 Indentity()
		{
			return __mat3(VecType(1.0f, 0.0f, 0.0f), VecType(0.0f, 1.0f, 0.0f), VecType(0.0f, 0.0f, 1.0f));
		}

		__host__ __device__ __forceinline__  static __mat3 Null()
		{
			return __mat3(VecType(0.0f, 0.0f, 0.0f), VecType(0.0f, 0.0f, 0.0f), VecType(0.0f, 0.0f, 0.0f));
		}

		__host__ __device__ __forceinline__ bool IsSymmetric() const
		{
			return i10 == i01 && i20 == i02 && i21 == i12;
		}

		__host__ __device__ __forceinline__ const VecType& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ __forceinline__ VecType& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format(const bool pretty = false) const
		{
			const char nl = pretty ? '\n' : ' ';
			return tfm::format("{%s,%c%s,%c%s}", x.format(), nl, y.format(), nl, z.format());
		}
	};

	using mat3 = __mat3<float>;
	using imat3 = __mat3<int>;
	using umat3 = __mat3<uint>;

	template<typename Type>
	__host__ __device__ __forceinline__ __mat3<Type> operator *(const __mat3<Type>& a, const __mat3<Type>& b)
	{
		__mat3<Type> r;
		r.i00 = a.i00 * b.i00 + a.i01 * b.i10 + a.i02 * b.i20;
		r.i10 = a.i10 * b.i00 + a.i11 * b.i10 + a.i12 * b.i20;
		r.i20 = a.i20 * b.i00 + a.i21 * b.i10 + a.i22 * b.i20;
		r.i01 = a.i00 * b.i01 + a.i01 * b.i11 + a.i02 * b.i21;
		r.i11 = a.i10 * b.i01 + a.i11 * b.i11 + a.i12 * b.i21;
		r.i21 = a.i20 * b.i01 + a.i21 * b.i11 + a.i22 * b.i21;
		r.i02 = a.i00 * b.i02 + a.i01 * b.i12 + a.i02 * b.i22;
		r.i12 = a.i10 * b.i02 + a.i11 * b.i12 + a.i12 * b.i22;
		r.i22 = a.i20 * b.i02 + a.i21 * b.i12 + a.i22 * b.i22;
		return r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ __mat3<Type>& operator *=(__mat3<Type>& a, const __mat3<Type>& b)
	{
		const __mat3<Type> r = a * b;
		return a = r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ typename __mat3<Type>::VecType operator *(const __mat3<Type>& a, const typename __mat3<Type>::VecType& b)
	{
		typename __mat3<Type>::VecType r;
		r.x = a.i00 * b.x + a.i01 * b.y + a.i02 * b.z;
		r.y = a.i10 * b.x + a.i11 * b.y + a.i12 * b.z;
		r.z = a.i20 * b.x + a.i21 * b.y + a.i22 * b.z;
		return r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ typename __mat2<Type>::VecType operator *(const __mat3<Type>& a, const typename __mat2<Type>::VecType& b)
	{
		typename __mat2<Type>::VecType r;
		r.x = a.i00 * b.x + a.i01 * b.y + a.i02;
		r.y = a.i10 * b.x + a.i11 * b.y + a.i12;
		return r;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ __mat3<Type> operator *(const __mat3<Type>& a, const Type& b)
	{
		return __mat3<Type>(a[0] * b, a[1] * b, a[2] * b);
	}

	template<typename Type>
	__host__ __device__ __forceinline__ Type trace(const __mat3<Type>& m)
	{
		return m.i00 + m.i11 + m.i22;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ Type det(const __mat3<Type>& m)
	{
		return m.i00 * m.i11 * m.i22 +
			m.i01 * m.i12 * m.i20 +
			m.i02 * m.i10 * m.i21 -
			m.i02 * m.i11 * m.i20 -
			m.i01 * m.i10 * m.i22 -
			m.i00 * m.i12 * m.i21;
	}

	template<typename Type>
	__host__ __device__ __forceinline__ __mat3<Type> transpose(const __mat3<Type>& m)
	{
		__mat3<Type> r;
		r.i00 = m.i00; r.i01 = m.i10; r.i02 = m.i20;
		r.i10 = m.i01; r.i11 = m.i11; r.i12 = m.i21;
		r.i20 = m.i02; r.i21 = m.i12; r.i22 = m.i22;
		return r;
	}

	template<typename Type, typename = typename std::enable_if<std::is_floating_point<Type>::value>>
	__host__ __device__ __forceinline__ __mat3<Type> inverse(const __mat3<Type>& m)
	{
		constexpr Type kInverseEpsilon = 1e-20f;

		const Type determinant = det(m);
		if (fabs(determinant) < kInverseEpsilon) { return __mat3<Type>::Null(); }
		const Type invDet = 1 / determinant;

		// The adjugate matrix divided by the determinant
		return { { _det2(m.i11, m.i12, m.i21, m.i22) * invDet, -_det2(m.i01, m.i02, m.i21, m.i22) * invDet, _det2(m.i01, m.i02, m.i11, m.i12) * invDet},
				 { -_det2(m.i10, m.i12, m.i20, m.i22) * invDet, _det2(m.i00, m.i02, m.i20, m.i22) * invDet, -_det2(m.i00, m.i02, m.i10, m.i12) * invDet},
				 { _det2(m.i10, m.i11, m.i20, m.i21) * invDet, -_det2(m.i00, m.i01, m.i20, m.i21) * invDet, _det2(m.i00, m.i01, m.i10, m.i11) * invDet } };
	}

	template<typename Type>
	__host__ __device__ __forceinline__ typename __mat3<Type>::VecType BasisU(const __mat3<Type>& m) { return typename __mat3<Type>::VecType(m.i00, m.i10, m.i20); }
	template<typename Type>
	__host__ __device__ __forceinline__ typename __mat3<Type>::VecType BasisV(const __mat3<Type>& m) { return typename __mat3<Type>::VecType(m.i01, m.i11, m.i21); }
	template<typename Type>
	__host__ __device__ __forceinline__ typename __mat3<Type>::VecType BasisW(const __mat3<Type>& m) { return typename __mat3<Type>::VecType(m.i02, m.i12, m.i22); }

	template<typename Type>
	__host__ __device__ __forceinline__ bool operator ==(const __mat3<Type>& a, const __mat3<Type>& b)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (a[i][j] != b[i][j]) { return false; }
			}
		}
		return true;
	}
	template<typename Type>
	__host__ __device__ __forceinline__ bool operator !=(const __mat3<Type>& a, const __mat3<Type>& b) { return !(a == b); }
}