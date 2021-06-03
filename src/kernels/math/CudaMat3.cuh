#pragma once

#include "CudaVecBase.cuh"
#include "CudaVec3.cuh"
#include "CudaVec4.cuh"

namespace Cuda
{
	#define _det2(a, b, c, d) ((a) * (d) - (b) * (c))
	
	struct mat3
	{
		enum _attrs : size_t { kDims = 3 };
		using kVecType = vec3;

		union
		{
			struct { vec3 x, y, z; };
			struct
			{
				float i00, i01, i02;
				float i10, i11, i12;
				float i20, i21, i22;
			};
			vec3 data[3];
		};

		mat3() = default;
		__host__ __device__ mat3(const vec3& x_, const vec3& y_, const vec3& z_) : x(x_), y(y_), z(z_) {}
		__host__ __device__ mat3(const mat3& other) : x(other.x), y(other.y), z(other.z) {}

		__host__ __device__  static mat3 indentity()
		{
			return mat3(vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
		}

		__host__ __device__  static mat3 null()
		{
			return mat3(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
		}

		__host__ __device__ inline const vec3& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline vec3& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format(const bool pretty = false) const
		{
			const char nl = pretty ? '\n' : ' ';
			return tfm::format("{%s,%c%s,%c%s}", x.format(), nl, y.format(), nl, z.format());
		}
	};

	__host__ __device__ inline mat3 operator *(const mat3& a, const mat3& b)
	{
		mat3 r;
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

	__host__ __device__ inline mat3& operator *=(mat3& a, const mat3& b)
	{
		const mat3 r = a * b;
		return a = r;
	}

	__host__ __device__ inline vec3 operator *(const mat3& a, const vec3& b)
	{
		vec3 r;
		r.x = a.i00 * b.x + a.i01 * b.y + a.i02 * b.z;
		r.y = a.i10 * b.x + a.i11 * b.y + a.i12 * b.z;
		r.z = a.i20 * b.x + a.i21 * b.y + a.i22 * b.z;
		return r;
	}

	__host__ __device__ inline mat3 operator *(const mat3& a, const float& b)
	{
		return mat3(a[0] * b, a[1] * b, a[2] * b);
	}

	__host__ __device__ inline float trace(const mat3& m)
	{
		return m.i00 + m.i11 + m.i22;
	}

	__host__ __device__ inline float det(const mat3& m)
	{
		return m.i00 * m.i11 * m.i22 +
			m.i01 * m.i12 * m.i20 +
			m.i02 * m.i10 * m.i21 -
			m.i02 * m.i11 * m.i20 -
			m.i01 * m.i10 * m.i22 -
			m.i00 * m.i12 * m.i21;
	}

	__host__ __device__ inline mat3 transpose(const mat3& m)
	{
		mat3 r;
		r.i00 = m.i00; r.i01 = m.i10; r.i02 = m.i20;
		r.i10 = m.i01; r.i11 = m.i11; r.i12 = m.i21;
		r.i20 = m.i02; r.i21 = m.i12; r.i22 = m.i22;
		return r;
	}

	__host__ __device__ inline mat3 inverse(const mat3& m)
	{
		constexpr float kInverseEpsilon = 1e-20f;

		const float determinant = det(m);
		if (fabs(determinant) < kInverseEpsilon) { return mat3::null(); }
		const float invDet = 1 / determinant;

		// The adjugate matrix divided by the determinant
		return { { _det2(m.i11, m.i12, m.i21, m.i22) * invDet, -_det2(m.i01, m.i02, m.i21, m.i22) * invDet, _det2(m.i01, m.i02, m.i11, m.i12) * invDet},
				 { -_det2(m.i10, m.i12, m.i20, m.i22) * invDet, _det2(m.i00, m.i02, m.i20, m.i22) * invDet, -_det2(m.i00, m.i02, m.i10, m.i12) * invDet},
				 { _det2(m.i10, m.i11, m.i20, m.i21) * invDet, -_det2(m.i00, m.i01, m.i20, m.i21) * invDet, _det2(m.i00, m.i01, m.i10, m.i11) * invDet } };
	}

	__host__ __device__ inline bool operator ==(const mat3& a, const mat3& b)
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
	__host__ __device__ inline bool operator !=(const mat3& a, const mat3& b) { return !(a == b); }
}