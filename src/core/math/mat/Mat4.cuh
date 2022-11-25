#pragma once

#include "CudaMat3.cuh"
#include "../vec/CudaVec4.cuh"

namespace Cuda
{
	struct __builtin_align__(16) mat4
	{
		enum _attrs : size_t { kDims = 4 };
		using VecType = vec4;

		union
		{
			struct { vec4 x, y, z, w; };
			struct
			{
				float i00, i01, i02, i03;
				float i10, i11, i12, i13;
				float i20, i21, i22, i23;
				float i30, i31, i32, i33;
			};
			vec4 data[4];
		};

		__host__ __device__  static mat4 Indentity()
		{
			return mat4(vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.0f, 0.0f, 0.0f, 1.0f));
		}

		__host__ __device__  static mat4 Null()
		{
			return mat4(vec4(0.0f, 0.0f, 0.0f, 0.0f), vec4(0.0f, 0.0f, 0.0f, 0.0f), vec4(0.0f, 0.0f, 0.0f, 0.0f), vec4(0.0f, 0.0f, 0.0f, 0.0f));
		}

		__host__ __device__ __forceinline__ mat4() {}
		//__host__ __device__ __forceinline__ mat4(const mat4&) = default; // NOTE: Commented out to suppress nvcc compiler warnings
		__host__ __device__ __forceinline__ mat4(const vec4& x_, const vec4& y_, const vec4& z_, const vec4& w_) : x(x_), y(y_), z(z_), w(w_) {}
		__host__ __device__ __forceinline__ mat4(const float& i00_, const float& i01_, const float& i02_, const float& i03_,
												 const float& i10_, const float& i11_, const float& i12_, const float& i13_,
											     const float& i20_, const float& i21_, const float& i22_, const float& i23_,
												 const float& i30_, const float& i31_, const float& i32_, const float& i33_) :
			i00(i00_), i01(i01_), i02(i02_), i03(i03_),
			i10(i10_), i11(i11_), i12(i12_), i13(i13_),
			i20(i20_), i21(i21_), i22(i22_), i23(i23_),
			i30(i30_), i31(i31_), i32(i32_), i33(i33_) {}

		__host__ __device__ __forceinline__ const vec4& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ __forceinline__ vec4& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format(const bool pretty = false) const
		{
			const char nl = pretty ? '\n' : ' ';
			return tfm::format("{%s,%c%s,%c%s,%c%}", x.format(), nl, y.format(), nl, z.format(), nl, w.format());
		}
	};

	//template<typename T>
	//__host__ __device__ __forceinline__ void cast(const mat4& m, T v[4]) { v[0] = cast<T>(m[0]); v[1] = cast<T>(m[1]); v[2] = cast<T>(m[2]); v[3] = cast<T>(m[3]); }

	__host__ __device__ __forceinline__ mat4 operator *(const mat4& a, const mat4& b)
	{
		mat4 r;
		r.i00 = a.i00 * b.i00 + a.i01 * b.i10 + a.i02 * b.i20 + a.i03 * b.i30;
		r.i10 = a.i10 * b.i00 + a.i11 * b.i10 + a.i12 * b.i20 + a.i13 * b.i30;
		r.i20 = a.i20 * b.i00 + a.i21 * b.i10 + a.i22 * b.i20 + a.i23 * b.i30;
		r.i30 = a.i30 * b.i00 + a.i31 * b.i10 + a.i32 * b.i20 + a.i33 * b.i30;
		r.i01 = a.i00 * b.i01 + a.i01 * b.i11 + a.i02 * b.i21 + a.i03 * b.i31;
		r.i11 = a.i10 * b.i01 + a.i11 * b.i11 + a.i12 * b.i21 + a.i13 * b.i31;
		r.i21 = a.i20 * b.i01 + a.i21 * b.i11 + a.i22 * b.i21 + a.i23 * b.i31;
		r.i31 = a.i30 * b.i01 + a.i31 * b.i11 + a.i32 * b.i21 + a.i33 * b.i31;
		r.i02 = a.i00 * b.i02 + a.i01 * b.i12 + a.i02 * b.i22 + a.i03 * b.i32;
		r.i12 = a.i10 * b.i02 + a.i11 * b.i12 + a.i12 * b.i22 + a.i13 * b.i32;
		r.i22 = a.i20 * b.i02 + a.i21 * b.i12 + a.i22 * b.i22 + a.i23 * b.i32;
		r.i32 = a.i30 * b.i02 + a.i31 * b.i12 + a.i32 * b.i22 + a.i33 * b.i32;
		r.i03 = a.i00 * b.i03 + a.i01 * b.i13 + a.i02 * b.i23 + a.i03 * b.i33;
		r.i13 = a.i10 * b.i03 + a.i11 * b.i13 + a.i12 * b.i23 + a.i13 * b.i33;
		r.i23 = a.i20 * b.i03 + a.i21 * b.i13 + a.i22 * b.i23 + a.i23 * b.i33;
		r.i33 = a.i30 * b.i03 + a.i31 * b.i13 + a.i32 * b.i23 + a.i33 * b.i33;
		return r;
	}

	__host__ __device__ __forceinline__ mat4& operator *=(mat4& a, const mat4& b)
	{
		const mat4 r = a * b;
		return a = r;
	}

	__host__ __device__ __forceinline__ vec4 operator *(const mat4& a, const vec4& b)
	{
		vec4 r;
		r.i0 = a.i00 * b.i0 + a.i01 * b.i1 + a.i02 * b.i2 + a.i03 * b[3];
		r.i1 = a.i10 * b.i0 + a.i11 * b.i1 + a.i12 * b.i2 + a.i13 * b[3];
		r.i2 = a.i20 * b.i0 + a.i21 * b.i1 + a.i22 * b.i2 + a.i23 * b[3];
		r[3] = a.i30 * b.i0 + a.i31 * b.i1 + a.i32 * b.i2 + a.i33 * b[3];
		return r;
	}

	__host__ __device__ __forceinline__ vec3 operator *(const mat4& a, const vec3& b)
	{
		vec3 r;
		r.x = a.i00 * b.x + a.i01 * b.y + a.i02 * b.z + a.i03;
		r.y = a.i10 * b.x + a.i11 * b.y + a.i12 * b.z + a.i13;
		r.z = a.i20 * b.x + a.i21 * b.y + a.i22 * b.z + a.i23;
		return r;
	}

	__host__ __device__ __forceinline__ float trace(const mat4& m)
	{
		return m.i00 + m.i11 + m.i22 + m.i33;
	}

	__host__ __device__ __forceinline__ mat4 transpose(const mat4& m)
	{
		mat4 r;
		r.i00 = m.i00; r.i01 = m.i10; r.i02 = m.i20; r.i03 = m.i30;
		r.i10 = m.i01; r.i11 = m.i11; r.i12 = m.i21; r.i13 = m.i31;
		r.i20 = m.i02; r.i21 = m.i12; r.i22 = m.i22; r.i23 = m.i32;
		r.i30 = m.i03; r.i31 = m.i13; r.i32 = m.i23; r.i33 = m.i33;
		return r;
	}

	__host__ __device__ __forceinline__ float det(const mat4& m)
	{
		// Laplace expansion 	
		return (m.i00 * m.i11 - m.i10 * m.i01) * (m.i22 * m.i33 - m.i32 * m.i23) -
			(m.i00 * m.i12 - m.i10 * m.i02) * (m.i21 * m.i33 - m.i31 * m.i23) +
			(m.i00 * m.i13 - m.i10 * m.i03) * (m.i21 * m.i32 - m.i31 * m.i22) +
			(m.i01 * m.i12 - m.i11 * m.i02) * (m.i20 * m.i33 - m.i30 * m.i23) -
			(m.i01 * m.i13 - m.i11 * m.i03) * (m.i20 * m.i32 - m.i30 * m.i22) +
			(m.i02 * m.i13 - m.i12 * m.i03) * (m.i20 * m.i31 - m.i30 * m.i21);
	}

	__host__ __device__ __forceinline__ mat4 inverse(const mat4& m)
	{
		constexpr float kInverseEpsilon = 1e-20f;

		// Laplace expansion 
		const float s0 = m.i00 * m.i11 - m.i10 * m.i01;
		const float s1 = m.i00 * m.i12 - m.i10 * m.i02;
		const float s2 = m.i00 * m.i13 - m.i10 * m.i03;
		const float s3 = m.i01 * m.i12 - m.i11 * m.i02;
		const float s4 = m.i01 * m.i13 - m.i11 * m.i03;
		const float s5 = m.i02 * m.i13 - m.i12 * m.i03;
		const float c5 = m.i22 * m.i33 - m.i32 * m.i23;
		const float c4 = m.i21 * m.i33 - m.i31 * m.i23;
		const float c3 = m.i21 * m.i32 - m.i31 * m.i22;
		const float c2 = m.i20 * m.i33 - m.i30 * m.i23;
		const float c1 = m.i20 * m.i32 - m.i30 * m.i22;
		const float c0 = m.i20 * m.i31 - m.i30 * m.i21;

		const float determinant = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
		if (fabs(determinant) < kInverseEpsilon) { return mat4::Null(); }

		const float invDet = 1 / determinant;
		mat4 r;
		r.i00 = (m.i11 * c5 - m.i12 * c4 + m.i13 * c3) * invDet;
		r.i01 = (-m.i01 * c5 + m.i02 * c4 - m.i03 * c3) * invDet;
		r.i02 = (m.i31 * s5 - m.i32 * s4 + m.i33 * s3) * invDet;
		r.i03 = (-m.i21 * s5 + m.i22 * s4 - m.i23 * s3) * invDet;
		r.i10 = (-m.i10 * c5 + m.i12 * c2 - m.i13 * c1) * invDet;
		r.i11 = (m.i00 * c5 - m.i02 * c2 + m.i03 * c1) * invDet;
		r.i12 = (-m.i30 * s5 + m.i32 * s2 - m.i33 * s1) * invDet;
		r.i13 = (m.i20 * s5 - m.i22 * s2 + m.i23 * s1) * invDet;
		r.i20 = (m.i10 * c4 - m.i11 * c2 + m.i13 * c0) * invDet;
		r.i21 = (-m.i00 * c4 + m.i01 * c2 - m.i03 * c0) * invDet;
		r.i22 = (m.i30 * s4 - m.i31 * s2 + m.i33 * s0) * invDet;
		r.i23 = (-m.i20 * s4 + m.i21 * s2 - m.i23 * s0) * invDet;
		r.i30 = (-m.i10 * c3 + m.i11 * c1 - m.i12 * c0) * invDet;
		r.i31 = (m.i00 * c3 - m.i01 * c1 + m.i02 * c0) * invDet;
		r.i32 = (-m.i30 * s3 + m.i31 * s1 - m.i32 * s0) * invDet;
		r.i33 = (m.i20 * s3 - m.i21 * s1 + m.i22 * s0) * invDet;

		return r;
	}

	__host__ __device__ __forceinline__ bool operator ==(const mat4& a, const mat4& b)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (a[i][j] != b[i][j]) { return false; }
			}
		}
		return true;
	}
	__host__ __device__ __forceinline__ bool operator !=(const mat4& a, const mat4& b) { return !(a == b); }
}