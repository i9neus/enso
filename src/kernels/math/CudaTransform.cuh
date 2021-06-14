#pragma once

#include "vec/CudaIVec2.cuh"
#include "vec/CudaIVec3.cuh"
#include "vec/CudaIVec4.cuh"
#include "vec/CudaVec2.cuh"
#include "vec/CudaVec3.cuh"
#include "vec/CudaVec4.cuh"
#include "mat/CudaMat3.cuh"
#include "mat/CudaMat4.cuh"

namespace Cuda
{
	class BidirectionalTransform
	{
	public:
		vec3 trans;
		mat3 fwd;
		mat3 inv;
		mat3 nInv;
		vec3 scale;

		__host__ __device__ BidirectionalTransform()
		{
			MakeIdentity();
		}

		__host__ __device__ inline BidirectionalTransform(const vec3& t, const mat3& f) : trans(t), fwd(f)
		{
			inv = inverse(fwd);
			nInv = transpose(fwd);
			scale = vec3(1 / length(fwd[0]), 1 / length(fwd[1]), 1 / length(fwd[2]));
		}

		__host__ __device__ inline void MakeIdentity()
		{
			trans = 0.0f;
			fwd = inv = nInv = mat3::Indentity();
			scale = 1.0f;
		}

		/*__device__ inline HitPoint HitToWorldSpace(const HitPoint& object) const
		{
			HitPoint world;
			world.p = inv * object.p;
			//world.n = nInv * object.n;
			world.n = normalize((inv * (object.n + object.o)) - world.o);
			return world;
		}

		__device__ inline HitPoint HitToObjectSpace(const HitPoint& world) const
		{
			HitPoint object;
			object.p = fwd * world.p;
			//world.n = nInv * world.n;
			object.n = normalize((fwd * (world.n + world.o)) - object.o);
			return object;
		}*/

		__device__ inline vec3 PointToWorldSpace(const vec3& object) const
		{
			return (inv * object) + trans;
		}
	};

	__host__ __device__ inline mat3 ScaleMat3(const vec3 scale)
	{
		const vec3 invScale = vec3(1.0f) / scale;
		return mat3(vec3(invScale.x, 0.0, 0.0),
			vec3(0.0, invScale.y, 0.0),
			vec3(0.0, 0.0, invScale.z));
	}

	__host__ __device__ inline mat3 RotXMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(1.0, 0.0, 0.0),
			vec3(0.0, cosTheta, -sinTheta),
			vec3(0.0, sinTheta, cosTheta));
	}

	__host__ __device__ inline mat3 RotYMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(cosTheta, 0.0, sinTheta),
			vec3(0.0, 1.0, 0.0),
			vec3(-sinTheta, 0.0, cosTheta));
	}

	__host__ __device__ inline mat3 RotZMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(cosTheta, -sinTheta, 0.0),
			vec3(sinTheta, cosTheta, 0.0),
			vec3(0.0, 0.0, 1.0));
	}

	// Builds a composite matrix from three Euler angles, scale and translation vectors
	__host__ __device__ inline BidirectionalTransform CreateCompoundTransform(const vec3& theta, const vec3& translate = vec3(0.0f), const vec3& scale = vec3(1.0f))
	{
		mat3 mat = mat3::Indentity();

		if (scale != vec3(1.0)) { mat *= ScaleMat3(scale); }

		if (theta.x != 0.0) { mat *= RotXMat3(theta.x); }
		if (theta.y != 0.0) { mat *= RotYMat3(theta.y); }
		if (theta.z != 0.0) { mat *= RotZMat3(theta.z); }

		return BidirectionalTransform(translate, mat);
	}

	// Fast construction of orthonormal basis using quarternions to avoid expensive normalisation and branching 
	// From Duf et al's technical report https://graphics.pixar.com/library/OrthonormalB/paper.pdf, inspired by
	// Frisvad's original paper: http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
	__host__ __device__ inline mat3 CreateBasis(vec3 n)
	{
		float s = sign(n.z);
		float a = -1 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat3(vec3(1 + s * n.x * n.x * a, s * b, -s * n.x),
			vec3(b, s + n.y * n.y * a, -n.y),
			n));
	}

	/*__host__ __device__ inline mat4 CreateBasis(vec3 n)
	{
		float s = sign(n.z);
		float a = -1.0 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat4(vec4(1.0f + s * n.x * n.x * a, s * b, -s * n.x, 0.0),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0),
			vec4(n, 0.0),
			vec4(0.0, 0.0, 0.0, 1.0)));

		vec3 tangent = normalize(cross(n, (abs(dot(n, vec3(1.0, 0.0, 0.0))) < 0.5) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
		vec3 cotangent = cross(tangent, n);

		return transpose(mat4(vec4(tangent, 0.0), vec4(cotangent, 0.0), vec4(n, 0.0), vec4(kZero, 1.0)));
	}*/

	__host__ __device__ inline mat3 CreateBasis(vec3 n, vec3 up)
	{
		/*float s = sign(n.z);
		float a = -1 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat3(vec4(1 + s * n.x * n.x * a, s * b, -s * n.x, 0.0f),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0f),
			vec4(n, 0.0f),
			vec4(0.0f, 0.0f, 0.0f, 1.0f)));*/

		const vec3 tangent = normalize(cross(n, up));
		const vec3 cotangent = cross(tangent, n);

		return transpose(mat3(tangent, cotangent, n));
	}
}