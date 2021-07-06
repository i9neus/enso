#pragma once

#include "CudaMathUtils.cuh"

namespace Json { class Node; }

namespace Cuda
{
	__host__ __device__ __forceinline__ mat3 ScaleMat3(const vec3 scale)
	{
		const vec3 invScale = vec3(1.0f) / scale;
		return mat3(vec3(invScale.x, 0.0, 0.0),
			vec3(0.0, invScale.y, 0.0),
			vec3(0.0, 0.0, invScale.z));
	}

	__host__ __device__ __forceinline__ mat3 RotXMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(1.0, 0.0, 0.0),
			vec3(0.0, cosTheta, -sinTheta),
			vec3(0.0, sinTheta, cosTheta));
	}

	__host__ __device__ __forceinline__ mat3 RotYMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(cosTheta, 0.0, sinTheta),
			vec3(0.0, 1.0, 0.0),
			vec3(-sinTheta, 0.0, cosTheta));
	}

	__host__ __device__ __forceinline__ mat3 RotZMat3(const float theta)
	{
		const float cosTheta = cosf(theta), sinTheta = sinf(theta);
		return mat3(vec3(cosTheta, -sinTheta, 0.0),
			vec3(sinTheta, cosTheta, 0.0),
			vec3(0.0, 0.0, 1.0));
	}
	
	class BidirectionalTransform
	{
	public:
		vec3 trans;
		vec3 rot;
		vec3 scale;

		mat3 fwd;
		mat3 inv;
		mat3 nInv;

		__host__ void FromJson(const ::Json::Node& json, const uint flags);
		__host__ void ToJson(::Json::Node& json) const;
		__host__ bool operator==(const BidirectionalTransform&) const;

		__host__ __device__ BidirectionalTransform()
		{
			MakeIdentity();
		}

		__host__ BidirectionalTransform(const ::Json::Node& json, const uint flags);

		__host__ __device__ __forceinline__ BidirectionalTransform(const vec3& t, const vec3& r, const vec3& s)
		{
			Create(t, r, s);
		}

		__host__ __device__ __forceinline__ void Create(const vec3& t, const vec3& r, const vec3& s)
		{
			trans = t;
			rot = r;
			scale = s;

			fwd = mat3::Indentity();

			if (rot.x != 0.0) { fwd *= RotXMat3(toRad(rot.x)); }
			if (rot.y != 0.0) { fwd *= RotYMat3(toRad(rot.y)); }
			if (rot.z != 0.0) { fwd *= RotZMat3(toRad(rot.z)); }

			nInv = transpose(fwd);

			if (scale != vec3(1.0)) { fwd *= ScaleMat3(scale); }

			inv = inverse(fwd);
		}

		__host__ __device__ __forceinline__ void MakeIdentity()
		{
			trans = 0.0f;
			rot = 0.0f;
			scale = 1.0f;

			fwd = inv = nInv = mat3::Indentity();
		}

		__device__ __forceinline__ vec3 PointToWorldSpace(const vec3& object) const
		{
			return (inv * object) + trans;
		}
	};

	// Fast construction of orthonormal basis using quarternions to avoid expensive normalisation and branching 
	// From Duf et al's technical report https://graphics.pixar.com/library/OrthonormalB/paper.pdf, inspired by
	// Frisvad's original paper: http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
	__host__ __device__ __forceinline__ mat3 CreateBasis(vec3 n)
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

	__host__ __device__ __forceinline__ mat3 CreateBasis(vec3 n, vec3 up)
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