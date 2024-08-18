#pragma once

#include "core/math/Math.cuh"
#include "Ray.cuh"

namespace Enso
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
		float sca;
		mat3 fwd;
		mat3 inv;

		__host__ __device__ BidirectionalTransform() 
		{
			MakeIdentity();
		}
		__host__ __device__ BidirectionalTransform(const vec3& tr, const vec3& euler, const float& sc)
		{ 
			MakeCompound(tr, euler, sc);
		}

		__host__ __device__ void MakeCompound(const vec3& tr, const vec3& euler, const float& sc)
		{			
			// Construct a compound rotation matrix from the Euler angles
			fwd = mat3::Identity();
			if (euler.x != 0.0) { fwd *= RotXMat3(euler.x); }
			if (euler.y != 0.0) { fwd *= RotYMat3(euler.y); }
			if (euler.z != 0.0) { fwd *= RotZMat3(euler.z); }
			inv = transpose(fwd);

			trans = tr;
			sca = sc;
		}

		__host__ __device__ void MakeCompound(const vec3& tr, const float& sc)
		{
			fwd = inv = mat3::Identity();
			trans = tr;
			sca = sc;
		}

		__host__ __device__ void MakeIdentity()
		{
			trans = kZero;
			sca = 1.;
			fwd = inv = mat3::Identity();
		}

		__host__ __device__ __forceinline__ RayBasic RayToObjectSpace(const RayBasic& world) const
		{
			RayBasic object;
			object.o = world.o - trans;
			object.d = world.d + object.o;
			object.o = fwd * object.o / sca;
			object.d = (fwd * object.d / sca) - object.o;
			return object;
		}

		__host__ __device__ __forceinline__ vec3 NormalToWorldSpace(const vec3& n) const
		{
			return inv * n;
		}

		__host__ __device__ __forceinline__ vec3 PointToWorldSpace(const vec3& p) const
		{
			return (inv * p) * sca + trans;
		}
	};	

	__host__ __device__ __forceinline__ vec2 PixelToNormalisedScreen(const vec2& p, const vec2& viewportRes)
	{
		// Normalisation happens relative to screen width. The final range is:
		// [ [-1, -viewportRes.y/viewportRes.x], [1, viewportRes.y/viewportRes.x] ] or
		// [ [-1, -0.5625], [1, 0.5625]] for 1200x675

		return 2.f * (p - viewportRes * 0.5f) / float(viewportRes.x);
	}

	__host__ __device__ __forceinline__ vec2 NormalisedScreenToPixel(const vec2& p, const vec2& viewportRes)
	{
		return (p * float(viewportRes.x) * 0.5f) + viewportRes * 0.5f;
	}
}