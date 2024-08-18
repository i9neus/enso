#pragma once

#include "vec/IVec2.cuh"
#include "vec/IVec3.cuh"
#include "vec/IVec4.cuh"

#include "vec/Vec2.cuh"
#include "vec/Vec3.cuh"
#include "vec/Vec4.cuh"

#include "mat/Mat2.cuh"
#include "mat/Mat3.cuh"
#include "mat/Mat4.cuh"

#include "MathUtils.cuh"

namespace Enso
{
	__host__ __device__ __forceinline__ vec3	mod2(vec3 a, vec3 b) { return fmod(fmod(a, b) + b, b); }
	__host__ __device__ __forceinline__ int		sum(ivec2 a) { return a.x + a.y; }
	__host__ __device__ __forceinline__ float	luminance(vec3 v) { return v.x * 0.17691f + v.y * 0.8124f + v.z * 0.01063f; }
	__host__ __device__ __forceinline__ float	mean(vec3 v) { return (v.x + v.y + v.z) / 3; }

	__host__ __device__ __forceinline__ float	Volume(const vec3& v) { return v.x * v.y * v.z; }
	__host__ __device__ __forceinline__ int		Volume(const ivec3& v) { return v.x * v.y * v.z; }
	__host__ __device__ __forceinline__ uint	Volume(const uvec3& v) { return v.x * v.y * v.z; }
	__host__ __device__ __forceinline__ float	Area(const vec2& v) { return v.x * v.y; }
	__host__ __device__ __forceinline__ int		Area(const ivec2& v) { return v.x * v.y; }
	__host__ __device__ __forceinline__ uint	Area(const uvec2& v) { return v.x * v.y; }

	__host__ __device__ __forceinline__ vec2	MinMax(const vec2& v, const float& f) { return vec2(fminf(v.x, f), fmaxf(v.y, f)); }

	__host__ __device__ __forceinline__ bool	IsPointInUnitBox(const vec3& v)
	{
		return v.x > -0.5f && v.x <= 0.5f && v.y > -0.5 && v.y <= 0.5f && v.z > -0.5f && v.z <= 0.5f;
	}
	__host__ __device__ __forceinline__ bool	IsPointInBox(const vec3& v, const vec3& lower, const vec3& upper)
	{
		return v.x > lower.x && v.x <= upper.x && v.y > lower.y && v.y <= upper.y && v.z > lower.z && v.z <= upper.z;
	}

	__host__ __device__ __forceinline__ vec3	PolarToCartesian(const vec2& polar)
	{
		const float sinTheta = sin(polar.x);
		return vec3(sin(polar.y) * sinTheta, cos(polar.x), cos(polar.y) * sinTheta);
	}

	template<typename VecType>
	__host__ __device__ __forceinline__ VecType SafeNormalize(const VecType& v)
	{
		constexpr float kSafeNormEpsilon = 1e-20f;
		const float l2 = length2(v);
		return (l2 < kSafeNormEpsilon) ? VecType(0.f) : v / sqrtf(l2);
	}

	#define kOne vec3(1.0f)
	#define kZero vec3(0.0f)

	#define kZero4f vec4(0.0f)
	#define kZero3f vec3(0.0f)
	#define kZero2f vec2(0.0f)
	#define kZero4i ivec4(0)
	#define kZero3i ivec3(0)
	#define kZero2i ivec2(0)
	#define kZero4u uvec4(0u)
	#define kZero3u uvec3(0u)
	#define kZero2u uvec2(0u)

	#define	kMinMaxReset vec2(kFltMax, -kFltMax)

}
