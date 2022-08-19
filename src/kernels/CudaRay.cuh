#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	#define kSpecularPdf  65000.0f;

	enum RayFlags : uchar
	{
		// Propoerty flags
		kRayIndirectSample =		1 << 0,
		kRayDirectLightSample =		1 << 1,
		kRayDirectBxDFSample =		1 << 2,
		kRaySpecular =				1 << 3,
		kRayLightProbe =			1 << 4,
		kRayScattered =				1 << 5,
		kRayDistantLightSample =    1 << 6,
		
		// Flags that persist throughout the path
		kRayPersistentFlags =		kRayLightProbe | kRayScattered,
		// Constitutes a direct sample
		kRayDirectSample =			kRayDirectLightSample | kRayDirectBxDFSample
	};

	struct RayBasic
	{
	public:
		vec3 o, d;

		__device__ RayBasic() {}
		__device__ RayBasic(const vec3& o_, const vec3& d_) : o(o_), d(d_) {}
		__device__ inline vec3 PointAt(const float& t) const { return o + d * t; }		
	};

	__device__ inline RayBasic RayToObjectSpace(const RayBasic& world, const BidirectionalTransform& bdt) 
	{
		RayBasic object;
		object.o = world.o - bdt.trans();
		object.d = world.d + object.o;
		object.o = bdt.fwd * object.o;
		object.d = (bdt.fwd * object.d) - object.o;
		return object;
	}

	__device__ __forceinline__ vec3 PointToObjectSpace(const vec3& p, const BidirectionalTransform& bdt) { return bdt.fwd * (p - bdt.trans()); }
	__device__ __forceinline__ vec3 PointToWorldSpace(const vec3& p, const BidirectionalTransform& bdt) { return (bdt.inv * p) + bdt.trans(); }
	__device__ __forceinline__ vec3 NormalToWorldSpace(const vec3& n, const BidirectionalTransform& bdt) { return bdt.nInv * n; }

	struct CompressedRay
	{
		__host__ __device__ CompressedRay() :
			flags(0),
			sampleIdx(0),
			depth(0) {}

		RayBasic	od;				// 24 bytes
		vec3		weight;			// 12 bytes
		half		pdf;			// 2 bytes
		uchar		lightId;		// 1 byte
		uint		accumIdx;		// 4 bytes		
		uchar		flags;			// 1 byte
		uchar		depth;			// 1 byte
		uint		sampleIdx;		// 4 bytes
		vec3		probeDir;		// 4 bytes

		__device__ __forceinline__ void Reset()
		{
			memset(this, 0, sizeof(CompressedRay));
		}

		__device__ __forceinline__ ivec2 GetViewportPos() const	{ return ivec2(accumIdx >> 16, accumIdx & 0xffff); }
		__device__ __forceinline__ void SetViewportPos(const int x, const int y) { accumIdx = (x << 16) | (y & 0xffff);  }
		__device__ __forceinline__ void SetViewportPos(const ivec2 v) { accumIdx = (v.x << 16) | (v.y & 0xffff); }

		__device__ __forceinline__ void Kill() { flags &= kRayPersistentFlags; }
		__device__ __forceinline__ bool IsAlive() const { return (flags & ~kRayPersistentFlags) != 0; }
		__device__ __forceinline__ void Set(const uchar f) { flags |= f; }
		__device__ __forceinline__ void Unset(const uchar f) { flags &= ~f; }
	};

	// The "full fat" ray objects that most methods will refer to
	struct Ray
	{				
		RayBasic	od;
		float		tNear;             // The parameterised intersection along the ray, defined in cartesian coordinates as o + d * tNear			
		uchar		flags;
		uchar		depth;

		Ray() = default;
		__device__ Ray(const CompressedRay& comp) :
			od(comp.od),
			tNear(FLT_MAX),
			flags(comp.flags),
			depth(comp.depth)
		{
		}

		__device__ __forceinline__ vec3 HitPoint() const { return od.o + od.d * tNear; }
		__device__ __forceinline__ vec3 PointAt(const float& t) const { return od.o + od.d * t; }
		__device__ __forceinline__ bool IsDirectSample() const { return flags & kRayDirectSample; }
		__device__ __forceinline__ bool IsIndirectSample() const { return flags & kRayIndirectSample; }
		__device__ __forceinline__ bool IsDistantLightSample() const { return flags & kRayDistantLightSample; }
	};
}
