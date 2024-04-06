#pragma once

#include "FwdDecl.cuh"
#include "core/math/Sampler.cuh"
#include "core/Hash.h"

namespace Enso
{
	using RNG = PseudoRNG;
	struct RenderCtx;

	enum RenderCtxFlags : uchar
	{
		kRenderCtxDebug = 1
	};
	
	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(const uint& probeHash, 
												const uint accumIdx, 
												const uchar& depth, 
												Device::ICamera2D& cam,
												const uchar fl = 0) :
			hash(HashOf(probeHash, uint(depth) + 9871251u, accumIdx)),
			rng(hash),
			camera(cam),
			flags(fl),
			debugData(nullptr)
		{
		}

		uint			hash;
		RNG				rng;
		Device::ICamera2D& camera;
		uchar			flags;

		void*			debugData;

		__device__ __inline__ bool IsDebug() const { return (flags & kRenderCtxDebug) || debugData; }
	};
}