#pragma once

#include "FwdDecl.cuh"
#include "core/math/samplers/PCG.cuh"
#include "core/math/Hash.cuh"

namespace Enso
{
	using RNG = PCG;
	struct RenderCtx;

	enum RenderCtxFlags : uchar
	{
		kRenderCtxDebug = 1
	};
	
	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(const uint& probeHash, 
												const uint sample, 
												//const uint accum,
												const uchar& depth, 
												Device::Camera& cam,
												const uchar fl = 0) :
			rng(HashOf(probeHash, uint(depth) + 9871251u, sample)),
			//sampleIdx(sample),
			//accumIdx(accum),
			camera(cam),
			flags(fl),
			debugData(nullptr)
		{
		}

		RNG								rng;
		Device::Camera&					camera;
		uchar							flags;
		//uint							sampleIdx;
		//uint							accumIdx;

		void*							debugData;

		__device__ __inline__ bool IsDebug() const { return (flags & kRenderCtxDebug) || debugData; }
	};
}