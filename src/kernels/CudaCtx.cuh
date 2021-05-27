#pragma once

#include "math/CudaMath.cuh"
#include "CudaSampler.cuh"

namespace Cuda
{
	namespace Device
	{
		struct RenderCtx
		{
			ivec2          viewportPos;
			ivec2		   viewportSize;
			PCG            pcg;
		};
	}
}