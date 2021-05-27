#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	using PCGState = uvec4;
	
	struct RenderCtx
	{
		ivec2          viewportPos;
		ivec2		   viewportSize;
		PCGState       pcgState;
	};
}