#pragma once

#include "CudaCommonIncludes.h"

namespace Cuda
{
	void CompositeBuffers(unsigned int width, unsigned int height, cudaSurfaceObject_t cuSurface, float time, cudaStream_t streamToRun);
}