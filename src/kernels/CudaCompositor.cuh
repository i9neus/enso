#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"

namespace Cuda
{	
	void CopyImageToD3DTexture(unsigned int width, unsigned int height, const Image* image, cudaSurfaceObject_t cuSurface, cudaStream_t streamToRun, unsigned int* signal);
}