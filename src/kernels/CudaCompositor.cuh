#pragma once

#include "CudaCommonIncludes.cuh"

namespace Cuda
{	
	void CopyImageToD3DTexture(unsigned int width, unsigned int height, const CudaImage& image, cudaSurfaceObject_t cuSurface, cudaStream_t streamToRun, unsigned int* signal);
}