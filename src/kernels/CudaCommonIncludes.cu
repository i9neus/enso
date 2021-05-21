#include "CudaCommonIncludes.cuh"

void CudaImage::create(unsigned int width, unsigned int height)
{
	m_width = width;
	m_height = height;
	checkCudaErrors(cudaMalloc((void**)&c_data, sizeof(float4) * m_width * m_height));
}

void CudaImage::destroy()
{
	checkCudaErrors(cudaFree((void*)c_data));
	c_data = nullptr;
}