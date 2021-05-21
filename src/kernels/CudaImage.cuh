#pragma once

#include "CudaCommonIncludes.cuh"

namespace Cuda
{
	class Image
	{
	protected:
		Image(const unsigned int width, const unsigned int height, float4* data) : m_width(width), m_height(height), m_data(data) {}

		unsigned int	m_width;
		unsigned int	m_height;
		float4*			m_data;

	public:
		__device__ float4* GetData() { return m_data; }
		__device__ unsigned int Width() const { return m_width; }
		__device__ unsigned int Height() const { return m_height; }

		static Image* Create(unsigned int width, unsigned int height)
		{
			Image* deviceImage;
			float4* deviceData;

			checkCudaErrors(cudaMalloc((void**)&deviceData, sizeof(float4) * width * height));
			Image hostImage(width, height, deviceData);

			checkCudaErrors(cudaMalloc((void**)&deviceImage, sizeof(Image)));
			checkCudaErrors(cudaMemcpy(deviceImage, &hostImage, sizeof(Image), cudaMemcpyHostToDevice));

			return deviceImage;
		}

		static void Destroy(Image* deviceImage)
		{
			checkCudaErrors(cudaFree((void*)(deviceImage->m_data)));
			checkCudaErrors(cudaFree((void*)deviceImage));
		}
	};
}