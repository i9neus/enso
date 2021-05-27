#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"

namespace Cuda
{
	namespace Device
	{
		class WavefrontTracer
		{		
		protected:
			__device__ WavefrontTracer() = default;

			Device::ImageRGBW*			m_deviceAccumBuffer;
			Device::PackedRayBuffer*	m_devicePackedRayBuffer;

		public:
			__device__ void Composite(const ivec2 fragCoord, Device::ImageRGBA* deviceOutputImage) const;

			__device__ void SeedRayBuffer(ivec2 fragCoord);
		};
	}

	namespace Host
	{
		class WavefrontTracer : public Device::WavefrontTracer, public AssetBase
		{
		private:
			Device::WavefrontTracer*		cu_deviceTracer;

			Asset<Host::ImageRGBW>			m_hostAccumBuffer;
			Asset<Host::PackedRayBuffer>    m_hostPackedRayBuffer;

			cudaStream_t			m_hostStream;

			dim3                    m_block, m_grid;

		public:
			__host__ WavefrontTracer(cudaStream_t hostStream);
			__host__ ~WavefrontTracer() = default;

			__host__ virtual void OnDestroyAsset() override final;

			__host__ void Composite(Asset<Host::ImageRGBA>& hostOutputImage);

			__host__ void Iterate();
		};
	}
}