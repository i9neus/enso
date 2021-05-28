#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaPerspectiveCamera.cuh"
#include "CudaCtx.cuh"
#include "CudaTracable.cuh"
#include "CudaArray.cuh"

namespace Cuda
{
	namespace Device
	{
		class WavefrontTracer
		{		
		protected:
			WavefrontTracer();

			Device::ImageRGBW*				m_deviceAccumBuffer;
			Device::CompressedRayBuffer*	m_deviceCompressedRayBuffer;

			ivec2							m_viewportDims;
			Device::PerspectiveCamera		m_camera;

		public:
			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;

			__device__ void SeedRayBuffer(const ivec2& viewportPos) const;
			__device__ void Trace(const ivec2& viewportPos) const;
			__device__ RenderCtx CreateRenderCtx(const ivec2& viewportPos, const uint depth) const;
		};
	}

	namespace Host
	{
		class WavefrontTracer : public Device::WavefrontTracer, public AssetBase
		{
		private:
			Device::WavefrontTracer*		cu_deviceTracer;

			Asset<Host::ImageRGBW>					m_hostAccumBuffer;
			Asset<Host::CompressedRayBuffer>		m_hostCompressedRayBuffer;
			Asset<Host::Array<Device::Tracable*>>   m_hostTracables;

			cudaStream_t			m_hostStream;
			dim3                    m_block, m_grid;

		public:
			__host__ WavefrontTracer(cudaStream_t hostStream);

			__host__ virtual void OnDestroyAsset() override final;

			__host__ void Composite(Asset<Host::ImageRGBA>& hostOutputImage);

			__host__ void Iterate();
		};
	}
}