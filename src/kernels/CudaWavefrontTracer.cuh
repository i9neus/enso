#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaPerspectiveCamera.cuh"
#include "CudaCtx.cuh"
#include "CudaTracable.cuh"
#include "CudaAssetContainer.cuh"

namespace Cuda
{
	namespace Device
	{
		class WavefrontTracer : public ManagedPair<Device::WavefrontTracer>
		{		
		protected:
			WavefrontTracer();

			Device::ImageRGBW*				cu_deviceAccumBuffer;
			Device::CompressedRayBuffer*	cu_deviceCompressedRayBuffer;
			Device::AssetContainer<Device::Tracable>* cu_deviceTracables;

			ivec2							m_viewportDims;
			Device::PerspectiveCamera		m_camera;

		public:
			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;

			__device__ void SeedRayBuffer(const ivec2& viewportPos) const;
			__device__ void Trace(const ivec2& viewportPos) const;
			__device__ RenderCtx CreateRenderCtx(const ivec2& viewportPos, const uint depth) const;

		protected:
			__device__ inline bool IsValid(const ivec2& viewportPos) const
			{
				return viewportPos.x >= 0 && viewportPos.x < cu_deviceAccumBuffer->Width() &&
					viewportPos.y >= 0 && viewportPos.y < cu_deviceAccumBuffer->Height();
			}
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
			Asset<Host::AssetContainer<Host::Tracable>>   m_hostTracables;

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