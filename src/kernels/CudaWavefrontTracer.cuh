﻿#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaPerspectiveCamera.cuh"
#include "CudaCtx.cuh"
#include "CudaAssetContainer.cuh"

#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornell.cuh"

namespace Cuda
{
	namespace Host { class WavefrontTracer; }
	
	namespace Device
	{
		class WavefrontTracer : public Device::Asset, public AssetTags<Host::WavefrontTracer, Device::WavefrontTracer>
		{		
			friend class Host::WavefrontTracer;

		protected:
			Device::ImageRGBW*				cu_deviceAccumBuffer;
			Device::CompressedRayBuffer*	cu_deviceCompressedRayBuffer;
			//Device::AssetContainer<Device::Tracable>* cu_deviceTracables;
			Device::Cornell*				cu_cornell;

			ivec2							m_viewportDims;
			Device::PerspectiveCamera		m_camera;

			float							m_wallTime;
			int								m_frameIdx;

			WavefrontTracer();
			__device__ inline bool IsValid(const ivec2& viewportPos) const
			{
				return viewportPos.x >= 0 && viewportPos.x < cu_deviceAccumBuffer->Width() &&
					viewportPos.y >= 0 && viewportPos.y < cu_deviceAccumBuffer->Height();
			}

		public:
			__device__ WavefrontTracer(Device::ImageRGBW* deviceAccumBuffer, Device::CompressedRayBuffer* deviceCompressedRayBuffer, Device::Cornell* cornell, const ivec2& viewportDims) :
				cu_deviceAccumBuffer(deviceAccumBuffer),
				cu_deviceCompressedRayBuffer(deviceCompressedRayBuffer),
				cu_cornell(cornell),
				m_viewportDims(viewportDims) {}

			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ void SeedRayBuffer(const ivec2& viewportPos) const;
			__device__ void Trace(const ivec2& viewportPos) const;
			__device__ RenderCtx CreateRenderCtx(const CompressedRay& compressed, const uint depth) const;
			__device__ RenderCtx CreateRenderCtx(const ivec2 viewportPos) const;
			__device__ void PreFrame(const float& wallTime, const int frameIdx);

		};
	}

	namespace Host
	{
		class WavefrontTracer : public Host::Asset
		{
		private:
			Device::WavefrontTracer*		cu_deviceData;
			Device::WavefrontTracer         m_hostData;

			AssetHandle<Host::ImageRGBW>						m_hostAccumBuffer;
			AssetHandle<Host::CompressedRayBuffer>				m_hostCompressedRayBuffer;
			AssetHandle<Host::AssetContainer<Host::Tracable>>   m_hostTracables;
			AssetHandle<Host::Cornell>                          m_hostCornell;

			cudaStream_t			m_hostStream;
			dim3                    m_block, m_grid;

		public:
			__host__ WavefrontTracer(cudaStream_t hostStream);
			__host__ ~WavefrontTracer() { OnDestroyAsset(); }

			__host__ virtual void OnDestroyAsset() override final;

			__host__ void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage);

			__host__ void Iterate(const float wallTime, const float frameIdx);
		};
	}
}