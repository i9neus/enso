#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaPerspectiveCamera.cuh"
#include "CudaCtx.cuh"
#include "CudaAssetContainer.cuh"
#include "CudaManagedArray.cuh"
#include "bxdfs/CudaLambert.cuh"

#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornell.cuh"

namespace Cuda
{
	namespace Host { class WavefrontTracer; }	
	
	namespace Device
	{
		using CompressedRayBuffer = Device::Array<CompressedRay>;
		
		class WavefrontTracer : public Device::Asset, public AssetTags<Host::WavefrontTracer, Device::WavefrontTracer>
		{		
			friend class Host::WavefrontTracer;

		protected:
			Device::ImageRGBW*				cu_deviceAccumBuffer;
			Device::CompressedRayBuffer*	cu_deviceCompressedRayBuffer;
			//Device::AssetContainer<Device::Tracable>* cu_deviceTracables;
			Device::Cornell*				cu_cornell;
			Device::Sphere*					cu_sphere;
			Device::LambertBRDF*			cu_lambert;

			ivec2							m_viewportDims;
			Device::PerspectiveCamera		m_camera;

			float							m_wallTime;
			int								m_frameIdx;
			int								m_maxRayDepth;

			WavefrontTracer() { memset(this, 0, sizeof(WavefrontTracer)); }

			__device__ inline bool IsValid(const ivec2& viewportPos) const
			{
				return viewportPos.x >= 0 && viewportPos.x < cu_deviceAccumBuffer->Width() &&
					viewportPos.y >= 0 && viewportPos.y < cu_deviceAccumBuffer->Height();
			}

			__device__ vec3 Shade(const Ray& incidentRay, const HitCtx& hitCtx, RenderCtx& renderCtx) const;

		public:
			__device__ WavefrontTracer(Device::ImageRGBW* deviceAccumBuffer, 
				Device::CompressedRayBuffer* deviceCompressedRayBuffer, 
				Device::Cornell* cornell, 
				Device::Sphere* sphere, 
				Device::LambertBRDF* lambert,
				const ivec2& viewportDims) :
				cu_deviceAccumBuffer(deviceAccumBuffer),
				cu_deviceCompressedRayBuffer(deviceCompressedRayBuffer),
				cu_cornell(cornell),
				cu_sphere(sphere),
				cu_lambert(lambert),
				m_viewportDims(viewportDims) {}

			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ void SeedRayBuffer(const ivec2& viewportPos) const;
			__device__ void Trace(const uint rayIdx) const;
			__device__ inline RenderCtx CreateRenderCtx(const CompressedRay& compressed) const;
			__device__ inline RenderCtx CreateRenderCtx(const ivec2 viewportPos, const int depth = 0) const;
			__device__ void PreFrame(const float& wallTime, const int frameIdx);

		};
	}

	namespace Host
	{
		using CompressedRayBuffer = Host::Array<CompressedRay>;
		
		class WavefrontTracer : public Host::Asset
		{
		private:
			Device::WavefrontTracer*		cu_deviceData;
			Device::WavefrontTracer         m_hostData;

			AssetHandle<Host::ImageRGBW>						m_hostAccumBuffer;
			AssetHandle<Host::CompressedRayBuffer>				m_hostCompressedRayBuffer;
			AssetHandle<Host::AssetContainer<Host::Tracable>>   m_hostTracables;
			AssetHandle<Host::Cornell>                          m_hostCornell;
			AssetHandle<Host::Sphere>                           m_hostSphere;
			AssetHandle<Host::LambertBRDF>                      m_hostLambert;

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