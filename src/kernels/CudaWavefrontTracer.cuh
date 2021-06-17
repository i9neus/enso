#pragma once

#include "CudaAssetContainer.cuh"
#include "generic/JsonUtils.h"
#include "math/CudaMath.cuh"
#include "CudaImage.cuh"

namespace Cuda
{
	class HitCtx;
	class RenderCtx;
	class Ray;
	class CompressedRay;
	
	namespace Host { class WavefrontTracer; }	

	enum TracerPixelFlags : uchar { kTracerPixelChanged = 1 };
	
	namespace Device
	{
		class Cornell;
		class Sphere;
		class LambertBRDF;
		class Plane;
		class KIFS;
		class SimpleMaterial;
		class PerspectiveCamera;
		template<typename T> class Array;
		
		using CompressedRayBuffer = Device::Array<CompressedRay>;
		using PixelFlagsBuffer = Device::Array<uchar>;

		class WavefrontTracer : public Device::Asset, public AssetTags<Host::WavefrontTracer, Device::WavefrontTracer>
		{		
			friend class Host::WavefrontTracer;
		public:
			struct Objects
			{
				Device::ImageRGBW*				cu_deviceAccumBuffer;
				Device::CompressedRayBuffer*	cu_deviceCompressedRayBuffer;
				Device::PixelFlagsBuffer*		cu_pixelFlagsBuffer;
				//Device::AssetContainer<Device::Tracable>* cu_deviceTracables;
				Device::Cornell*				cu_cornell;
				Device::Sphere*					cu_sphere;
				Device::LambertBRDF*			cu_lambert;
				Device::Plane*                  cu_groundPlane;
				Device::KIFS*                   cu_kifs;
				Device::SimpleMaterial*			cu_simpleMaterial;
				Device::PerspectiveCamera*		cu_camera;
				ivec2							viewportDims;
			};		

		protected:			
			Objects							m_objects;

			float							m_wallTime;
			int								m_frameIdx;
			int								m_maxRayDepth;

			WavefrontTracer() { memset(this, 0, sizeof(WavefrontTracer)); }

			__device__ __forceinline__ bool IsValid(const ivec2& viewportPos) const
			{
				return viewportPos.x >= 0 && viewportPos.x < m_objects.cu_deviceAccumBuffer->Width() &&
					viewportPos.y >= 0 && viewportPos.y < m_objects.cu_deviceAccumBuffer->Height();
			}

			__device__ vec3 Shade(const Ray& incidentRay, const HitCtx& hitCtx, RenderCtx& renderCtx) const;

		public:
			__device__ WavefrontTracer(const Objects* objects);

			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ void SeedRayBuffer(const ivec2& viewportPos) const;
			__device__ void Trace(const uint rayIdx) const;
			__device__ void PreFrame(const float& wallTime, const int frameIdx);
			//__device__ void OnSyncParams(const Params* params);

		};
	}

	namespace Host
	{
		class Cornell;
		class Sphere;
		class LambertBRDF;
		class Plane;
		class KIFS;
		class SimpleMaterial;
		class Tracable;
		class PerspectiveCamera;
		template<typename T> class Array;

		using CompressedRayBuffer = Host::Array<CompressedRay>;
		using PixelFlagsBuffer = Host::Array<uchar>;
		
		class WavefrontTracer : public Host::Asset
		{
		private:
			Device::WavefrontTracer*		cu_deviceData;
			Device::WavefrontTracer::Objects m_hostData;

			AssetHandle<Host::ImageRGBW>						m_hostAccumBuffer;
			AssetHandle<Host::CompressedRayBuffer>				m_hostCompressedRayBuffer;
			AssetHandle<Host::PixelFlagsBuffer>					m_hostPixelFlagsBuffer;

			AssetHandle<Host::AssetContainer<Host::Tracable>>   m_hostTracables;
			AssetHandle<Host::Cornell>                          m_hostCornell;
			AssetHandle<Host::Sphere>                           m_hostSphere;
			AssetHandle<Host::Plane>                            m_hostGroundPlane;
			AssetHandle<Host::KIFS>								m_hostKifs;

			AssetHandle<Host::LambertBRDF>                      m_hostLambert;

			AssetHandle<Host::SimpleMaterial>					m_hostSimpleMaterial;

			AssetHandle<Host::PerspectiveCamera>				m_hostPerspectiveCamera;

			cudaStream_t			m_hostStream;
			dim3                    m_block, m_grid;
			bool					m_isDirty;

		public:
			__host__ WavefrontTracer(cudaStream_t hostStream);
			__host__ ~WavefrontTracer() { OnDestroyAsset(); }

			__host__ virtual void OnDestroyAsset() override final;
			__host__ void OnJson(const Json::Node& renderParamsJson);
			__host__ void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage);
			__host__ void Iterate(const float wallTime, const float frameIdx);
		};
	}
}