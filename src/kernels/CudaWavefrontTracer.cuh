#pragma once

#include "CudaRenderObject.cuh"
#include "CudaAssetContainer.cuh"
#include "math/CudaMath.cuh"
#include "CudaImage.cuh"

namespace Json { class Node; }

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
		class PerspectiveCamera;
		class Material;
		class Tracable;
		class Light;
		class BxDF;
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
				Device::AssetContainer<Device::Tracable>*	cu_deviceTracables;
				Device::AssetContainer<Device::Material>*	cu_deviceMaterials;
				Device::AssetContainer<Device::Light>*		cu_deviceLights;
				Device::AssetContainer<Device::BxDF>*		cu_deviceBxDFs;

				Device::PerspectiveCamera*		cu_camera;
				ivec2							viewportDims;
			};		

		protected:			
			Objects							m_objects;

			float							m_wallTime;
			int								m_frameIdx;
			int								m_maxRayDepth;

			__device__ __forceinline__ bool IsValid(const ivec2& viewportPos) const
			{
				return viewportPos.x >= 0 && viewportPos.x < m_objects.cu_deviceAccumBuffer->Width() &&
					viewportPos.y >= 0 && viewportPos.y < m_objects.cu_deviceAccumBuffer->Height();
			}

			__device__ vec3 Shade(const Ray& incidentRay, const HitCtx& hitCtx, RenderCtx& renderCtx) const;
			__device__ void InitaliseScratchpadObjects() const;

		public:
			__device__ WavefrontTracer() = default;

			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ void SeedRayBuffer(const ivec2& viewportPos) const;
			__device__ void Trace(const uint rayIdx) const;
			__device__ void PreFrame(const float& wallTime, const int frameIdx);
			__device__ void PreBlock() const;
			__device__ void Synchronise(const Objects& params);

		};
	}

	namespace Host
	{
		class Material;
		class Tracable;
		class Light;
		class BxDF;
		class PerspectiveCamera;
		template<typename T> class Array;

		using CompressedRayBuffer = Host::Array<CompressedRay>;
		using PixelFlagsBuffer = Host::Array<uchar>;
		
		class WavefrontTracer : public Host::RenderObject
		{
		private:
			Device::WavefrontTracer*		cu_deviceData;
			Device::WavefrontTracer::Objects m_hostObjects;

			AssetHandle<Host::ImageRGBW>						m_hostAccumBuffer;
			AssetHandle<Host::CompressedRayBuffer>				m_hostCompressedRayBuffer;
			AssetHandle<Host::PixelFlagsBuffer>					m_hostPixelFlagsBuffer;

			AssetHandle<Host::AssetContainer<Host::Tracable>>   m_hostTracables;
			AssetHandle<Host::AssetContainer<Host::Light>>      m_hostLights;

			AssetHandle<Host::PerspectiveCamera>				m_hostPerspectiveCamera;

			dim3                    m_block, m_grid;
			bool					m_isDirty;

		public:
			__host__ WavefrontTracer(const ::Json::Node& node);
			__host__ virtual ~WavefrontTracer();

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
			__host__ static std::string GetAssetTypeString() { return "wavefront"; }

			__host__ virtual void OnDestroyAsset() override final;
			__host__ virtual void FromJson(const ::Json::Node& renderParamsJson, const uint flags) override final;
			__host__ virtual void Bind(RenderObjectContainer& sceneObjects) override final;
			__host__ virtual void Synchronise() override final;

			__host__ void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage);
			__host__ void Iterate(const float wallTime, const float frameIdx);
		};
	}
}