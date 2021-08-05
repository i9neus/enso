#pragma once

#include "../CudaRenderObject.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class HitCtx;
	class RenderCtx;
	class Ray;
	class CompressedRay;
	
	namespace Host { class Camera; }

	namespace Device
	{		
		template<typename T> class Array;
		using CompressedRayBuffer = Device::Array<CompressedRay>;
		using PixelFlagsBuffer = Device::Array<uchar>;
		
		struct RenderState
		{
			struct Stats
			{
				uint	deadRays = 0;
			};

			Device::CompressedRayBuffer*	cu_compressedRayBuffer = nullptr;
			Device::PixelFlagsBuffer*		cu_pixelFlagsBuffer = nullptr;
			Device::Array<uint>*			cu_blockRayOccupancy = nullptr;
			Stats*							cu_renderStats = nullptr;
		};
		
		class Camera : public Device::Asset, public AssetTags<Host::Camera, Device::Camera>
		{
		public:
			__device__ Camera() {}

			__device__ virtual void Accumulate(RenderCtx& ctx, const vec3& value) = 0;
			__device__ virtual const Device::RenderState& GetRenderState() const = 0;		
		};
	}

	namespace Host
	{
		template<typename T> class Array;
		using CompressedRayBuffer = Host::Array<CompressedRay>;
		using PixelFlagsBuffer = Host::Array<uchar>;
		using IndirectionBuffer = Host::Array<uint>;
		
		class Camera : public Host::RenderObject, public AssetTags<Host::Camera, Device::Camera>
		{
		public:
			__host__ Camera(const ::Json::Node& parentNode, const std::string& id);
			__host__ virtual ~Camera() {  }

			__host__ virtual void OnDestroyAsset() override;
			__host__ virtual Device::Camera* GetDeviceInstance() const = 0;
			__host__ virtual AssetHandle<Host::ImageRGBW> GetAccumulationBuffer() = 0;
			__host__ virtual void ClearRenderState() = 0;
			__host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override;
			__host__ virtual void SeedRayBuffer() = 0;
			__host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const {};

			__host__ static std::string GetAssetTypeString() { return "camera"; }
			__host__ bool IsLive() const { return m_isLiveCamera; }
			__host__ AssetHandle<Host::ManagedObject<Device::RenderState::Stats>> GetRenderStats() { return m_hostRenderStats; }
			__host__ AssetHandle<Host::CompressedRayBuffer> GetCompressedRayBuffer() { return m_hostCompressedRayBuffer; }

		protected:
			bool	m_isLiveCamera;

			AssetHandle<Host::ManagedObject<Device::RenderState::Stats>>	m_hostRenderStats;
			AssetHandle<Host::CompressedRayBuffer>							m_hostCompressedRayBuffer;
			AssetHandle<Host::IndirectionBuffer>							m_hostRayIndirectionBuffer;
			AssetHandle<Host::PixelFlagsBuffer>								m_hostPixelFlagsBuffer;
			AssetHandle<Host::Array<uint>>									m_hostBlockRayOccupancy;
		};
	}
}