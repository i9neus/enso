#pragma once

#include "../CudaRenderObject.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class RenderCtx;
	
	namespace Host { class Camera; }

	namespace Device
	{		
		class Camera : public Device::Asset, public AssetTags<Host::Camera, Device::Camera>
		{
		public:
			__device__ Camera() {}

			__device__ virtual void CreateRay(RenderCtx& renderCtx) const = 0;
		};
	}

	namespace Host
	{
		class Camera : public Host::RenderObject, public AssetTags<Host::Camera, Device::Camera>
		{
		public:
			__host__ Camera(const ::Json::Node& parentNode) {}
			__host__ virtual ~Camera() {  }

			__host__ virtual Device::Camera* GetDeviceInstance() const = 0;
			__host__ static std::string GetAssetTypeString() { return "camera"; }
		};
	}
}