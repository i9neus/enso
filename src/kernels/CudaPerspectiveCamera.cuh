#pragma once

#include <stdlib.h>

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;

	namespace Host { class PerspectiveCamera; }
	
	namespace Device
	{
		class PerspectiveCamera : public Device::Asset, public AssetTags<Host::PerspectiveCamera, Device::PerspectiveCamera>
		{
		public:
			struct Params
			{
			}
			m_params;

		public:
			__device__ PerspectiveCamera();
			__device__ void CreateRay(CompressedRay& newRay, RenderCtx& renderCtx) const;
			__device__ void OnSyncParameters(const Params& params) { m_params = params; }

		private:
			bool		m_useHaltonSpectralSampler;
			vec2		m_cameraPos;
			vec2        m_cameraLook;
			vec2		m_cameraFLength;
			vec2        m_cameraFStop;
		};
	}

	namespace Host
	{
		class PerspectiveCamera : public Host::Asset, public AssetTags<Host::PerspectiveCamera, Device::PerspectiveCamera>
		{
		private:
			Device::PerspectiveCamera*				cu_deviceData;

		public:
			__host__ PerspectiveCamera();
			__host__ virtual ~PerspectiveCamera() { OnDestroyAsset(); }

			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       OnJson(const Json::Node& jsonNode) override final;
			__host__ Device::PerspectiveCamera*			GetDeviceInstance() const { return cu_deviceData; }
		};
	}
}