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
				vec2 cameraFStop;
				vec2 cameraPos;
				vec2 cameraLook;
				vec2 cameraFLength;
				bool useHaltonSpectralSampler;
			};

		public:
			__device__ PerspectiveCamera();
			__device__ void CreateRay(CompressedRay& newRay, RenderCtx& renderCtx) const;
			__device__ void OnSyncParameters(const Params& params) 
			{ 
				m_params = params; 
				Prepare();
			}

		private:
			__device__ void Prepare();

		private:
			Params 		m_params;

			mat3		m_basis;
			vec3		m_cameraPos;
			float		m_d1, m_d2;
			float		m_focalLength;
			float		m_focalDistance;
			float		m_fStop;
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