#pragma once

#include "CudaRenderObject.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;

	namespace Host { class PerspectiveCamera; }

	struct PerspectiveCameraParams : public AssetParams
	{
		__host__ __device__ PerspectiveCameraParams();
		__host__ PerspectiveCameraParams(const ::Json::Node& node);

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);
		__host__ bool operator==(const PerspectiveCameraParams&) const;
		
		vec3 position;
		vec3 lookAt;
		float focalPlane;
		float fLength;
		float fStop;
	};
	
	namespace Device
	{
		class PerspectiveCamera : public Device::Asset, public AssetTags<Host::PerspectiveCamera, Device::PerspectiveCamera>
		{
		public:
			__device__ PerspectiveCamera();
			__device__ void CreateRay(RenderCtx& renderCtx) const;
			__device__ void Synchronise(const PerspectiveCameraParams& params)
			{ 
				m_params = params; 
				Prepare();
			}

		private:
			__device__ void Prepare();

		private:
			PerspectiveCameraParams 		m_params;

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
		class PerspectiveCamera : public Host::RenderObject, public AssetTags<Host::PerspectiveCamera, Device::PerspectiveCamera>
		{
		private:
			Device::PerspectiveCamera*				cu_deviceData;

		public:
			__host__ PerspectiveCamera(const ::Json::Node& parentNode);
			__host__ virtual ~PerspectiveCamera() { OnDestroyAsset(); }

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
			__host__ Device::PerspectiveCamera*			GetDeviceInstance() const { return cu_deviceData; }
			__host__ static std::string GetAssetTypeString() { return "perspective"; }
		};
	}
}