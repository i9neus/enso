#pragma once

#include "CudaCamera.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;
	class PseudoRNG;
	class QuasiRNG;

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
		float displayGamma;

		ivec2 viewportDims;
	};
	
	namespace Device
	{
		class PerspectiveCamera : public Device::Camera
		{
		public:
			struct Objects
			{
				Device::RenderState renderState;
				Device::ImageRGBW* cu_accumBuffer = nullptr;
			};

			__device__ PerspectiveCamera();
			__device__ virtual void Accumulate(RenderCtx& ctx, const vec3& value) override final;
			__device__ void SeedRayBuffer(const ivec2& viewportPos);
			__device__ virtual const Device::RenderState& GetRenderState() const override final { return m_objects.renderState; }
			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;

			__device__ void Synchronise(const PerspectiveCameraParams& params)
			{ 
				m_params = params; 
				Prepare();
			}
			__device__ void Synchronise(const Objects& objects)
			{
				m_objects = objects;
			}

		private:
			__device__ void Prepare();
			__device__ void CreateRay(const ivec2& viewportPos, CompressedRay& ray) const;

		private:
			PerspectiveCameraParams 		m_params;
			Objects							m_objects;

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
		class PerspectiveCamera : public Host::Camera
		{
		private:
			Device::PerspectiveCamera*				cu_deviceData;
			PerspectiveCameraParams					m_params;

		public:
			__host__ PerspectiveCamera(const ::Json::Node& parentNode, const std::string& id);
			__host__ virtual ~PerspectiveCamera() { OnDestroyAsset(); }

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
			__host__ virtual Device::PerspectiveCamera* GetDeviceInstance() const override final { return cu_deviceData; }
			__host__ virtual AssetHandle<Host::ImageRGBW> GetAccumulationBuffer() override final { return m_hostAccumBuffer; }
			__host__ virtual void						Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
			__host__ virtual void						ClearRenderState() override final;
			__host__ virtual void						SeedRayBuffer() override final;
			__host__ static std::string					GetAssetTypeString() { return "perspective"; }
			__host__ static std::string					GetAssetDescriptionString() { return "Perspective Camera"; }

		private:
			AssetHandle<Host::ImageRGBW>				m_hostAccumBuffer;

			dim3                    m_blockSize;
			dim3					m_gridSize;
		};
	}
}