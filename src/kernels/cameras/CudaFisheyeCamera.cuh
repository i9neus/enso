#pragma once

#include "CudaCamera.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;
	class PseudoRNG;
	class QuasiRNG;

	namespace Host { class FisheyeCamera; }

	struct FisheyeCameraParams
	{
		__host__ __device__ FisheyeCameraParams();
		__host__ FisheyeCameraParams(const ::Json::Node& node);

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);

		CameraParams				camera;
		BidirectionalTransform		transform;
	};

	namespace Device
	{
		class FisheyeCamera : public Device::Camera
		{
		public:
			struct Objects
			{
				Device::RenderState renderState;
				Device::ImageRGBW* cu_accumBuffer = nullptr;
			};

			__device__ FisheyeCamera();
			__device__ virtual void Accumulate(const RenderCtx& ctx, const Ray& incidentRay, const HitCtx& hitCtx, const vec3& value) override final;
			__device__ virtual void SeedRayBuffer(const ivec2& viewportPos);
			__device__ virtual const Device::RenderState& GetRenderState() const override final { return m_objects.renderState; }
			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ virtual const CameraParams& GetParams() const override final { return m_params.camera; }

			__device__ void Synchronise(const FisheyeCameraParams& params)
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
			__device__ bool RectilinearViewportToCartesian(const ivec2& viewportPos, vec3& cart) const;
			__device__ bool SphericalViewportToCartesian(const ivec2& viewportPos, vec3& cart) const;
			__device__ bool LambertViewportToCartesian(const ivec2& viewportPos, vec3& cart) const;

		private:
			FisheyeCameraParams 		m_params;
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
		class FisheyeCamera : public Host::Camera
		{
		private:
			Device::FisheyeCamera* cu_deviceData;
			FisheyeCameraParams					m_params;

		public:
			__host__ FisheyeCamera(const ::Json::Node& parentNode, const std::string& id);
			__host__ virtual ~FisheyeCamera() { OnDestroyAsset(); }

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
			__host__ virtual Device::FisheyeCamera* GetDeviceInstance() const override final { return cu_deviceData; }
			__host__ virtual AssetHandle<Host::ImageRGBW> GetAccumulationBuffer() override final { return m_hostAccumBuffer; }
			__host__ virtual void						Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
			__host__ virtual void						ClearRenderState() override final;
			__host__ virtual void						OnPreRenderPass(const float wallTime, const uint frameIdx) override final;
			__host__ static std::string					GetAssetTypeString() { return "fisheye"; }
			__host__ static std::string					GetAssetDescriptionString() { return "Fisheye Camera"; }
			__host__ virtual const CameraParams& GetParams() const override final { return m_params.camera; }
			__host__ virtual bool						IsBakingCamera() const override final { return false; }

		private:
			AssetHandle<Host::ImageRGBW>				m_hostAccumBuffer;

			dim3                    m_blockSize;
			dim3					m_gridSize;
		};
	}
}