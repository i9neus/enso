#pragma once

#include "CudaCamera.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;
	class PseudoRNG;
	class QuasiRNG;

	namespace Host { class LightProbeCamera; }

	struct LightProbeCameraParams : public AssetParams
	{
		__host__ __device__ LightProbeCameraParams();
		__host__ LightProbeCameraParams(const ::Json::Node& node);

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);
		__host__ bool operator==(const LightProbeCameraParams&) const;

		BidirectionalTransform transform;
		ivec3 density;
	};

	namespace Device
	{
		class LightProbeCamera : public Device::Camera
		{
		public:
			struct Objects
			{
				Device::RenderState renderState;
				Device::Array<vec4>* cu_accumBuffer = nullptr;
			};

			__device__ LightProbeCamera();
			__device__ virtual void Accumulate(RenderCtx& ctx, const vec3& value) override final;
			__device__ void SeedRayBuffer(const ivec2& viewportPos);
			__device__ virtual const Device::RenderState& GetRenderState() const override final { return m_objects.renderState; }

			__device__ void Synchronise(const LightProbeCameraParams& params)
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
			LightProbeCameraParams 		m_params;
			Objects						m_objects;

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
		class LightProbeCamera : public Host::Camera
		{
		private:
			Device::LightProbeCamera* cu_deviceData;

		public:
			__host__ LightProbeCamera(const ::Json::Node& parentNode, const std::string& id);
			__host__ virtual ~LightProbeCamera() { OnDestroyAsset(); }

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
			__host__ virtual Device::LightProbeCamera* GetDeviceInstance() const override final { return cu_deviceData; }
			__host__ virtual AssetHandle<Host::ImageRGBW> GetAccumulationBuffer() override final { return nullptr; }
			__host__ virtual void						ClearRenderState() override final;
			__host__ virtual void						SeedRayBuffer() override final;
			__host__ static std::string					GetAssetTypeString() { return "lightprobe"; }
			__host__ static std::string					GetAssetDescriptionString() { return "Light Probe Camera"; }

		private:
			AssetHandle<Host::Array<vec4>>				m_hostAccumBuffer;

			dim3                    m_block;
			dim3					m_grid;
		};
	}
}