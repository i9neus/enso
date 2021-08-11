#pragma once

#include "CudaCamera.cuh"
#include "../CudaLightProbeGrid.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;
	class PseudoRNG;
	class QuasiRNG;

	namespace Host { class LightProbeCamera; }

	struct LightProbeCameraParams
	{
		__host__ __device__ LightProbeCameraParams();
		__host__ LightProbeCameraParams(const ::Json::Node& node);

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);

		LightProbeGridParams		grid;
		CameraParams				camera;

		uint						numProbes;
		uint						bucketsPerProbe;
		uint						bucketsPerCoefficient;
		uint						reduceBatchSizePow2;
		uint						totalBuckets;
		uint						coefficientsPerProbe;
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
				Device::Array<vec4>* cu_reduceBuffer = nullptr;
				Device::LightProbeGrid* cu_probeGrid = nullptr;
			};

			__device__ LightProbeCamera();
			__device__ virtual void Accumulate(RenderCtx& ctx, const vec3& value) override final;
			__device__ void SeedRayBuffer();	
			__device__ virtual const Device::RenderState& GetRenderState() const override final { return m_objects.renderState; }
			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ virtual const CameraParams& GetParams() const override final { return m_params.camera; }
			__device__ void ReduceAccumulationBuffer(const uint batchSizeBegin, const uvec2 batchRange);

			__device__ void Synchronise(const LightProbeCameraParams& params);
			__device__ void Synchronise(const Objects& objects);

		private:
			__device__ void Prepare();
			__device__ void CreateRay(const uint& accumIdx, CompressedRay& ray) const;
			__device__ inline void GetProbeAttributesFromIndex(const uint& accumIdx, int& probeIdx, int& coeffIdx, ivec3& gridIdx) const;

		private:
			LightProbeCameraParams 		m_params;
			Objects						m_objects;			
		};
	}

	namespace Host
	{	
		class LightProbeCamera : public Host::Camera
		{
		public:
			__host__ LightProbeCamera(const ::Json::Node& parentNode, const std::string& id);
			__host__ virtual ~LightProbeCamera() { OnDestroyAsset(); }

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
			__host__ virtual void						Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
			__host__ virtual Device::LightProbeCamera* GetDeviceInstance() const override final { return cu_deviceData; }
			__host__ virtual AssetHandle<Host::ImageRGBW> GetAccumulationBuffer() override final { return nullptr; }
			__host__ virtual void						ClearRenderState() override final;
			__host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;

			__host__ virtual void						OnPreRenderPass(const float wallTime, const float frameIdx) override final;
			__host__ virtual void						OnPostRenderPass() override final;

			__host__ static std::string					GetAssetTypeString() { return "lightprobe"; }
			__host__ static std::string					GetAssetDescriptionString() { return "Light Probe Camera"; }
			__host__ virtual const CameraParams&		GetParams() const override final { return m_params.camera; }
			__host__ AssetHandle<Host::LightProbeGrid>  GetLightProbeGrid() { return m_hostLightProbeGrid; }

		private:
			Device::LightProbeCamera*					cu_deviceData;
			Device::LightProbeCamera::Objects			m_deviceObjects;
			LightProbeCameraParams						m_params;

			AssetHandle<Host::Array<vec4>>				m_hostAccumBuffer;
			AssetHandle<Host::Array<vec4>>				m_hostReduceBuffer;
			AssetHandle<Host::LightProbeGrid>			m_hostLightProbeGrid;

			dim3										m_block;
			dim3										m_grid;
			int											m_frameIdx;
			std::string									m_probeGridID;
		};
	}
}