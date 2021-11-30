﻿#pragma once

#include "CudaCamera.cuh"
#include "../lightprobes/CudaLightProbeGrid.cuh"
#include "../CudaDeviceObjectRAII.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class CompressedRay;
	class RenderCtx;
	class PseudoRNG;
	class QuasiRNG;

	namespace Host { class LightProbeCamera; }
	 
	enum ProbeBakeLightingMode : int
	{
		kBakeLightingCombined,
		kBakeLightingSeparated
	};

	enum LightProbeBufferIndices : int
	{
		kLightProbeBufferDirect = 0,
		kLightProbeBufferIndirect,
		kLightProbeBufferHalf,
		kLightProbeNumBuffers
	};

	struct LightProbeCameraParams
	{
		__host__ __device__ LightProbeCameraParams();
		__host__ LightProbeCameraParams(const ::Json::Node& node);

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);

		LightProbeGridParams		grid;
		CameraParams				camera;

		float						minViableValidity;

		uint						subsamplesPerProbe;	//			<-- A sub-sample is a set of SH coefficients plus data.
		uint						bucketsPerProbe; //				<-- The total number of accumulation units per probe

		uint						totalBuckets; //				<-- The total number of accumulation units in the grid
		uint						totalSubsamples; //				<-- The total number of subsamples in the grid

		int							maxSamplesPerBucket; // 		<-- The maximum number of samples that should be taken per bucket
		vec3						aspectRatio;

		int							lightingMode;
		int							gridUpdateInterval;
	};

	struct LightProbeGridExportParams
	{
		std::vector<std::string>	usdExportPaths;
		bool						exportToUSD = false;
		float						minGridValidity = 0.0f;
		float						maxGridValidity = 1.0f;
	};

	namespace Device
	{
		class LightProbeCamera : public Device::Camera
		{
		public:
			struct AggregateStatistics
			{
				__host__ __device__ AggregateStatistics() : minMaxSamples(-1.0f), meanValidity(-1.0f), meanDistance(-1.0f), probeCount(0) {}

				vec2	minMaxSamples;
				vec2	minMaxCoeffs[4];
				float	meanValidity;
				float	meanDistance;
				int		probeCount;
			};

			struct Objects
			{
				__host__ __device__ Objects()
				{
					for (int i = 0; i < kLightProbeNumBuffers; ++i)
					{
						cu_accumBuffers[i] = nullptr;
						cu_probeGrids[i] = nullptr;
					}
				}

				Device::RenderState renderState;
				Device::Array<vec4>* cu_accumBuffers[kLightProbeNumBuffers];
				Device::Array<vec4>* cu_reduceBuffer = nullptr;
				Device::LightProbeGrid* cu_probeGrids[kLightProbeNumBuffers];
			};

			__device__ LightProbeCamera();
			__device__ virtual void Accumulate(const RenderCtx& ctx, const Ray& incidentRay, const HitCtx& hitCtx, const vec3& value, const bool isAlive) override final;
			__device__ void SeedRayBuffer(const int frameIdx);
			__device__ virtual const Device::RenderState& GetRenderState() const override final { return m_objects.renderState; }
			__device__ void Composite(const ivec2& accumPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ virtual const CameraParams& GetParams() const override final { return m_params.camera; }
			__device__ void ReduceAccumulationBuffer(Device::Array<vec4>* accumBuffer, Device::LightProbeGrid* cu_probeGrid, const uint batchSizeBegin, const uvec2 batchRange);

			__device__ void Synchronise(const LightProbeCameraParams& params);
			__device__ void Synchronise(const Objects& objects);

		private:
			__device__ void Prepare();
			__device__ void CreateRays(const uint& accumIdx, CompressedRay* rays, const int frameIdx) const;
			__device__ __forceinline__ void ReduceAccumulatedSample(vec4& dest, const vec4& source);

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
			struct AggregateStatistics
			{
				float			meanGridValidity = -1.0f;
				float			bakeProgress = -1.0f;
				int				numActiveGrids = 0;
				vec2			minMaxSamples;
				bool			isConverged = false;
			};

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
			__host__ void								Prepare();
			__host__ virtual bool						IsBakingCamera() const override final { return true; }
			__host__ virtual bool						EmitStatistics(Json::Node& node) const override final;

			__host__ virtual void						OnPreRenderPass(const float wallTime, const uint frameIdx) override final;
			__host__ virtual void						OnPostRenderPass() override final;

			__host__ static std::string					GetAssetTypeString() { return "lightprobe"; }
			__host__ static std::string					GetAssetDescriptionString() { return "Light Probe Camera"; }
			__host__ virtual const CameraParams& GetParams() const override final { return m_params.camera; }
			__host__ void								SetLightProbeCameraParams(const LightProbeCameraParams& params);
			__host__ const LightProbeCameraParams& GetLightProbeCameraParams() const { return m_params; }
			__host__ AssetHandle<Host::LightProbeGrid>  GetLightProbeGrid(const int idx) { return m_hostLightProbeGrids[idx]; }

			__host__ const AggregateStatistics&			PollBakeProgress();
			__host__ bool								ExportProbeGrid(const LightProbeGridExportParams& params);

		private:
			__host__ void								UpdateProbeGridAggregateStatistics();
			__host__ void								BuildLightProbeGrids();

			Device::LightProbeCamera*					cu_deviceData;
			Device::LightProbeCamera::Objects			m_deviceObjects;
			LightProbeCameraParams						m_params;
			AggregateStatistics							m_aggregateStats;

			std::array<AssetHandle<Host::Array<vec4>>, kLightProbeNumBuffers>		m_hostAccumBuffers;
			std::array<AssetHandle<Host::LightProbeGrid>, kLightProbeNumBuffers>	m_hostLightProbeGrids;
			AssetHandle<Host::Array<vec4>>											m_hostReduceBuffer;
			std::array<std::string, 3>												m_gridIDs;

			dim3										m_block;
			dim3										m_seedGrid, m_reduceGrid;
			int											m_frameIdx;
			std::string									m_probeGridID;
		};
	}
}