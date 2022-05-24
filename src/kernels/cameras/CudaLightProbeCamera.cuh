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
		kLightProbeBufferDirectHalf,
		kLightProbeBufferIndirectHalf,
		kLightProbeNumBuffers
	};

	enum ProbeBakeTraversalMode : int
	{
		kBakeTraversalLinear,
		kBakeTraversalHilbert
	};

	enum ProbeBakeSampleMode : int
	{
		kBakeSamplerRandom,
		kBakeSamplerGeodesic
	};

	enum ProbeDebugFlags : uint
	{
		kLightProbeDebugBakePosition		= 1u,
		kLightProbeDebugBakeDirection		= 2u,
		kLightProbeDebugDisableValidity		= 4u
	};

	struct LightProbeCameraParams
	{
		__host__ __device__ LightProbeCameraParams();
		__host__ LightProbeCameraParams(const ::Json::Node& node, const uint flags);

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);

		LightProbeGridParams		grid;
		CameraParams				camera;

		float						minViableValidity;

		uint						subprobesPerProbe;	//			<-- A sub-probe is a set of SH coefficients + data. Multiple sub-probes are accumulated to make a full probe. 
		uint						bucketsPerProbe; //				<-- The total number of accumulation units (coefficients + data) per probe

		uint						totalBuckets; //				<-- The total number of accumulation units in the grid
		uint						totalSubprobes; //				<-- The total number of subprobes in the grid

		ivec2						minMaxSamplesPerSubprobe; // 	<-- The min/max number of samples that should be taken per sub-probe
		vec3						aspectRatio;

		int							lightingMode;
		int							traversalMode;
		int							outputColourSpace;
		int							gridUpdateInterval;
		bool						filterGrids;

		int							samplerType;
		int							fixedSampleSubdivisions;

		uint 						debugFlags;
	};

	struct LightProbeGridExportParams
	{
		std::vector<std::string>	exportPaths;
		bool						isArmed = false;
		float						minGridValidity = 0.0f;
		float						maxGridValidity = 1.0f;
	};

	struct LightProbeCameraAggregateStatistics
	{
		static constexpr float kInvalidMetric = -1.0f;
		
		LightProbeCameraAggregateStatistics() : minMaxSamples(kInvalidMetric) {}

		float			meanValidity = kInvalidMetric;
		float			meanDistance = kInvalidMetric;
		vec2			minMaxSamples;
		float			meanSamples = kInvalidMetric;

		struct
		{
			float			probesFull = kInvalidMetric;
			float			probesConverged = kInvalidMetric;
			float			progress = kInvalidMetric;
		} 
		bake;

		struct
		{
			float			meanI = 0.0f;
			float			MSE = 0.0f;
		} 
		error;		
	};

	namespace Device
	{
		class LightProbeCamera : public Device::Camera
		{	
		public:
			struct Objects
			{
				__host__ __device__ Objects() :
					cu_reduceBuffer(nullptr),
					cu_convergenceGrid(nullptr)
				{
					for (int i = 0; i < kLightProbeNumBuffers; ++i)
					{
						cu_accumBuffers[i] = nullptr;
						cu_probeGrids[i] = nullptr;
						cu_filteredProbeGrids[i] = nullptr;
					}

					cu_indirectionBuffer = nullptr;
					cu_lightProbeErrorGrids[0] = nullptr;
					cu_lightProbeErrorGrids[1] = nullptr;
					cu_meanI = nullptr;
					cu_sampleBuffer = nullptr;
				}

				Device::RenderState			renderState;
				Device::Array<vec4>*		cu_accumBuffers[kLightProbeNumBuffers];
				Device::Array<vec4>*		cu_reduceBuffer;
				Device::Array<uint>*		cu_indirectionBuffer;
				Device::LightProbeGrid*		cu_probeGrids[kLightProbeNumBuffers];
				Device::LightProbeGrid*		cu_filteredProbeGrids[kLightProbeNumBuffers];
				Device::Array<vec2>*		cu_lightProbeErrorGrids[2];
				Device::Array<uchar>*		cu_convergenceGrid;
				Device::Array<vec3>*		cu_sampleBuffer;
				float*						cu_meanI;
			};

			__device__ LightProbeCamera();
			__device__ virtual void Accumulate(const RenderCtx& ctx, const Ray& incidentRay, const HitCtx& hitCtx, vec3 L, const vec3& albedo, const bool isAlive) override final;
			__device__ void SeedRayBuffer(const int frameIdx);
			__device__ virtual const Device::RenderState& GetRenderState() const override final { return m_objects.renderState; }
			__device__ void Composite(const ivec2& accumPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ virtual const CameraParams& GetParams() const override final { return m_params.camera; }
			__device__ void ReduceAccumulationBuffer(Device::Array<vec4>* accumBuffer, Device::LightProbeGrid* cu_probeGrid, const uint batchSizeBegin, const uvec2 batchRange);
			__device__ void BuildLightProbeErrorGrid();
			__device__ void DilateLightProbeErrorGrid();
			__device__ void ReduceLightProbeErrorData(LightProbeCameraAggregateStatistics& stats);

			__device__ void Synchronise(const LightProbeCameraParams& params);
			__device__ void Synchronise(const Objects& objects);

		private:
			__device__ void Prepare();
			__device__ void CreateRays(const int& probeIdx, const int& subsampleIdx, CompressedRay* rays, const int frameIdx) const;
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
			__host__ LightProbeCamera(const std::string& id, const ::Json::Node& parentNode);
			__host__ virtual ~LightProbeCamera() { OnDestroyAsset(); }

			__host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
			
			__host__ virtual void						Bind(RenderObjectContainer& sceneObjects) override final;
			__host__ virtual void						OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;
			__host__ virtual void                       OnDestroyAsset() override final;
			__host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
			__host__ virtual void						Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
			__host__ virtual Device::LightProbeCamera* GetDeviceInstance() const override final { return cu_deviceData; }
			__host__ virtual AssetHandle<Host::ImageRGBW> GetAccumulationBuffer() override final { return nullptr; }
			__host__ virtual void						ClearRenderState() override final;
			__host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;
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

			__host__ const LightProbeCameraAggregateStatistics& PollBakeProgress();
			__host__ bool								ExportProbeGrid(const LightProbeGridExportParams& params);

		private:
			__host__ void								Compile();
			__host__ void								UpdateProbeGridAggregateStatistics();
			__host__ void								BuildLightProbeGrids();
			__host__ void								BuildLightProbeErrorGrid();
			__host__ void								Prepare(LightProbeCameraParams newParams);
			__host__ void								PrepareHilbertBuffer(const LightProbeCameraParams& newParams);
			__host__ void								PrepareSampleBufffer(LightProbeCameraParams& newParams);

			Device::LightProbeCamera*					cu_deviceData;
			Device::LightProbeCamera::Objects			m_deviceObjects;
			LightProbeCameraParams						m_params;

			std::array<AssetHandle<Host::Array<vec4>>, kLightProbeNumBuffers>		m_hostAccumBuffers;
			std::array<AssetHandle<Host::LightProbeGrid>, kLightProbeNumBuffers>	m_hostLightProbeGrids;
			std::array<AssetHandle<Host::LightProbeGrid>, kLightProbeNumBuffers>	m_hostFilteredLightProbeGrids;
			AssetHandle<Host::Array<vec4>>											m_hostReduceBuffer;
			AssetHandle<Host::Array<uint>>											m_hostIndirectionBuffer;
			AssetHandle<Host::Array<vec3>>											m_hostSampleBuffer;
			DeviceObjectRAII<LightProbeCameraAggregateStatistics>					m_aggregateStats;
			std::array<std::string, 4>												m_gridIDs;
			std::array<std::string, 4>												m_filteredGridIDs;

			std::array<AssetHandle<Host::Array<vec2>>, 2>							m_hostLightProbeErrorGrids;
			AssetHandle<Host::Array<uchar>>											m_hostConvergenceGrid;
			DeviceObjectRAII<float>													m_hostMeanI;

			dim3										m_block;
			dim3										m_seedGrid, m_reduceGrid;
			int											m_frameIdx;
			std::string									m_probeGridID;
			bool										m_needsRebind;
		};
	}
}