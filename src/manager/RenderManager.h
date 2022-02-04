#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/HighResolutionTimer.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"

#include "kernels/CudaImage.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAsset.cuh"
#include "kernels/CudaRenderObjectContainer.cuh"
#include "kernels/cameras/CudaLightProbeCamera.cuh"

enum RenderManagerRenderState : int { kRenderManagerUndefined, kRenderManagerIdle, kRenderManagerRun, kRenderManagerHalt, kRenderManagerError };
enum RenderManagerBakeType : int { kBakeTypeProbeGrid, kBakeTypeRender };

enum RenderManagerJobState : int 
{ 
	kRenderManagerJobInvalidState =		0,
	kRenderManagerJobIdle =				1, 
	kRenderManagerJobDispatched =		2, 
	kRenderManagerJobRunning =			4, 
	kRenderManagerJobCompleted =		8, 
	kRenderManagerJobAborting =			16,
    kRenderManagerJobActive =			kRenderManagerJobDispatched | kRenderManagerJobRunning	
};

class RenderManager
{
public:
	RenderManager();

	void InitialiseCuda(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight);
	void LinkSynchronisationObjects(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Fence>& d3dFence, HANDLE fenceEvent);
	void LinkD3DOutputTexture(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Resource>& d3dTexture, const UINT textureWidth, const UINT textureHeight, const UINT clientWidth, const UINT clientHeight);
	void UpdateD3DOutputTexture(UINT64& currentFenceValue);
	void StartRenderer();
	void StopRenderer();
	void Destroy();

	void LoadDefaultScene();
	void LoadScene(const std::string filePath);
	void UnloadScene(bool report = false);

	void Dispatch(const Json::Document& commandJson);
	bool PollRenderState(Json::Document& stateJson);

	void ExportLiveViewport(const std::string& pngExportPath);	
	//bool IsStable() const { return m_threadSignal == kRun; }

	const Json::Document& GetSceneJSON() const { return m_sceneJson; }
	const Cuda::AssetHandle<Cuda::RenderObjectContainer> GetRenderObjectContainer() const { return m_renderObjects; }		

private:
	enum DirtyState : int { kDirtinessStateClean, kDirtinessStateSoftReset, kDirtinessStateHardReset };

	std::mutex		    m_jsonInputMutex;
	std::mutex			m_jsonOutputMutex;
	std::thread			m_managerThread;
	std::atomic<int>	m_threadSignal;
	Json::Document		m_patchJson;
	Json::Document		m_sceneJson;
	std::atomic<int>	m_dirtiness; 
	int					m_frameIdx;

	Cuda::AssetHandle<Cuda::RenderObjectContainer> m_renderObjects;

	HighResolutionTimer		m_renderStatsTimer;
	Json::Document			m_renderObjectStatsJson;
	std::vector<float>		m_frameTimes;

	using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
	TimePoint					m_renderStartTime;

	// CUDA objects
	cudaExternalMemory_t	    m_externalTextureMemory;
	cudaExternalSemaphore_t     m_externalSemaphore;
	cudaStream_t				m_D3DStream;
	cudaStream_t				m_renderStream;
	LUID						m_dx12deviceluid;
	UINT						m_cudaDeviceID;
	UINT						m_nodeMask;
	float						m_AnimTime;
	void*						m_cudaTexturePtr = NULL;
	cudaSurfaceObject_t         m_cuSurface;
	cudaEvent_t                 m_renderEvent;
	ComPtr<ID3D12Fence>		    m_d3dFence;
	HANDLE						m_fenceEvent;
	cudaDeviceProp				m_deviceProp;
	
	uint32_t					m_D3DTextureWidth;
	uint32_t				    m_D3DTextureHeight;
	uint32_t				    m_clientWidth;
	uint32_t                    m_clientHeight;

	struct Job
	{
		Job() noexcept : state(kRenderManagerJobIdle) {}

		std::function<bool(Job&)>								onDispatch;
		std::function<bool(Json::Node&, Job&)>					onPoll;
		std::atomic<int>										state;
		Json::Document											json;
	};
	
	std::unordered_map<std::string, Job&>						m_jobMap;
	Job															m_renderStatsJob;	
	Job															m_memoryStatsJob;
	Job															m_exportViewportJob;
	Job															m_exportGridsJob;

	struct 
	{
		Job															job;
		int															type;
		bool														succeeded;
		float														progress;

		Cuda::LightProbeGridExportParams							probeGridExportParams;
		std::string													pngExportPath;
	}
	m_bake;
	
	float														m_meanFrameTime;		

	Cuda::AssetHandle<Cuda::Host::ImageRGBA>					m_compositeImage;
	Cuda::AssetHandle<Cuda::Host::WavefrontTracer>				m_wavefrontTracer;

	std::vector<Cuda::AssetHandle<Cuda::Host::Camera>>			m_activeCameras;
	Cuda::AssetHandle<Cuda::Host::Camera>						m_liveCamera;
	Cuda::AssetHandle<Cuda::Host::LightProbeCamera>				m_lightProbeCamera;

private:
	void Build(const Json::Document& sceneJson);
	void Run();
	void ClearRenderStates();
	void HandleBakeOperations();
	void OnBakePreFrame();
	void OnBakePostFrame();
	void PatchSceneObjects();
	void GatherRenderObjectStatistics();
	void Prepare();

	bool OnExportGridsDispatch(Job&);
	bool OnGatherRenderStatsPoll(Json::Node&, Job&);
	bool OnGatherMemoryStatsPoll(Json::Node&, Job&);
	bool OnBakeDispatch(Job&);
	bool OnBakePoll(Json::Node&, Job&);
	bool OnDefaultDispatch(Job&);
	bool OnDefaultPoll(Json::Node&, Job&) { return true; }

	template<typename DispatchLambda, typename PollLambda>
	void RegisterJob(Job& job, const std::string& name, DispatchLambda onDispatch, PollLambda onPoll)
	{
		m_jobMap.emplace(name, job);
		job.onDispatch = std::bind(onDispatch, this, std::placeholders::_1);
		job.onPoll = std::bind(onPoll, this, std::placeholders::_1, std::placeholders::_2);
	}

};