#pragma once

#include "CudaObjectManager.h"
#include "RenderManagerInterface.h"

#include "kernels/CudaImage.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAsset.cuh"
#include "kernels/CudaRenderObjectContainer.cuh"
#include "kernels/cameras/CudaLightProbeCamera.cuh"

enum RenderManagerRenderState : int 
{ 
	kRenderManagerUndefined, 
	kRenderManagerIdle, 
	kRenderManagerRun, 
	kRenderManagerHalt, 
	kRenderManagerError 
};

enum RenderManagerBakeType : int 
{ 
	kBakeTypeNoisyProbeGrid =		1, 
	kBakeTypeReferenceProbeGrid =	2, 
	kBakeTypeRender =				4 
};

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

enum RenderManagerCommand : int
{
	kRenderManagerClearStates,
	kRenderMangerUpdateParams
};

class RenderManager : public RenderManagerInterface, public CudaObjectManager
{
public:
	RenderManager();

	virtual void Initialise() override final;
	virtual void Destroy() override final;

	virtual void OnResizeClient() override final;

	void StartRenderer();
	void StopRenderer();

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

	Cuda::AssetHandle<Cuda::RenderObjectContainer> m_renderObjects;

	HighResolutionTimer		m_renderStatsTimer;
	Json::Document			m_renderObjectStatsJson;
	std::vector<float>		m_frameTimes;

	using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
	TimePoint					m_renderStartTime;

	std::mutex													m_commandMutex;
	std::unordered_map<int, std::function<void()>>				m_commandMap;
	std::unordered_set<int>										m_commandSet;

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

	Cuda::AssetHandle<Cuda::Host::WavefrontTracer>				m_wavefrontTracer;
	std::vector<Cuda::AssetHandle<Cuda::Host::Camera>>			m_activeCameras;
	Cuda::AssetHandle<Cuda::Host::Camera>						m_liveCamera;
	Cuda::AssetHandle<Cuda::Host::LightProbeCamera>				m_lightProbeCamera;

private:
	void Build(const Json::Document& sceneJson);
	void Run();
	void ClearAllRenderStates();
	void HandleBakeOperations();
	void OnBakePreFrame();
	void OnBakePostFrame();
	uint PatchSceneObjects();
	void GatherRenderObjectStatistics();
	void Prepare();
	void EmplaceRenderCommand(const int cmd);

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