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

enum class BakeStatus : int { kReady, kRunning, kHalt };

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

	void OnJson(const Json::Document& patchJson);

	void ExportLiveViewport(const std::string& pngExportPath);
	void StartBake(const std::vector<std::string>& usdExportPaths, const bool exportToUSD);
	void AbortBake();
	BakeStatus GetBakeStatus() const { return m_bakeStatus; }
	float GetBakeProgress() const { return (m_bakeStatus == BakeStatus::kRunning) ? m_bakeProgress : 0.0f; }

	const Json::Document& GetSceneJSON() const { return m_sceneJson; }
	const Cuda::AssetHandle<Cuda::RenderObjectContainer> GetRenderObjectContainer() const { return m_renderObjects; }	
	
	void GetRenderStats(Json::Document& outputJson) 
	{ 
		std::lock_guard<std::mutex> lock(m_jsonMutex);
		outputJson.DeepCopy(m_renderStatsJson);
	}

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

	enum ThreadSignal : int { kIdle, kRun, kHalt };
	enum DirtyState : int { kClean, kSoftReset, kHardReset };

	std::mutex		    m_renderResourceMutex;
	std::mutex			m_jsonMutex;
	std::thread			m_managerThread;
	std::atomic<int>	m_threadSignal;
	Json::Document		m_paramsPatchJson;
	Json::Document		m_sceneJson;
	std::atomic<int>	m_dirtiness; 
	int					m_frameIdx;
	
	struct
	{
		std::string dagPath;
		Json::Document json;
	} 
	m_paramsPatch;

	Cuda::AssetHandle<Cuda::RenderObjectContainer> m_renderObjects;

	HighResolutionTimer		m_renderStatsTimer;
	Json::Document			m_renderStatsJson;
	std::array<float, 20>	m_frameTimes;

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

	uint32_t					m_D3DTextureWidth;
	uint32_t				    m_D3DTextureHeight;
	uint32_t				    m_clientWidth;
	uint32_t                    m_clientHeight;

	std::atomic<BakeStatus>		m_bakeStatus;
	float						m_bakeProgress;
	float						m_meanFrameTime;
	std::vector<std::string>	m_usdExportPaths;
	bool						m_exportToUSD;
	std::string                 m_pngExportPath;
	std::atomic<bool>			m_exportToPNG;

	Cuda::AssetHandle<Cuda::Host::ImageRGBA>					m_compositeImage;
	Cuda::AssetHandle<Cuda::Host::WavefrontTracer>				m_wavefrontTracer;

	std::vector<Cuda::AssetHandle<Cuda::Host::Camera>>			m_activeCameras;
	Cuda::AssetHandle<Cuda::Host::Camera>						m_liveCamera;
	Cuda::AssetHandle<Cuda::Host::LightProbeCamera>				m_lightProbeCamera;

};