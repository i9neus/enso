#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"

#include "kernels/CudaImage.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAsset.cuh"
#include "kernels/CudaRenderObject.cuh"

class RenderManager
{
public:
	RenderManager();

	void InitialiseCuda(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight);
	void LinkSynchronisationObjects(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Fence>& d3dFence, HANDLE fenceEvent);
	void LinkD3DOutputTexture(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Resource>& d3dTexture, const UINT textureWidth, const UINT textureHeight, const UINT clientWidth, const UINT clientHeight);
	void UpdateD3DOutputTexture(UINT64& currentFenceValue);
	void Build();
	void Start();
	void Destroy();
	void OnJson(const std::string& dagPath, const std::string& newJson);

	const Json::Document& GetSceneJSON() const { return m_sceneJson; }
	const Cuda::AssetHandle<Cuda::RenderObjectContainer> GetRenderObjectContainer() const { return m_renderObjects; }	
	
	template<typename Lambda>
	void GetRenderStats(Lambda statsLambda) 
	{ 
		std::lock_guard<std::mutex> lock(m_jsonMutex);
		statsLambda(m_renderStatsJson);
	}

private:
	void Run();

	enum ThreadSignal : int { kRun, kRestart, kHalt };

	std::mutex		    m_renderResourceMutex;
	std::mutex			m_jsonMutex;
	std::thread			m_managerThread;
	std::atomic<int>	m_threadSignal;
	Json::Document		m_renderParamsJson;
	Json::Document		m_sceneJson;
	Json::Document		m_configJson;
	std::atomic<bool>	m_isDirty; 
	
	struct
	{
		std::string dagPath;
		Json::Document json;
	} 
	m_paramsPatch;

	Cuda::AssetHandle<Cuda::RenderObjectContainer> m_renderObjects;

	Json::Document			m_renderStatsJson;
	std::array<float, 20>	m_frameTimes;

	using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
	TimePoint					m_renderStartTime;

	// CUDA objects
	cudaExternalMemory_t	     m_externalTextureMemory;
	cudaExternalSemaphore_t      m_externalSemaphore;
	cudaStream_t				 m_D3DStream;
	cudaStream_t				 m_renderStream;
	LUID						 m_dx12deviceluid;
	UINT						 m_cudaDeviceID;
	UINT						 m_nodeMask;
	float						 m_AnimTime;
	void*						 m_cudaTexturePtr = NULL;
	cudaSurfaceObject_t          m_cuSurface;
	cudaEvent_t                  m_renderEvent;
	ComPtr<ID3D12Fence>		     m_d3dFence;
	HANDLE						 m_fenceEvent;

	uint32_t					 m_D3DTextureWidth;
	uint32_t				     m_D3DTextureHeight;
	uint32_t				     m_clientWidth;
	uint32_t                     m_clientHeight;

	Cuda::AssetHandle<Cuda::Host::ImageRGBA>		m_compositeImage;
	Cuda::AssetHandle<Cuda::Host::WavefrontTracer>  m_wavefrontTracer;

};