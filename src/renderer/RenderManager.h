#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "kernels/CudaImage.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAsset.cuh"

class RenderManager
{
public:
	RenderManager();

	void InitialiseCuda(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight);
	void LinkSynchronisationObjects(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Fence>& d3dFence);
	void LinkD3DOutputTexture(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Resource>& d3dTexture, const UINT textureWidth, const UINT textureHeight, const UINT clientWidth, const UINT clientHeight);
	void UpdateD3DOutputTexture(UINT64& currentFenceValue);
	void Start();
	void Destroy();

private:
	void Run();

	enum ThreadSignal : int { kRun, kRestart, kHalt };

	std::thread m_managerThread;
	std::atomic<int> m_threadSignal;

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

	uint32_t					 m_D3DTextureWidth;
	uint32_t				     m_D3DTextureHeight;
	uint32_t				     m_clientWidth;
	uint32_t                     m_clientHeight;

	Cuda::AssetHandle<Cuda::Host::ImageRGBA>		m_compositeImage;
	Cuda::AssetHandle<Cuda::Host::WavefrontTracer>  m_wavefrontTracer;
};