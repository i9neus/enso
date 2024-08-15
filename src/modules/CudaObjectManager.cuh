#pragma once

#include "win/D3DHeaders.h"
#include "core/utils/HighResolutionTimer.h"
#include <cuda_runtime.h>
#include "io/json/JsonUtils.h"
#include <deque>

#include "core/containers/Image.cuh"
#include "core/assets/Asset.cuh"

namespace Enso
{	
	class CudaObjectManager
	{
	public:
		CudaObjectManager();
		~CudaObjectManager() noexcept;

		void						InitialiseCuda(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight);
		void						DestroyCuda();

		void						LinkSynchronisationObjects(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Fence>& d3dFence, HANDLE fenceEvent);
		void						LinkD3DOutputTexture(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Resource>& d3dTexture, const UINT textureWidth, const UINT textureHeight,
			const UINT clientWidth, const UINT clientHeight);
		void						UpdateD3DOutputTexture(UINT64& currentFenceValue, AssetHandle<Host::ImageRGBA>& compositeImage, const bool doRedraw);

		void						OnWindowResize(const UINT clientWidth, const UINT clientHeight);

	protected:
		const cudaDeviceProp& GetCudaDeviceProperties() const { return m_deviceProp; }

		cudaStream_t				m_D3DStream;
		cudaStream_t				m_renderStream;

		uint32_t				    m_clientWidth;
		uint32_t                    m_clientHeight;

	private:

		// CUDA objects
		cudaExternalMemory_t	    m_externalTextureMemory;
		cudaExternalSemaphore_t     m_externalSemaphore;
		LUID						m_dx12deviceluid;
		UINT						m_cudaDeviceID;
		UINT						m_nodeMask;
		float						m_AnimTime;
		void* m_cudaTexturePtr = NULL;
		cudaSurfaceObject_t         m_cuSurface;
		cudaEvent_t                 m_renderEvent;
		ComPtr<ID3D12Fence>		    m_d3dFence;
		HANDLE						m_fenceEvent;
		cudaDeviceProp				m_deviceProp;

		uint32_t					m_D3DTextureWidth;
		uint32_t				    m_D3DTextureHeight;
	};
}