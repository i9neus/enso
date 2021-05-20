#include "RenderManager.h"

#include "dx12/SecurityAttributes.h"
#include "dx12/DXSampleHelper.h"
#include "generic/thirdparty/nvidia/helper_cuda.h"

#include "kernels/CudaCompositor.h"

RenderManager::RenderManager() : 
	m_threadSignal(kHalt)
{
}

void RenderManager::InitialiseCuda(const LUID& dx12DeviceLUID)
{
	int num_cuda_devices = 0;
	checkCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

	if (!num_cuda_devices)
	{
		throw std::exception("No CUDA Devices found");
	}
	for (UINT devId = 0; devId < (UINT)num_cuda_devices; devId++)
	{
		cudaDeviceProp devProp;
		checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));

		if ((memcmp(&dx12DeviceLUID.LowPart, devProp.luid, sizeof(dx12DeviceLUID.LowPart)) == 0) &&
			(memcmp(&dx12DeviceLUID.HighPart, devProp.luid + sizeof(dx12DeviceLUID.LowPart), sizeof(dx12DeviceLUID.HighPart)) == 0))
		{
			checkCudaErrors(cudaSetDevice(devId));
			m_cudaDeviceID = devId;
			m_nodeMask = devProp.luidDeviceNodeMask;
			checkCudaErrors(cudaStreamCreate(&m_streamToRun));
			std::printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
			break;
		}
	}
}

void RenderManager::LinkD3DOutputTexture(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Resource>& d3dTexture, const UINT textureWidth, const UINT textureHeight)
{
	m_outputWidth = textureWidth;
	m_outputHeight = textureHeight;
	
	HANDLE sharedHandle;
	WindowsSecurityAttributes windowsSecurityAttributes;
	LPCWSTR name = NULL;
	ThrowIfFailed(d3dDevice->CreateSharedHandle(d3dTexture.Get(), &windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle));

	D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
	d3d12ResourceAllocationInfo = d3dDevice->GetResourceAllocationInfo(m_nodeMask, 1, &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT, textureWidth, textureHeight));

	std::printf("d3d12ResourceAllocationInfo.SizeInBytes: %i\n", d3d12ResourceAllocationInfo.SizeInBytes);

	cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
	memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

	externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
	externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
	externalMemoryHandleDesc.size = d3d12ResourceAllocationInfo.SizeInBytes;
	externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

	checkCudaErrors(cudaImportExternalMemory(&m_externalTextureMemory, &externalMemoryHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
	cuExtmemMipDesc.extent = make_cudaExtent(textureWidth, textureHeight, 0);
	cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
	cuExtmemMipDesc.numLevels = 1;
	cuExtmemMipDesc.flags = cudaArraySurfaceLoadStore;

	cudaMipmappedArray_t cuMipArray{};
	checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cuMipArray, m_externalTextureMemory, &cuExtmemMipDesc));

	cudaArray_t cuArray{};
	checkCudaErrors(cudaGetMipmappedArrayLevel(&cuArray, cuMipArray, 0));

	cudaResourceDesc cuResDesc{};
	cuResDesc.resType = cudaResourceTypeArray;
	cuResDesc.res.array.array = cuArray;
	checkCudaErrors(cudaCreateSurfaceObject(&m_cuSurface, &cuResDesc));
}

void RenderManager::UpdateD3DOutputTexture(UINT64& currentFenceValue)
{
	cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
	memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

	externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
	externalSemaphoreWaitParams.flags = 0;

	checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun));

	Cuda::CompositeBuffers(m_outputWidth, m_outputHeight, m_cuSurface, m_AnimTime, m_streamToRun);

	cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
	memset(&externalSemaphoreSignalParams, 0, sizeof(externalSemaphoreSignalParams));
	currentFenceValue++;
	externalSemaphoreSignalParams.params.fence.value = currentFenceValue;
	externalSemaphoreSignalParams.flags = 0;

	checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));
}

void RenderManager::LinkSynchronisationObjects(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Fence>& d3dFence)
{
	cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;

	memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
	WindowsSecurityAttributes windowsSecurityAttributes;
	LPCWSTR name = NULL;
	HANDLE sharedHandle = NULL;
	externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
	d3dDevice->CreateSharedHandle(d3dFence.Get(), &windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle);
	externalSemaphoreHandleDesc.handle.win32.handle = (void*)(sharedHandle);
	externalSemaphoreHandleDesc.flags = 0;

	checkCudaErrors(cudaImportExternalSemaphore(&m_externalSemaphore, &externalSemaphoreHandleDesc));
}

void RenderManager::Start()
{
	m_threadSignal = kRun;
	m_managerThread = std::thread(std::bind(&RenderManager::Run, this));

	Assert(m_managerThread.joinable());
}

void RenderManager::Run()
{	
	int ticks = 0;
	while (m_threadSignal.load() == kRun)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));

		std::printf("Tick: %i\n", ++ticks);
	}
}

void RenderManager::Destroy()
{
	if (!m_managerThread.joinable()) { return; }
	
	m_threadSignal.store(kHalt);
	std::printf("Shutting down...\n");

	m_managerThread.join();

	checkCudaErrors(cudaDestroyExternalSemaphore(m_externalSemaphore));
	checkCudaErrors(cudaDestroyExternalMemory(m_externalTextureMemory));
}