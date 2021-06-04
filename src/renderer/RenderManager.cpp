#include "RenderManager.h"

#include "dx12/SecurityAttributes.h"
#include "dx12/DXSampleHelper.h"
#include "thirdparty/nvidia/helper_cuda.h"

RenderManager::RenderManager() : 
	m_threadSignal(kHalt)
{
}

void RenderManager::InitialiseCuda(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight)
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
			IsOk(cudaSetDevice(devId));
			m_cudaDeviceID = devId;
			m_nodeMask = devProp.luidDeviceNodeMask;
			checkCudaErrors(cudaStreamCreateWithFlags(&m_D3DStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreate(&m_renderStream));
			std::printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
			break;
		}
	}

	constexpr size_t kCudaHeapSizeLimit = 128 * 1024 * 1024;

	IsOk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, kCudaHeapSizeLimit));

	checkCudaErrors(cudaEventCreate(&m_renderEvent));

	// Create some Cuda objects
	m_compositeImage = Cuda::AssetHandle<Cuda::Host::ImageRGBA>("id_compositeImage", 512, 512, m_renderStream);	
	m_wavefrontTracer = Cuda::AssetHandle<Cuda::Host::WavefrontTracer>("id_wavefrontTracer", m_renderStream);

	Cuda::VerifyTypeSizes();

	IsOk(cudaDeviceSynchronize());
}

void RenderManager::Destroy()
{
	if (!m_managerThread.joinable()) { return; }

	std::printf("Shutting down and destroying %i managed assets:", Cuda::AR().Size());
	Cuda::AR().Report();

	m_threadSignal.store(kHalt);
	std::printf("Shutting down...\n");

	m_managerThread.join();
	std::printf("Killed threads.\n");

	// Destroy assets
	m_wavefrontTracer.DestroyAsset();
	m_compositeImage.DestroyAsset();

	// Destroy events
	checkCudaErrors(cudaEventDestroy(m_renderEvent));

	// Destroy D3D linked objects
	checkCudaErrors(cudaDestroyExternalSemaphore(m_externalSemaphore));
	checkCudaErrors(cudaDestroyExternalMemory(m_externalTextureMemory));
}

void RenderManager::LinkD3DOutputTexture(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Resource>& d3dTexture, const UINT textureWidth, const UINT textureHeight, const UINT clientWidth, const UINT clientHeight)
{
	m_D3DTextureWidth = textureWidth;
	m_D3DTextureHeight = textureHeight;
	m_clientWidth = math::min(clientWidth, textureWidth);
	m_clientHeight = math::min(clientHeight, textureHeight);

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

	checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_D3DStream));

	m_compositeImage->CopyImageToD3DTexture(m_clientWidth, m_clientHeight, m_cuSurface, m_D3DStream);
	checkCudaErrors(cudaStreamSynchronize(m_D3DStream));

	cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
	std::memset(&externalSemaphoreSignalParams, 0, sizeof(externalSemaphoreSignalParams));

	externalSemaphoreSignalParams.params.fence.value = ++currentFenceValue;
	externalSemaphoreSignalParams.flags = 0;

	checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_D3DStream));
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
	std::printf("Start!\n");
	
	m_threadSignal = kRun;
	m_managerThread = std::thread(std::bind(&RenderManager::Run, this));

	m_renderStartTime = std::chrono::high_resolution_clock::now();

	Assert(m_managerThread.joinable());
}

void RenderManager::Run()
{		
	checkCudaErrors(cudaStreamSynchronize(m_renderStream));
	
	int iterationIdx = 0, frameIdx = 0;
	constexpr float kTargetFps = 60.0f;
	constexpr int kMaxSubframes = 1;
	int numSubframes = kMaxSubframes;
	while (m_threadSignal.load() == kRun)
	{				
		Timer timer([&](float elapsed) -> std::string 
			{ 
				const float fps = 1.0f / elapsed;
				return tfm::format("Iteration %i: Frame %i, Subframes: %i, FPS: %f ", iterationIdx, frameIdx, numSubframes, fps);
			});
		
		for (int subFrameIdx = 0; subFrameIdx < numSubframes; subFrameIdx++)
		{
			std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - m_renderStartTime;

			m_wavefrontTracer->Iterate(timeDiff.count(), frameIdx);

			frameIdx++;
		}		 

		m_wavefrontTracer->Composite(m_compositeImage);

		checkCudaErrors(cudaStreamSynchronize(m_renderStream));

		//numSubframes = int(numSubframes * (1 / kTargetFps) / timer.Get());
		//numSubframes = math::clamp(numSubframes, 1, kMaxSubframes);			

		iterationIdx++;
		
		//std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}