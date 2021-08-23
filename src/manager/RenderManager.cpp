#include "RenderManager.h"

#include "dx12/SecurityAttributes.h"
#include "dx12/DXSampleHelper.h"
#include "thirdparty/nvidia/helper_cuda.h"

#include "kernels/CudaTests.cuh"
#include "kernels/CudaCommonIncludes.cuh"
#include "kernels/CudaRenderObjectFactory.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAssetContainer.cuh"
#include "kernels/tracables/CudaTracable.cuh"
#include "kernels/cameras/CudaCamera.cuh"

#include "io/USDIO.h"

RenderManager::RenderManager() : 
	m_threadSignal(kHalt),
	m_dirtiness(kClean),
	m_frameIdx(0),
	m_iterationIdx(0),
	m_bakeStatus(BakeStatus::kReady),
	m_bakeProgress(0.0f)
{
}

void RenderManager::InitialiseCuda(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight)
{
	Log::NL();
	Log::Indent indent("Initialising Cuda...\n");

	int num_cuda_devices = 0;
	checkCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

	if (!num_cuda_devices)
	{
		throw std::exception("No CUDA Devices found");
	}
	for (UINT devId = 0; devId < (UINT)num_cuda_devices; devId++)
	{
		Log::Indent indent1;

		cudaDeviceProp devProp;
		checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));

		if ((memcmp(&dx12DeviceLUID.LowPart, devProp.luid, sizeof(dx12DeviceLUID.LowPart)) == 0) &&
			(memcmp(&dx12DeviceLUID.HighPart, devProp.luid + sizeof(dx12DeviceLUID.LowPart), sizeof(dx12DeviceLUID.HighPart)) == 0))
		{
			cudaDeviceProp deviceProp;
			IsOk(cudaGetDeviceProperties(&deviceProp, devId));

			int pLow, pHigh;
			IsOk(cudaDeviceGetStreamPriorityRange(&pLow, &pHigh));

			Log::Debug("Stream priority range: [%i, %i]\n", pLow, pHigh);

			IsOk(cudaSetDevice(devId));
			m_cudaDeviceID = devId;
			m_nodeMask = devProp.luidDeviceNodeMask;
			checkCudaErrors(cudaStreamCreateWithPriority(&m_D3DStream, cudaStreamNonBlocking, pHigh));
			checkCudaErrors(cudaStreamCreate(&m_renderStream));
			Log::Write("CUDA Device Used [%d] %s\n", devId, devProp.name);
			{
				Log::Indent indent2;
				Log::Debug("- sharedMemPerMultiprocessor: %i bytes\n", deviceProp.sharedMemPerMultiprocessor);
				Log::Debug("- sharedMemPerBlock: %i bytes\n", deviceProp.sharedMemPerBlock);
			}
			break;
		}
	}

	constexpr size_t kCudaHeapSizeLimit = 128 * 1024 * 1024;

	IsOk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, kCudaHeapSizeLimit));

	checkCudaErrors(cudaEventCreate(&m_renderEvent));

	//cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize);

	// Create some Cuda objects
	m_compositeImage = Cuda::AssetHandle<Cuda::Host::ImageRGBA>("id_compositeImage", 512, 512, m_renderStream);
	//m_wavefrontTracer = Cuda::AssetHandle<Cuda::Host::WavefrontTracer>("id_wavefrontTracer", m_renderStream);

	Cuda::VerifyTypeSizes();

	Cuda::TestScheduling();

	IsOk(cudaDeviceSynchronize());
}

void RenderManager::Build()
{
	AssertMsg(!m_renderObjects, "Render objects have already been instantiated.");

	Log::NL();
	const Log::Snapshot beginState = Log::GetMessageState();
	{
		Log::Indent indent("Building render manager...\n");

		// Load the root config
		Log::Write("Loading 'config.json'...\n");
		m_configJson.Deserialise("config.json");

		std::string sceneJsonPath;
		m_configJson.GetValue("scene", sceneJsonPath, Json::kRequiredAssert);

		Log::Write("Loading scene file '%s'...\n", sceneJsonPath);
		m_sceneJson.Deserialise(sceneJsonPath);

		// Create a container for the render objects
		m_renderObjects = Cuda::AssetHandle<Cuda::RenderObjectContainer>("__root_renderObjectContainer");

		// Instantiate them according to the scene JSON		
		{
			Log::Indent indent("Creating scene objects...\n");

			Cuda::RenderObjectFactory objectFactory(m_renderStream);
			objectFactory.InstantiateSceneObjects(m_sceneJson, m_renderObjects);
			objectFactory.InstantiatePeripherals(m_sceneJson, m_renderObjects);
		}
		Log::Write("Successfully created %i objects\n", m_renderObjects->Size());

		// Bind together to create the scene DAG
		{
			Log::Indent indent("Binding scene objects...\n");
			m_renderObjects->Bind();
		}
		Log::Write("Okay!\n");

		// Synchronise the render objects with the device
		{
			Log::Indent indent("Synchronising scene with device...\n");
			m_renderObjects->Synchronise();
		}
		Log::Write("Okay!\n");

		// Finalise all the render objects
		m_renderObjects->Finalise();

		// Get a handle to the first wavefront tracer object we find
		m_wavefrontTracer = m_renderObjects->FindFirstOfType<Cuda::Host::WavefrontTracer>();
		Assert(m_wavefrontTracer, "No wavefront tracer objects were instantiated.");

		// Prepare the scene 
		Prepare();
	}

	Log::Snapshot deltaState = Log::GetMessageState() - beginState;
	Log::Write("*** BUILD COMPLETE***\n");
	Log::Write("%i objects: %i errors, %i warnings\n", m_renderObjects->Size(), deltaState[kLogError], deltaState[kLogWarning]);
}

void RenderManager::Destroy()
{
	if (!m_managerThread.joinable()) { return; }

	Log::Indent("Shutting down...\n");

	Log::Write("Halting renderer...\n");
	m_threadSignal.store(kHalt);
	m_managerThread.join();
	Log::Write("Done!\n");

	{
		Log::Indent indent(tfm::format("Destroying %i managed assets:", Cuda::AR().Size()));
		Cuda::AR().Report();

		// Destroy the render objects
		m_renderObjects.DestroyAsset();

		// Destroy assets
		m_compositeImage.DestroyAsset();
	}

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
	/*cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
	memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));
	externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
	externalSemaphoreWaitParams.flags = 0;

	checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_D3DStream));*/

	// If the next frame is not ready to be rendered yet, wait until it is ready.
	if (m_d3dFence->GetCompletedValue() < currentFenceValue)
	{
		//std::printf("%i is waiting for %i (%i)\n", m_frameIndex, m_fenceValues[m_frameIndex], currentFenceValue);
		ThrowIfFailed(m_d3dFence->SetEventOnCompletion(currentFenceValue, m_fenceEvent));
		WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
	}

	// Only emplace another copy call if the previous one has successfully executed
	if (cudaEventQuery(m_renderEvent) == cudaSuccess)
	{
		m_compositeImage->CopyImageToD3DTexture(m_clientWidth, m_clientHeight, m_cuSurface, m_D3DStream);
		IsOk(cudaEventRecord(m_renderEvent));
	}
	//IsOk(cudaStreamSynchronize(m_D3DStream));

	/*cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
	std::memset(&externalSemaphoreSignalParams, 0, sizeof(externalSemaphoreSignalParams));
	externalSemaphoreSignalParams.params.fence.value = ++currentFenceValue;
	externalSemaphoreSignalParams.flags = 0;

	checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_D3DStream));*/

	m_d3dFence->Signal(++currentFenceValue);
}

void RenderManager::LinkSynchronisationObjects(ComPtr<ID3D12Device>& d3dDevice, ComPtr<ID3D12Fence>& d3dFence, HANDLE fenceEvent)
{
	m_d3dFence = d3dFence;
	m_fenceEvent = fenceEvent;

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

void RenderManager::OnJson(const Json::Document& patchJson)
{
	std::lock_guard<std::mutex> lock(m_renderResourceMutex);

	m_paramsPatchJson.Clear();
	m_paramsPatchJson.DeepCopy(patchJson);

	Log::Debug("Updated! %s\n", m_paramsPatchJson.Stringify(true));

	m_dirtiness = kSoftReset;
}

void RenderManager::StartBake(const std::string& usdExportPath)
{
	if (!m_lightProbeCamera)
	{
		Log::Error("ERROR: Can't start a bake because there are no light probe cameras in this scene.\n");
		return;
	}

	Assert(!usdExportPath.empty());
	AssertMsg(m_bakeStatus == BakeStatus::kReady, "A bake has already been started. Wait for it to complete or abort it before starting a new one.");

	std::lock_guard<std::mutex> lock(m_renderResourceMutex);
	 
	m_usdExportPath = usdExportPath;
	m_lightProbeCamera->SetExporterState(Cuda::Host::LightProbeCamera::kArmed);
	
	m_dirtiness = kHardReset;
	m_bakeStatus = BakeStatus::kRunning;
}

void RenderManager::AbortBake()
{
	m_bakeStatus = BakeStatus::kHalt;

	if (m_lightProbeCamera)
	{
		m_lightProbeCamera->SetExporterState(Cuda::Host::LightProbeCamera::kDisarmed);
	}
}

void RenderManager::Prepare()
{
	m_liveCamera = nullptr;

	// Get a list of cameras that are marked as active. 
	m_activeCameras = m_renderObjects->FindAllOfType<Cuda::Host::Camera>([this](const Cuda::AssetHandle<Cuda::Host::Camera>& object) -> bool
		{
			Assert(object);
			const auto& params = object->GetParams();
			if (!params.isActive) { return false; }

			if (params.isLive) { m_liveCamera = object; }
			return true;
		});

	// Get a handle to the light probe camera (if any)
	for (auto& camera : m_activeCameras)
	{
		if (m_lightProbeCamera = camera.DynamicCast<Cuda::Host::LightProbeCamera>()) { break; }
	}

	// Spit out some errors if needs be
	if (m_activeCameras.empty()) { Log::Error("WARNING: No camera objects were instantiated and enabled.\n"); }
	else if (!m_liveCamera) { Log::Warning("WARNING: There are no live cameras in this scene. The viewport will not update.\n"); }

	//if (!m_lightProbeCamera) { Log::Error("WARNING: There are no light probe cameras in this scene. Baking is disabled.\n"); }
}

void RenderManager::Start()
{
	m_threadSignal = kRun;
	m_managerThread = std::thread(std::bind(&RenderManager::Run, this));

	m_renderStartTime = std::chrono::high_resolution_clock::now();

	Assert(m_managerThread.joinable());
}

void RenderManager::ClearRenderStates()
{
	// Clear the render states of all active camera objects
	for (auto camera : m_activeCameras) { camera->ClearRenderState(); }

	// Reset the render manager state
	m_frameIdx = 0;
	m_dirtiness = kClean;
}

void RenderManager::Run()
{
	checkCudaErrors(cudaStreamSynchronize(m_renderStream));

	constexpr float kTargetFps = 60.0f;
	constexpr int kMaxSubframes = 1;
	int numSubframes = kMaxSubframes;
	m_frameIdx = 0;
	m_iterationIdx = 0;

	while (m_threadSignal.load() == kRun)
	{
		/*Timer timer([&](float elapsed) -> std::string
			{
				const float fps = 1.0f / elapsed;
				return tfm::format("Iteration %i: Frame %i, Subframes: %i, FPS: %f ", iterationIdx, frameIdx, numSubframes, fps);
			});*/
		Timer timer;

		// Has the scene graph been dirtied?
		if (m_dirtiness != kClean && !m_activeCameras.empty())
		{
			if (m_dirtiness == kHardReset || m_frameIdx >= 2)
			{
				PatchSceneObjects();
			}

			// Reset the render state
			ClearRenderStates();
		}

		// Render a pass through each camera to its render state
		for (auto& camera : m_activeCameras)
		{
			m_wavefrontTracer->AttachCamera(camera);

			// Render up to N subframes
			for (int subFrameIdx = 0; subFrameIdx < numSubframes; subFrameIdx++)
			{
				std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - m_renderStartTime;

				// Notify objects that we're about to start the pass
				m_wavefrontTracer->OnPreRenderPass(timeDiff.count(), m_frameIdx);
				camera->OnPreRenderPass(timeDiff.count(), m_frameIdx);

				// Trace those rays through the wavefront tracer 
				m_wavefrontTracer->Trace();

				m_frameIdx++;
			}

			// If this wavefront tracer is live, update the composite image
			if (camera == m_liveCamera)
			{
				camera->Composite(m_compositeImage);
			}

			camera->OnPostRenderPass();

			checkCudaErrors(cudaStreamSynchronize(m_renderStream));
		}

		// Handle any post-frame baking operations
		OnBakePostFrame();

		m_iterationIdx++;
		m_frameTimes[m_frameIdx % m_frameTimes.size()] = timer.Get();
		float meanFrameTime = 0.0f;
		for (const auto& ft : m_frameTimes)
		{
			meanFrameTime += ft;
		}
		meanFrameTime /= math::min(m_frameIdx, int(m_frameTimes.size()));

		if (m_liveCamera)
		{
			std::lock_guard<std::mutex> lock(m_jsonMutex);
			Cuda::Device::RenderState::Stats stats;
			m_liveCamera->GetRenderStats()->SynchroniseHost(stats);

			m_renderStatsJson.Clear();
			m_renderStatsJson.AddValue("frameTime", timer.Get());
			m_renderStatsJson.AddValue("meanFrameTime", meanFrameTime);
			m_renderStatsJson.AddValue("deadRays", stats.deadRays);
		}

		//std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

void RenderManager::PatchSceneObjects()
{
	std::lock_guard<std::mutex> lock(m_renderResourceMutex);

	if (!m_paramsPatchJson.NumMembers()) { return; }

	int validPatches = 0;
	for (Json::Node::Iterator it = m_paramsPatchJson.begin(); it != m_paramsPatchJson.end(); ++it)
	{
		Cuda::AssetHandle<Cuda::Host::RenderObject> asset = m_renderObjects->FindByDAG(it.Name());
		if (asset)
		{
			asset->FromJson(*it, Json::kSilent);
			validPatches++;
		}
	}

	// Prepare the scene for rendering
	if (validPatches > 0) { Prepare(); }

	m_paramsPatchJson.Clear();
}

void RenderManager::OnBakePostFrame()
{
	if (m_bakeStatus == BakeStatus::kHalt)
	{
		// Do shutdown stuff here
		m_bakeStatus = BakeStatus::kReady;
	}

	else if (m_bakeStatus == BakeStatus::kRunning && m_dirtiness == kClean)
	{
		if (m_lightProbeCamera->GetExporterState() == Cuda::Host::LightProbeCamera::kArmed)
		{
			m_bakeProgress = m_lightProbeCamera->GetBakeProgress();
			if (m_bakeProgress == 1.0f)
			{
				m_lightProbeCamera->ExportProbeGrid(m_usdExportPath);
				m_bakeStatus = BakeStatus::kReady;
				Log::Debug("Export!");
			}
		}
		else
		{
			Log::Warning("Internal warning: probe grid exporter is not armed.\n");
			m_bakeStatus = BakeStatus::kReady;
		}
	}
}