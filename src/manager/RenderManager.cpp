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
#include "io/ImageIO.h"
#include "generic/GlobalStateAuthority.h"

RenderManager::RenderManager() : 
	m_threadSignal(kIdle),
	m_dirtiness(kClean),
	m_frameIdx(0),
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
	m_compositeImage = Cuda::AssetHandle<Cuda::Host::ImageRGBA>("id_compositeImage", clientWidth, clientHeight, m_renderStream);
	//m_wavefrontTracer = Cuda::AssetHandle<Cuda::Host::WavefrontTracer>("id_wavefrontTracer", m_renderStream);

	Cuda::VerifyTypeSizes();

	IsOk(cudaDeviceSynchronize());
}

void RenderManager::LoadDefaultScene()
{
	LoadScene(GSA().GetDefaultScenePath());
}

void RenderManager::LoadScene(const std::string filePath)
{
	Log::Write("Loading scene file '%s'...\n", filePath);
	try
	{
		m_sceneJson.Deserialise(filePath);
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Unable to load scene JSON: %s", err.what());
		return;
	}
	
	// Unload the old scene
	UnloadScene();	

	// Load and build the new scene
	Build(m_sceneJson);
}

void RenderManager::UnloadScene(bool report)
{
	if (!m_renderObjects) { return; }
	
	Log::Indent indent("Unloading scene...");
	{
		// Stop the renderer
		StopRenderer();

		// Destroy the references to the linked render objects
		m_wavefrontTracer = nullptr;
		m_liveCamera = nullptr;
		m_lightProbeCamera = nullptr;
		m_activeCameras.clear();
		
		if (report)
		{
			Log::Indent indent(tfm::format("Destroying %i managed assets:", Cuda::AR().Size()));
			Cuda::AR().Report();
		}

		// Destroy the render objects
		m_renderObjects.DestroyAsset();
	}


}

void RenderManager::Build(const Json::Document& sceneJson)
{
	Assert(m_threadSignal == kIdle);
	Assert(!m_renderObjects);

	Log::NL();
	const Log::Snapshot beginState = Log::GetMessageState();
	{
		Log::Indent indent("Building render manager...\n");		

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


		Cuda::AR().Report();

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

	// Finally, start the renderer
	StartRenderer();
}

void RenderManager::StopRenderer()
{
	if (!m_managerThread.joinable() || m_threadSignal != kRun)
	{
		Log::Warning("Renderer is not running.");
		return; 
	}

	Log::Write("Halting renderer...\r");
	m_threadSignal.store(kHalt);
	m_managerThread.join();

	Log::Write("Done!\n");
}

void RenderManager::Destroy()
{	
	UnloadScene(true);

	// Destroy assets
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

void RenderManager::StartBake(const Cuda::LightProbeGridExportParams& params)
{
	if (!m_lightProbeCamera)
	{
		Log::Error("ERROR: Can't start a bake because there are no light probe cameras in this scene.\n");
		return;
	}

	Assert(!params.usdExportPaths.empty());
	AssertMsg(m_bakeStatus == BakeStatus::kReady, "A bake has already been started. Wait for it to complete or abort it before starting a new one.");

	std::lock_guard<std::mutex> lock(m_renderResourceMutex);
	 
	m_probeGridExportParams = params;
	m_lightProbeCamera->SetExporterState(Cuda::Host::LightProbeCamera::kArmed);
	
	m_dirtiness = kHardReset;
	m_bakeStatus = BakeStatus::kRunning;
}

void RenderManager::ExportLiveViewport(const std::string& pngExportPath)
{
	if (m_exportToPNG) 
	{ 
		m_exportToPNG = false;
		return; 
	}

	m_pngExportPath = pngExportPath;
	m_exportToPNG = true;
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

void RenderManager::StartRenderer()
{
	Log::Write("Starting rendering...\b");

	m_threadSignal = kRun;
	m_managerThread = std::thread(std::bind(&RenderManager::Run, this));

	m_renderStartTime = std::chrono::high_resolution_clock::now();

	Assert(m_managerThread.joinable());

	Log::Write("Okay!");
}

void RenderManager::ClearRenderStates()
{
	// Clear the render states of all active camera objects
	for (auto& camera : m_activeCameras) { camera->ClearRenderState(); }

	// Notify scene objects that the render has been restarted
	for (auto& object : *m_renderObjects) { object->OnPreRender(); }

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

	try
	{
		while (m_threadSignal.load() == kRun)
		{
			/*Timer timer([&](float elapsed) -> std::string
				{
					const float fps = 1.0f / elapsed;
					return tfm::format("Iteration %i: Frame %i, Subframes: %i, FPS: %f ", iterationIdx, frameIdx, numSubframes, fps);
				});*/
			Timer timer;

			// Has the scene graph been dirtied?
			if (!m_activeCameras.empty() && (m_dirtiness == kHardReset || (m_dirtiness == kSoftReset && m_frameIdx >= 2)))
			{
				PatchSceneObjects();

				// Reset the render state
				ClearRenderStates();
			}

			Assert(!(m_bakeStatus == BakeStatus::kRunning && m_wavefrontTracer->GetParams().shadingMode != Cuda::kShadeFull));

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
				}

				// If this wavefront tracer is live, update the composite image
				if (camera == m_liveCamera)
				{
					camera->Composite(m_compositeImage);
				}

				checkCudaErrors(cudaStreamSynchronize(m_renderStream));
			}

			// Signal to the render objects that the pass is complete
			for (auto& object : *m_renderObjects) { object->OnPostRenderPass(); }

			// Handle any post-frame baking operations
			OnBakePostFrame();

			// Compute some stats on the framerate
			m_frameIdx++;
			m_frameTimes[m_frameIdx % m_frameTimes.size()] = timer.Get();
			m_meanFrameTime = 0.0f;
			for (const auto& ft : m_frameTimes)
			{
				m_meanFrameTime += ft;
			}
			m_meanFrameTime /= math::min(m_frameIdx, int(m_frameTimes.size()));

			// Gather statistics for the render objects
			GatherRenderObjectStatistics();
		}
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());
		m_threadSignal.store(kAssert);
	}
	catch (...)
	{
		Log::Error("Unhandled error");
		m_threadSignal.store(kAssert);
	}

	// Signal that the renderer has finished
	m_threadSignal.store(kIdle);
}

void RenderManager::GatherRenderObjectStatistics()
{
	// Limit this operation to 2 times per second
	if (m_renderStatsTimer.Get() < 0.5f) { return; }
	
	Json::Document renderObjectJson;
	Json::Document aggregatedStatsJson;
	for (auto& object : *m_renderObjects) 
	{  		
		// Poll each render object for the latest stats
		if (object->HasDAGPath() && object->EmitStatistics(renderObjectJson))
		{
			Json::Node childNode = aggregatedStatsJson.AddChildObject(object->GetDAGPath());
			childNode.DeepCopy(renderObjectJson);
			renderObjectJson.Clear();
		}
	}

	//Log::Debug(aggregatedStatsJson.Stringify(true));

	aggregatedStatsJson.AddValue("frameIdx", m_frameIdx);
	aggregatedStatsJson.AddValue("meanFrameTime", m_meanFrameTime);

	std::lock_guard<std::mutex> lock(m_jsonMutex);
	m_renderStatsJson.DeepCopy(aggregatedStatsJson);
	m_renderStatsTimer.Reset();
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

	// Some objects may need to adjust their bindings now that the scene graph has been dirtied
	for (auto& object : *m_renderObjects)
	{
		object->OnUpdateSceneGraph(*m_renderObjects);
	}

	// Prepare the scene for rendering
	if (validPatches > 0) { Prepare(); }

	m_paramsPatchJson.Clear();
}

void RenderManager::OnBakePostFrame()
{
	if (m_exportToPNG)
	{
		std::vector<Cuda::vec4> rawData;
		Cuda::ivec2 dataDimensions;
		m_liveCamera->GetRawAccumulationData(rawData, dataDimensions);

		ImageIO::WriteAccumulationBufferPNG(rawData, dataDimensions, m_pngExportPath, 2.2f);

		m_exportToPNG = false;
	}
	
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
				m_lightProbeCamera->ExportProbeGrid(m_probeGridExportParams);
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