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
	m_threadSignal(kRenderManagerIdle),
	m_dirtiness(kDirtinessStateClean),
	m_frameIdx(0),
	m_frameTimes(20)
{
	// Register the list of jobs
	RegisterJob(m_bake.job, "bake", &RenderManager::OnBakeDispatch, &RenderManager::OnBakePoll);
	RegisterJob(m_exportViewportJob, "exportViewport", &RenderManager::OnExportViewportDispatch, &RenderManager::OnNullPoll);
	RegisterJob(m_statsJob, "getStats", &RenderManager::OnGatherStatsDispatch, &RenderManager::OnGatherStatsPoll);
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
	Assert(m_threadSignal == kRenderManagerIdle);
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
	if (!m_managerThread.joinable() || m_threadSignal != kRenderManagerRun)
	{
		Log::Warning("Renderer is not running.");
		return; 
	}

	Log::Write("Halting renderer...\r");
	m_threadSignal.store(kRenderManagerHalt);
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

bool RenderManager::OnBakeDispatch(Job& job)
{
	std::string action;
	job.json.GetValue("action", action, Json::kRequiredAssert);

	// Aborting the action...
	if (action == "abort")
	{
		job.state = kRenderManagerJobAborting;
		return true;
	}
	
	// Sanity checks
	AssertMsg(action == "start", "Valid bake actions are 'start' and 'abort'");
	AssertMsgFmt(job.state == kRenderManagerJobIdle, "The bake job is not idle (code %i). Wait for it to complete or abort it before starting a new one.", job.state.load());

	// Read in the parameters from the JSON dictionary
	try
	{
		std::lock_guard<std::mutex> lock(m_jsonInputMutex);

		job.json.GetEnumeratedParameter("type", std::vector<std::string>({ "probeGrid", "render" }), m_bake.type, Json::kRequiredAssert);
		if (m_bake.type == kBakeTypeProbeGrid)
		{		
			job.json.GetValue("exportToUSD", m_bake.probeGridExportParams.exportToUSD, Json::kRequiredWarn);
			job.json.GetValue("minGridValidity", m_bake.probeGridExportParams.minGridValidity, Json::kRequiredWarn);
			job.json.GetValue("maxGridValidity", m_bake.probeGridExportParams.maxGridValidity, Json::kRequiredWarn);
			job.json.GetArrayValues("usdExportPaths", m_bake.probeGridExportParams.usdExportPaths, Json::kRequiredWarn);

			Assert(!m_bake.probeGridExportParams.usdExportPaths.empty());
		}
		else if (m_bake.type == kBakeTypeRender)
		{			
			job.json.GetValue("pngExportPath", m_bake.pngExportPath, Json::kRequiredWarn);
		}
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Error: invalid command parameters: %s", err.what());
		return false;
	}
	
	// Dirty the scene, and mark the command has having started
	m_dirtiness = kDirtinessStateHardReset;
	job.state = kRenderManagerJobDispatched;
	m_bake.succeeded = true;

	return true;
}

bool RenderManager::OnBakePoll(Json::Node& outJson, const Job& job)
{
	outJson.AddValue("progress", m_bake.progress);
	if (m_bake.job.state == kRenderManagerJobCompleted)
	{
		outJson.AddValue("succeeded", m_bake.succeeded);
	}
	return true;
}

bool RenderManager::OnGatherStatsDispatch(Job& job)
{
	job.state = kRenderManagerJobDispatched;
	return true;
}

bool RenderManager::OnGatherStatsPoll(Json::Node& outJson, const Job& job)
{
	if (job.state != kRenderManagerJobCompleted) { return true; }

	// If render object stats have been been generated, copy them over
	if (m_renderObjectStatsJson.NumMembers())
	{
		std::lock_guard<std::mutex> lock(m_jsonOutputMutex);

		Json::Node statsJson = outJson.AddChildObject("renderObjects");
		statsJson.DeepCopy(m_renderObjectStatsJson);
		m_renderObjectStatsJson.Clear();
	}

	return true;
}

bool RenderManager::OnExportViewportDispatch(Job& job)
{
	job.state = kRenderManagerJobDispatched;
	return true;
}

bool RenderManager::PollRenderState(Json::Document& stateJson)
{
	/* 
		IMPORTANT: This function may be polled rapidly by the UI, so it must remain as lightweight as possible and 
		only return JSON data where absolutely necessary

		{
			"renderManager": { 
				"frameIdx": 10,
				...
			},
			"jobs": {
				"bake": {
					"state": 1,
					...
				}
			}
		}

	*/
	
	stateJson.Clear();	

	// Add some generic data about the renderer that's exported each time the state is polled
	Json::Node managerJson = stateJson.AddChildObject("renderManager");
	managerJson.AddValue("frameIdx", m_frameIdx);
	managerJson.AddValue("meanFrameTime", m_meanFrameTime);
	const int threadSignal = m_threadSignal;
	managerJson.AddValue("rendererStatus", threadSignal);

	// Report data on the status of the commands that may be running or have data waiting to be dispatched
	Json::Node jobJson = stateJson.AddChildObject("jobs");
	for (auto& jobObject : m_jobMap)
	{
		// Only emit JSON data if the command isn't idle i.e. it's dispatched or completed
		auto& job = jobObject.second;
		if (job.state != kRenderManagerJobIdle)
		{
			Json::Node statsJson = jobJson.AddChildObject(jobObject.first);
			const int state = job.state;
			statsJson.AddValue("state", state);

			// If this command has a functor, call it now
			if (job.onPoll) { job.onPoll(statsJson, job); }

			// Flip the state from completed to idle once the data has been polled
			if (job.state == kRenderManagerJobCompleted) { job.state = kRenderManagerJobIdle; }
		}
	}

	return true;
}

void RenderManager::Dispatch(const Json::Document& rootJson)
{
	/*
		Job dictionary contains patch data for the render objects and command data for the render manager.
		Sample JSON file should be:

		{
			"patches": {
				...
			},
			"commands":	{
				"bake": {
					"action": "start"
				},
				...
			}
		}
	*/
	
	Assert(rootJson.NumMembers() != 0);

	//Log::Debug(rootJson.Stringify(true));

	if (rootJson.GetChildObject("patches", Json::kSilent | Json::kLiteralID))
	{
		// Overwrite the command list with the new data
		std::lock_guard<std::mutex> lock(m_jsonInputMutex);

		m_patchJson.DeepCopy(rootJson);
		
		// Found a scene object parameter parameter patch, so signal that the scene graph is dirty
		m_dirtiness = kDirtinessStateSoftReset;

		Log::Debug("Updated! %s\n", m_patchJson.Stringify(true));
	}

	const Json::Node commandsJson = rootJson.GetChildObject("commands", Json::kSilent | Json::kLiteralID);
	if (commandsJson)
	{
		// Examine the command list
		for (auto nodeIt = commandsJson.begin(); nodeIt != commandsJson.end(); ++nodeIt)
		{
			auto commandIt = m_jobMap.find(nodeIt.Name());
			if (commandIt != m_jobMap.end())
			{
				try
				{
					auto& job = commandIt->second;
					
					// Copy any JSON data that accompanies this command
					job.json.DeepCopy(*nodeIt);

					// Call the dispatch functor
					Assert(job.onDispatch);
					job.onDispatch(job);

					//Log::System("Dispatched new job: %s", nodeIt.Name());
				}
				catch (const std::runtime_error& err)
				{
					Log::Error("Error: render manager command '%s' failed: %s", nodeIt.Name(), err.what());
				}
			}
			else
			{
				Log::Error("Error: '%s' is not a valid render manager command", nodeIt.Name());
			}
		}		
	}
}

void RenderManager::Prepare()
{
	m_liveCamera = nullptr;
	m_lightProbeCamera = nullptr;

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

	if (!m_lightProbeCamera) { Log::Warning("WARNING: There are no light probe cameras in this scene. Baking is disabled.\n"); }
}

void RenderManager::StartRenderer()
{
	Log::Write("Starting rendering...\b");

	m_threadSignal = kRenderManagerRun;
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
	m_dirtiness = kDirtinessStateClean;
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
		while (m_threadSignal.load() == kRenderManagerRun)
		{
			/*Timer timer([&](float elapsed) -> std::string
				{
					const float fps = 1.0f / elapsed;
					return tfm::format("Iteration %i: Frame %i, Subframes: %i, FPS: %f ", iterationIdx, frameIdx, numSubframes, fps);
				});*/
			Timer timer;

			// Has the scene graph been dirtied?
			if (m_dirtiness == kDirtinessStateHardReset || (m_dirtiness == kDirtinessStateSoftReset && m_frameIdx >= 2))
			{
				PatchSceneObjects();

				// Reset the render state
				ClearRenderStates();
			}

			Assert(!(m_bake.job.state == kRenderManagerJobDispatched && m_wavefrontTracer->GetParams().shadingMode != Cuda::kShadeFull));

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

			GatherRenderObjectStatistics(); // Gather statistics from the render objects			
		}
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());
		m_threadSignal.store(kRenderManagerError);
	}
	catch (...)
	{
		Log::Error("Unhandled error");
		m_threadSignal.store(kRenderManagerError);
	}

	// Signal that the renderer has finished
	m_threadSignal.store(kRenderManagerIdle);
}

void RenderManager::GatherRenderObjectStatistics()
{	
	if (m_statsJob.state != kRenderManagerJobDispatched) { return; }

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

	std::lock_guard<std::mutex> lock(m_jsonOutputMutex);
	m_renderObjectStatsJson.DeepCopy(aggregatedStatsJson);
	m_renderStatsTimer.Reset();

	m_statsJob.state = kRenderManagerJobCompleted;
}

void RenderManager::PatchSceneObjects()
{
	std::lock_guard<std::mutex> lock(m_jsonInputMutex);

	Json::Node patchJson = m_patchJson.GetChildObject("patches", Json::kRequiredAssert);

	if (!patchJson.NumMembers()) { return; }

	int validPatches = 0;
	for (Json::Node::Iterator it = patchJson.begin(); it != patchJson.end(); ++it)
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

	m_patchJson.Clear();
}

void RenderManager::OnBakePostFrame()
{
	// If a viewport export has been requested, do so now
	if (m_exportViewportJob.state == kRenderManagerJobDispatched)
	{
		auto& job = m_exportViewportJob;
		
		std::vector<Cuda::vec4> rawData;
		Cuda::ivec2 dataDimensions;
		m_liveCamera->GetRawAccumulationData(rawData, dataDimensions);

		std::string exportPath;
		if (!job.json.GetValue("path", exportPath, Json::kRequiredWarn)) { return; }

		ImageIO::WriteAccumulationBufferPNG(rawData, dataDimensions, exportPath, 2.2f);

		job.state = kRenderManagerJobIdle;
	}

	// Are we aborting the bake job?
	if (m_bake.job.state == kRenderManagerJobAborting)
	{
		m_bake.job.state = kRenderManagerJobIdle;
		return;
	}

	// Not baking or the scene graph is dirty? We're done.
	if (!(m_bake.job.state & kRenderManagerJobActive) || m_dirtiness != kDirtinessStateClean) { return; }

	// If the job has just been dispatched, do some pre-flight checks
	if (m_bake.job.state & kRenderManagerJobDispatched)
	{		
		if (m_bake.type == kBakeTypeProbeGrid)
		{
			if (!m_lightProbeCamera)
			{
				Log::Error("ERROR: Can't start a bake because there are no light probe cameras in this scene.\n");
				m_bake.job.state = kRenderManagerJobIdle;
				return;
			}
		}
		else if (m_bake.type == kBakeTypeRender && !m_liveCamera)
		{
			Log::Error("ERROR: Can't start a render bake because there are no live cameras in this scene.\n");
			m_bake.job.state = kRenderManagerJobIdle;
			return;
		}
		
		m_bake.job.state = kRenderManagerJobRunning;
	}
	
	// Baking a probe grid...
	if (m_bake.type == kBakeTypeProbeGrid)
	{		
		constexpr float kMinSamplesForEstimate = 8.0f;
		const auto& stats = m_lightProbeCamera->PollBakeProgress();
		m_bake.progress = stats.bakeProgress;

		// If the mean validity is outside the pre-set bounds, complete the job without 
		if (stats.minMaxSamples[1] > kMinSamplesForEstimate && stats.meanGridValidity >= 0.0f &&
			(stats.meanGridValidity < m_bake.probeGridExportParams.minGridValidity ||
				stats.meanGridValidity > m_bake.probeGridExportParams.maxGridValidity))
		{
			Log::Error("Warning: bake was aborted because the grid validity %f was outside the specified bounds [%f, %f]", 
				stats.meanGridValidity, m_bake.probeGridExportParams.minGridValidity, m_bake.probeGridExportParams.maxGridValidity);

			m_bake.succeeded = false;
			m_bake.job.state = kRenderManagerJobCompleted;
			return;
		}

		if (m_bake.progress >= 1.0f)
		{
			m_lightProbeCamera->ExportProbeGrid(m_bake.probeGridExportParams);
			m_bake.job.state = kRenderManagerJobCompleted;
		}		
	}

	// Baking a regular render...
	else if (m_bake.type == kBakeTypeRender)
	{
		// Estimate the render progress. This isn't terribly accurate, but it's good enough for now.
		const auto& liveCam = m_liveCamera->GetParams();
		m_bake.progress = float(m_frameIdx) / float(liveCam.maxSamples * liveCam.overrides.maxDepth);
		if (m_bake.progress > 1.0f)
		{
			std::vector<Cuda::vec4> rawData;
			Cuda::ivec2 dataDimensions;
			m_liveCamera->GetRawAccumulationData(rawData, dataDimensions);

			ImageIO::WriteAccumulationBufferPNG(rawData, dataDimensions, m_bake.pngExportPath, 2.2f);

			m_bake.succeeded = true;
			m_bake.job.state = kRenderManagerJobCompleted;
		}
	}
}