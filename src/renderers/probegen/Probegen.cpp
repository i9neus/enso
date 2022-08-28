#include "Probegen.h"

#include "win/SecurityAttributes.h"
#include "win/DXSampleHelper.h"
#include "thirdparty/nvidia/helper_cuda.h"

#include "kernels/CudaTests.cuh"
#include "kernels/CudaCommonIncludes.cuh"
#include "kernels/CudaRenderObjectFactory.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAssetContainer.cuh"
#include "kernels/tracables/CudaTracable.cuh"
#include "kernels/cameras/CudaCamera.cuh"
#include "kernels/cameras/CudaPerspectiveCamera.cuh"

#include "io/USDIO.h"
#include "io/ImageIO.h"
#include "generic/GlobalStateAuthority.h"
#include "generic/debug/ProcessMemoryMonitor.h"

Probegen::Probegen() :
	m_threadSignal(kRenderManagerIdle),
	m_frameTimes(20)
{
	// Register the list of jobs
	RegisterJob(m_bake.job, "bake", &Probegen::OnBakeDispatch, &Probegen::OnBakePoll);
	RegisterJob(m_exportViewportJob, "exportViewport", &Probegen::OnDefaultDispatch, &Probegen::OnDefaultPoll);
	RegisterJob(m_exportGridsJob, "exportGrids", &Probegen::OnExportGridsDispatch, &Probegen::OnDefaultPoll);
	RegisterJob(m_renderStatsJob, "getRenderStats", &Probegen::OnDefaultDispatch, &Probegen::OnGatherRenderStatsPoll);
	RegisterJob(m_memoryStatsJob, "getMemoryStats", &Probegen::OnDefaultDispatch, &Probegen::OnGatherMemoryStatsPoll);

	// Register the list of render commands
	m_commandMap.emplace(kRenderManagerClearStates, std::bind(&Probegen::ClearAllRenderStates, this));
	m_commandMap.emplace(kRenderMangerUpdateParams, std::bind(&Probegen::PatchSceneObjects, this));
}

/*void Probegen::Initialise()
{

}*/

void Probegen::OnResizeClient()
{

}

void Probegen::LoadDefaultScene()
{
	LoadScene(GSA().GetDefaultScenePath());
}

void Probegen::LoadScene(const std::string filePath)
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

void Probegen::UnloadScene(bool report)
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
			Log::Indent indent(tfm::format("Destroying %i managed assets:", Cuda::AR().NumAssets()));
			Cuda::AR().Report();
		}

		// Destroy the render objects
		m_renderObjects.DestroyAsset();
	}

}

void Probegen::Build(const Json::Document& sceneJson)
{
	Assert(m_threadSignal == kRenderManagerIdle);
	Assert(!m_renderObjects);

	Log::NL();
	const Log::Snapshot beginState = Log::GetMessageState();
	{
		Log::Indent indent("Building render manager...\n");

		// Create a container for the render objects
		m_renderObjects = Cuda::CreateAsset<Cuda::RenderObjectContainer>("__root_renderObjectContainer");

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
	Log::Success("*** BUILD COMPLETE***\n");
	Log::Success("%i objects: %i errors, %i warnings\n", m_renderObjects->Size(), deltaState[kLogError], deltaState[kLogWarning]);

	// Finally, start the renderer
	StartRenderer();
}

void Probegen::StopRenderer()
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

void Probegen::Destroy()
{
	UnloadScene(true);
}

void Probegen::EmplaceRenderCommand(const int cmd)
{
	std::lock_guard<std::mutex> lock(m_commandMutex);
	m_commandSet.emplace(cmd);
}

bool Probegen::OnBakeDispatch(Job& job)
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

		job.json.GetValue("type", m_bake.type, Json::kRequiredAssert);
		if (m_bake.type & (kBakeTypeNoisyProbeGrid | kBakeTypeReferenceProbeGrid))
		{
			job.json.GetValue("isArmed", m_bake.probeGridExportParams.isArmed, Json::kRequiredWarn);
			job.json.GetValue("minGridValidity", m_bake.probeGridExportParams.minGridValidity, Json::kRequiredWarn);
			job.json.GetValue("maxGridValidity", m_bake.probeGridExportParams.maxGridValidity, Json::kRequiredWarn);
			job.json.GetArrayValues("exportPaths", m_bake.probeGridExportParams.exportPaths, Json::kRequiredWarn);

			Assert(!m_bake.probeGridExportParams.exportPaths.empty());
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
	EmplaceRenderCommand(kRenderManagerClearStates);

	job.state = kRenderManagerJobDispatched;
	m_bake.succeeded = true;

	return true;
}

bool Probegen::OnBakePoll(Json::Node& outJson, Job& job)
{
	outJson.AddValue("progress", m_bake.progress);
	if (m_bake.job.state == kRenderManagerJobCompleted)
	{
		outJson.AddValue("succeeded", m_bake.succeeded);
	}
	return true;
}

bool Probegen::OnExportGridsDispatch(Job& job)
{
	job.state = kRenderManagerJobDispatched;
	return true;
}

bool Probegen::OnGatherRenderStatsPoll(Json::Node& outJson, Job& job)
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

bool Probegen::OnGatherMemoryStatsPoll(Json::Node& outJson, Job& job)
{
	std::lock_guard<std::mutex> lock(m_jsonOutputMutex);
	job.state = kRenderManagerJobCompleted;

	// Device memory stats
	size_t freeBytes, totalBytes;
	IsOk(cudaMemGetInfo(&freeBytes, &totalBytes));
	Json::Node deviceJson = outJson.AddChildObject("device");
	//deviceJson.AddValue("name", std::string(GetCudaDeviceProperties().name));
	deviceJson.AddValue("freeMB", float(freeBytes) / 1048576.f);
	deviceJson.AddValue("totalMB", float(totalBytes) / 1048576.f);

	// Host memory stats
	const int64_t workingSetSize = ProcessMemoryMonitor::Get().GetWorkingSetSize();
	Json::Node hostJson = outJson.AddChildObject("host");
	hostJson.AddValue("totalMB", float(workingSetSize) / 1048576.f);

	// Asset stats
	Json::Node assetJson = outJson.AddChildObject("assets");
	const auto& registry = Cuda::GlobalResourceRegistry::Get();
	const auto& assetMap = registry.GetAssetMap();
	const auto& deviceMemoryMap = registry.GetDeviceMemoryMap();

	for (const auto& asset : deviceMemoryMap)
	{
		Json::Node objectJson = assetJson.AddChildObject(asset.first);
		auto it = assetMap.find(asset.first);
		if (it != assetMap.end())
		{
			auto& ptr = it->second;
			Assert(!ptr.expired());
			objectJson.AddValue("parent", ptr.lock()->GetParentAssetID());
		}
		objectJson.AddValue("currMB", float(asset.second.currentBytes) / 1048576.f);
		objectJson.AddValue("peakMB", float(asset.second.peakBytes) / 1048576.f);
		objectJson.AddValue("deltaMB", float(asset.second.deltaBytes) / 1048576.f);
	}

	return true;
}

bool Probegen::OnDefaultDispatch(Job& job)
{
	// Default dispatch handler for any job that doesn't have its own custom callback
	job.state = kRenderManagerJobDispatched;
	return true;
}

bool Probegen::PollRenderState(Json::Document& stateJson)
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
	managerJson.AddValue("frameIdx", m_liveCamera ? m_liveCamera->GetFrameIdx() : -1);
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

			// If this command has a functor, call it now
			if (job.onPoll) { job.onPoll(statsJson, job); }

			// Write the state to the stats
			statsJson.AddValue("state", job.state.load());

			// Flip the state from completed to idle once the data has been polled
			if (job.state == kRenderManagerJobCompleted) { job.state = kRenderManagerJobIdle; }
		}
	}

	return true;
}

void Probegen::Dispatch(const Json::Document& rootJson)
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
		std::lock_guard<std::mutex> lock(m_jsonInputMutex);

		// Overwrite the command list with the new data
		m_patchJson = rootJson;

		// Found a scene object parameter parameter patch, so signal that the scene graph is dirty
		EmplaceRenderCommand(kRenderMangerUpdateParams);

		//Log::Debug("Updated! %s\n", m_patchJson.Stringify(true));
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
				auto& job = commandIt->second;
				try
				{
					// Copy any JSON data that accompanies this command
					job.json = *nodeIt;

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

void Probegen::Prepare()
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

void Probegen::StartRenderer()
{
	Log::Write("Starting rendering...\b");

	m_threadSignal = kRenderManagerRun;
	m_managerThread = std::thread(std::bind(&Probegen::Run, this));

	m_renderStartTime = std::chrono::high_resolution_clock::now();

	Assert(m_managerThread.joinable());

	Log::Success("Okay!");
}

void Probegen::ClearAllRenderStates()
{
	// Clear the render states of all active camera objects
	for (auto& camera : m_activeCameras) { camera->ClearRenderState(); }

	// Notify scene objects that the render has been restarted
	for (auto& object : *m_renderObjects) { object->OnPreRender(); }
}

void Probegen::Run()
{
	checkCudaErrors(cudaStreamSynchronize(m_renderStream));

	constexpr float kTargetFps = 60.0f;
	constexpr int kMaxSubframes = 1;
	int numSubframes = kMaxSubframes;
	int frameIdx = 0;

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

			// Process any commands that are waiting
			if (!m_commandSet.empty())
			{
				std::lock_guard<std::mutex> lock(m_commandMutex);
				for (auto cmd : m_commandSet)
				{
					m_commandMap[cmd]();
				}
				m_commandSet.clear();
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
					m_wavefrontTracer->OnPreRenderPass(timeDiff.count());
					camera->OnPreRenderPass(timeDiff.count());

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
			frameIdx++;
			m_frameTimes[frameIdx % m_frameTimes.size()] = timer.Get();
			m_meanFrameTime = 0.0f;
			for (const auto& ft : m_frameTimes)
			{
				m_meanFrameTime += ft;
			}
			m_meanFrameTime /= min(frameIdx, int(m_frameTimes.size()));

			GatherRenderObjectStatistics(); // Gather statistics from the render objects			
		}
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());
		StackBacktrace::Print();

		m_threadSignal.store(kRenderManagerError);
	}
	catch (...)
	{
		Log::Error("Unhandled error");
		StackBacktrace::Print();

		m_threadSignal.store(kRenderManagerError);
	}

	// Signal that the renderer has finished
	m_threadSignal.store(kRenderManagerIdle);
}

void Probegen::GatherRenderObjectStatistics()
{
	if (m_renderStatsJob.state != kRenderManagerJobDispatched) { return; }

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
	m_renderObjectStatsJson = aggregatedStatsJson;
	m_renderStatsTimer.Reset();

	m_renderStatsJob.state = kRenderManagerJobCompleted;
}

uint Probegen::PatchSceneObjects()
{
	std::lock_guard<std::mutex> lock(m_jsonInputMutex);

	Json::Node patchJson = m_patchJson.GetChildObject("patches", Json::kRequiredAssert);

	if (!patchJson.NumMembers()) { return Cuda::kRenderObjectClean; }

	int validPatches = 0;
	uint dirtyFlags = Cuda::kRenderObjectClean;
	for (Json::Node::Iterator it = patchJson.begin(); it != patchJson.end(); ++it)
	{
		Cuda::AssetHandle<Cuda::Host::RenderObject> asset = m_renderObjects->FindByDAG(it.Name());
		if (asset)
		{
			dirtyFlags |= asset->FromJson(*it, Json::kSilent);
			validPatches++;
		}
	}

	if (dirtyFlags != Cuda::kRenderObjectClean)
	{
		// Some objects may need to adjust their bindings now that the scene graph has been dirtied
		{
			Log::Indent indent("Updating scene graph...");
			for (auto& object : *m_renderObjects)
			{
				object->OnUpdateSceneGraph(*m_renderObjects, dirtyFlags);
			}
		}

		// Prepare the scene for rendering
		if (validPatches > 0) { Prepare(); }
	}

	m_patchJson.Clear();
	return dirtyFlags;
}

void Probegen::OnBakePostFrame()
{
	// If a viewport export has been requested, do so now
	if (m_exportViewportJob.state == kRenderManagerJobDispatched)
	{
		auto& job = m_exportViewportJob;
		auto perspCam = m_liveCamera.DynamicCast<Cuda::Host::PerspectiveCamera>();
		if (perspCam)
		{
			// Get the data and attributes from the camera
			std::vector<Cuda::vec4> rawData;
			Cuda::ivec2 dataDimensions;
			perspCam->GetRawAccumulationData(rawData, dataDimensions);
			const auto& params = perspCam->GetPerspectiveParams();

			// Get the export path from the job metadata
			std::string exportPath;
			if (!job.json.GetValue("path", exportPath, Json::kRequiredWarn)) { return; }

			// Write out the PNG image
			ImageIO::WriteAccumulationBufferPNG(rawData, dataDimensions, exportPath, params.displayExposure, params.displayGamma);
		}
		else
		{
			Log::Error("Error: cannot export PNG of current viewport. Set a perspective camera object as live then try again.");
		}

		job.state = kRenderManagerJobIdle;
	}

	if (m_exportGridsJob.state == kRenderManagerJobDispatched)
	{
		Cuda::LightProbeGridExportParams exportParams;
		exportParams.isArmed = true;
		exportParams.minGridValidity = 0;
		exportParams.maxGridValidity = 1;
		m_exportGridsJob.json.GetArrayValues("paths", exportParams.exportPaths, Json::kRequiredAssert);

		m_lightProbeCamera->ExportProbeGrid(exportParams);
		m_exportGridsJob.state = kRenderManagerJobCompleted;
	}

	// Are we aborting the bake job?
	if (m_bake.job.state == kRenderManagerJobAborting)
	{
		m_bake.job.state = kRenderManagerJobIdle;
		return;
	}

	// Not baking or the scene graph is dirty? We're done.
	if (!(m_bake.job.state & kRenderManagerJobActive) || m_commandSet.find(kRenderMangerUpdateParams) != m_commandSet.end()) { return; }

	// If the job has just been dispatched, do some pre-flight checks
	if (m_bake.job.state & kRenderManagerJobDispatched)
	{		
		if (m_bake.type & (kBakeTypeNoisyProbeGrid | kBakeTypeReferenceProbeGrid))
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
	if (m_bake.type & (kBakeTypeNoisyProbeGrid | kBakeTypeReferenceProbeGrid))
	{		
		constexpr float kMinSamplesForEstimate = 8.0f;
		const auto& stats = m_lightProbeCamera->PollBakeProgress();
		m_bake.progress = stats.bake.progress;

		// If the mean validity is outside the pre-set bounds, complete the job without 
		if (stats.minMaxSamples[1] > kMinSamplesForEstimate && stats.meanValidity >= 0.0f &&
			(stats.meanValidity < m_bake.probeGridExportParams.minGridValidity ||
				stats.meanValidity > m_bake.probeGridExportParams.maxGridValidity))
		{
			Log::Error("Warning: bake was aborted because the grid validity %f was outside the specified bounds [%f, %f]", 
				stats.meanValidity, m_bake.probeGridExportParams.minGridValidity, m_bake.probeGridExportParams.maxGridValidity);

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
		m_bake.progress = float(m_liveCamera->GetFrameIdx()) / float(liveCam.minMaxSamples.y * liveCam.overrides.maxDepth);
		if (m_bake.progress > 1.0f)
		{
			// Get the raw data from the camera
			std::vector<Cuda::vec4> rawData;
			Cuda::ivec2 dataDimensions;
			m_liveCamera->GetRawAccumulationData(rawData, dataDimensions);

			// Try to pull exposure and gamma values from the perspective camera
			float pngExposure = 1.0, pngGamma = 2.2f;
			auto perspCam = m_liveCamera.DynamicCast<Cuda::Host::PerspectiveCamera>();
			if (perspCam)
			{
				const auto& params = perspCam->GetPerspectiveParams();
				pngExposure = params.displayExposure;
				pngGamma = params.displayGamma;
			}

			// Write the PNG image
			ImageIO::WriteAccumulationBufferPNG(rawData, dataDimensions, m_bake.pngExportPath, pngExposure, pngGamma);

			m_bake.succeeded = true;
			m_bake.job.state = kRenderManagerJobCompleted;
		}
	}
}