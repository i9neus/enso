#pragma once

#include "ModuleInterface.h"

namespace Enso
{
	ModuleInterface::ModuleInterface() :
		m_frameTimes(20),
		m_mouseWheelAngle(0.0f),
		m_clientWidth(1.0f),
		m_clientHeight(1.0f),
		m_dirtyFlags(0),
		m_uiGraph(m_keyCodes),
		m_renderSemaphore(kRenderManagerD3DBlitFinished)
	{
		m_mouse.pos = std::numeric_limits<int>::min();
		m_mouse.prevPos = std::numeric_limits<int>::min();
		m_mouse.delta = 0.0f;
	}

	ModuleInterface::~ModuleInterface()
	{

	}

	void ModuleInterface::SetCudaObjects(AssetHandle<Host::ImageRGBA>& compositeImage, cudaStream_t renderStream)
	{
		m_compositeImage = compositeImage;
		m_renderStream = renderStream;
	}

	void ModuleInterface::Initialise(const UINT clientWidth, const UINT clientHeight)
	{
		SetClientSize(clientWidth, clientHeight);

		OnInitialise();
	}

	void ModuleInterface::Destroy()
	{
		// Stop and clean up the renderer object
		Stop();
		OnDestroy();
	}

	void ModuleInterface::Start()
	{
		Log::Write("Starting %s...\b", GetRendererName());

		m_threadSignal = kRenderManagerRun;
		m_managerThread = std::thread(std::bind(&ModuleInterface::RunThread, this));

		m_renderStartTime = std::chrono::high_resolution_clock::now();

		Assert(m_managerThread.joinable());

		Log::Success("Okay!");
	}

	void ModuleInterface::Stop()
	{
		if (!m_managerThread.joinable() || m_threadSignal != kRenderManagerRun) { return; }

		Log::Indent indent(tfm::format("Halting %s...\r", GetRendererName()));

		m_threadSignal.store(kRenderManagerHalt);
		m_managerThread.join();

		Log::Success("Successfully halted '%s'!", GetRendererName());
	}

	void ModuleInterface::RunThread()
	{
		checkCudaErrors(cudaStreamSynchronize(m_renderStream));

		// Notify the inheriting class that the render is about to start
		{
			//std::lock_guard<std::mutex> lock(m_resourceMutex);
			OnPreRender();
		}

		m_frameIdx = 0;
		try
		{
			while (m_threadSignal.load() == kRenderManagerRun)
			{
				HighResolutionTimer timer;

				// Notify that a render "tick" has begun
				{
					//std::lock_guard<std::mutex> lock(m_resourceMutex);
					OnRender();
				}

				// Compute some stats on the framerate
				m_frameIdx++;
				m_lastFrameTime = timer.Get();
				m_frameTimes[m_frameIdx % m_frameTimes.size()] = m_lastFrameTime;
				m_meanFrameTime = 0.0f;
				for (const auto& ft : m_frameTimes)
				{
					m_meanFrameTime += ft;
				}
				m_meanFrameTime /= min(m_frameIdx, int(m_frameTimes.size()));
			}
		}
		catch (const std::runtime_error& err)
		{
			Log::Error("Runtime error: %s\n", err.what());
			StackBacktrace::Print();
		}
		catch (...)
		{
			Log::Error("Unhandled error");
			StackBacktrace::Print();
		}

		// Notify that the render has completed
		{
			//std::lock_guard<std::mutex> lock(m_resourceMutex);
			OnPostRender();
		}

		// Signal that the renderer has finished
		m_threadSignal.store(kRenderManagerIdle);
	}

	bool ModuleInterface::Poll(Json::Document& stateJson)
	{
		stateJson.Clear();

		// Add some generic data about the renderer that's exported each time the state is polled
		Json::Node managerJson = stateJson.AddChildObject("renderer");
		managerJson.AddValue("frameIdx", m_frameIdx);
		managerJson.AddValue("smoothedFrameTime", m_meanFrameTime);
		managerJson.AddValue("smoothedFPS", 1.0f / m_meanFrameTime);
		managerJson.AddValue("lastFrameTime", m_lastFrameTime);
		managerJson.AddValue("lastFPS", 1.0f / m_lastFrameTime);
		const int threadSignal = m_threadSignal;
		managerJson.AddValue("rendererStatus", threadSignal);

		return true;
	}

	void ModuleInterface::SetKey(const uint code, const bool isSysKey, const bool isDown)
	{
		m_keyCodes.Update(code, isDown);

		if (code == VK_ESCAPE)
		{
			m_uiGraph.Reset();
		}
		else
		{
			m_uiGraph.OnTriggerTransition(kUITriggerOnKeyboard);
		}

		// Notify the superclass that a key state has changed
		OnKey(code, isSysKey, isDown);
	}

	void ModuleInterface::SetMouseButton(const uint code, const bool isDown)
	{
		// TODO: Calling Update() here feels messy and brittle. Should the UI graph have ownership of the codes?
		m_keyCodes.Update(code, isDown);
		m_uiGraph.OnTriggerTransition(kUITriggerOnMouseButton);

		// Notify the superclass that a mouse state has changed
		OnMouseButton(code, isDown);
	}

	void ModuleInterface::SetMousePos(const int mouseX, const int mouseY, const WPARAM flags)
	{
		// Update the mouse position information
		m_mouse.prevPos = (m_mouse.pos.x == std::numeric_limits<int>::min()) ? ivec2(mouseX, m_clientHeight - 1 - mouseY) : m_mouse.pos;
		m_mouse.pos = ivec2(mouseX, m_clientHeight - 1 - mouseY);
		m_mouse.delta = m_mouse.pos - m_mouse.prevPos;

		m_keyCodes.Update();
		m_uiGraph.OnTriggerTransition(kUITriggerOnMouseMove);

		// Notify the superclass that a mouse state has changed
		OnMouseMove();
	}

	void ModuleInterface::SetMouseWheel(const float angle)
	{
		m_mouseWheelAngle = angle;

		m_keyCodes.Update();
		m_uiGraph.OnTriggerTransition(kUITriggerOnMouseWheel);

		// Notify the superclass that a mouse wheel state has changed
		OnMouseWheel();
	}

	void ModuleInterface::SetClientSize(const int width, const int height)
	{
		m_clientWidth = width;
		m_clientHeight = height;

		m_clientToNormMatrix = mat3::Indentity();
		m_clientToNormMatrix.i00 = 1.0f / height;
		m_clientToNormMatrix.i11 = 1.0f / height;
		m_clientToNormMatrix.i02 = -0.5f * float(width) / float(height);
		m_clientToNormMatrix.i12 = -0.5f;

		OnResizeClient();
	}
}
