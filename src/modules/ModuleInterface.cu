#pragma once

#include "ModuleInterface.cuh"
#include "io/CommandQueue.h"

namespace Enso
{
	ModuleInterface::ModuleInterface(std::shared_ptr<CommandQueue> outQueue) :
		m_frameTimes(20),
		m_mouseWheelAngle(0.0f),
		m_clientWidth(1.0f),
		m_clientHeight(1.0f),
		m_uiGraph(m_keyCodes),
		m_renderSemaphore(kRenderManagerD3DBlitFinished),
		m_outboundCmdQueue(outQueue)
	{
		m_mouse.pos = std::numeric_limits<int>::min();
		m_mouse.prevPos = std::numeric_limits<int>::min();
		m_mouse.delta = 0.0f;

		m_uiEventQueue.maxEvents = 100;
		m_uiEventQueue.autoFlushAfterEvents = -1;
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
		//#define DISABLE_EXCEPTION_HANDLING
#ifndef DISABLE_EXCEPTION_HANDLING
		try
		{
#endif
			while (m_threadSignal.load() == kRenderManagerRun)
			{
				HighResolutionTimer timer;

				if (m_inboundCmdQueue && !m_inboundCmdQueue->IsEmpty())
				{
					OnCommandsWaiting(*m_inboundCmdQueue);
				}

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

#ifndef DISABLE_EXCEPTION_HANDLING
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
#endif

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

	template<typename T>
	T ModuleInterface::PopUIEventQueue(std::deque<T>& queue)
	{
		Assert(!queue.empty());
		const T item = queue.front();
		queue.pop_front();
		return item;
	}

	template<typename T>
	void ModuleInterface::PushUIEventQueue(const int event, std::deque<T>& queue, const T& newItem)
	{
		m_controlQueueMutex.lock();

		// If the control queue hasn't been purged, just replace the most recent event
		if (m_uiEventQueue.events.size() >= m_uiEventQueue.maxEvents)
		{
			Log::Debug("Warning: UI control queue exceeded max size of %i events", m_uiEventQueue.maxEvents);

			m_uiEventQueue.events.back() = event;
			queue.back() = newItem;
		}
		// Otherwise, push the event to the deque
		else
		{
			m_uiEventQueue.events.push_back(event);
			queue.push_back(newItem);
		}

		m_controlQueueMutex.unlock();

		// If the queue is full and auto-flush is enabled, flush everything now. 
		if (m_uiEventQueue.autoFlushAfterEvents >= 0 && m_uiEventQueue.events.size() >= m_uiEventQueue.autoFlushAfterEvents)
		{
			FlushUIEventQueue();
		}
	}

	void ModuleInterface::FlushUIEventQueue()
	{
		if (m_uiEventQueue.events.empty()) { return; }

		std::lock_guard<std::mutex> lock(m_controlQueueMutex);

		// Dispatch queued events in the order that they were posted
		while (!m_uiEventQueue.events.empty())
		{
			const int event = m_uiEventQueue.events.front();
			m_uiEventQueue.events.pop_front();

			switch (event)
			{
			case kControlEventKeyboard:
			{
				const auto keyButton = PopUIEventQueue(m_uiEventQueue.keyButton);

				m_keyCodes.Update(keyButton.first, keyButton.second);

				// Notify the superclass that a key state has changed
				OnKey(keyButton.first, false, keyButton.second);

				if (keyButton.first == VK_ESCAPE)
				{
					m_uiGraph.Reset();
				}
				else
				{
					m_uiGraph.OnTriggerTransition(kUITriggerOnKeyboard);
				}
			}
			break;

			case kControlEventMouseMove:
			{
				const auto mousePos = PopUIEventQueue(m_uiEventQueue.mouseMove);

				// Update the mouse position information
				m_mouse.prevPos = (m_mouse.pos.x == std::numeric_limits<int>::min()) ? ivec2(mousePos.x, m_clientHeight - 1 - mousePos.y) : m_mouse.pos;
				m_mouse.pos = ivec2(mousePos.x, m_clientHeight - 1 - mousePos.y);
				m_mouse.delta = m_mouse.pos - m_mouse.prevPos;

				// Notify the superclass that a mouse state has changed
				OnMouseMove();

				m_keyCodes.Update();
				m_uiGraph.OnTriggerTransition(kUITriggerOnMouseMove);
			}
			break;

			case kControlEventMouseButton:
			{
				const auto mouseButton = PopUIEventQueue(m_uiEventQueue.mouseButton);

				// Notify the superclass that a mouse state has changed
				OnMouseButton(mouseButton.first, mouseButton.second);

				// TODO: Calling Update() here feels messy and brittle. Should the UI graph have ownership of the codes?
				m_keyCodes.Update(mouseButton.first, mouseButton.second);
				m_uiGraph.OnTriggerTransition(kUITriggerOnMouseButton);
			}
			break;

			case kControlEventMouseWheel:
			{
				m_mouseWheelAngle = PopUIEventQueue(m_uiEventQueue.mouseWheel);

				// Notify the superclass that a mouse wheel state has changed
				OnMouseWheel();

				m_keyCodes.Update();
				m_uiGraph.OnTriggerTransition(kUITriggerOnMouseWheel);
			}
			break;

			default:
				AssertMsgFmt(false, "Unrecognised UI control event %i", event);
			}
		}

		// Sanity check
		Assert(m_uiEventQueue.keyButton.empty());
		Assert(m_uiEventQueue.mouseButton.empty());
		Assert(m_uiEventQueue.mouseMove.empty());
		Assert(m_uiEventQueue.mouseWheel.empty());

		/*m_uiEventQueue.keyButton.clear();
		m_uiEventQueue.mouseButton.clear();
		m_uiEventQueue.mouseMove.clear();
		m_uiEventQueue.mouseWheel.clear();*/
	}

	void ModuleInterface::SetKey(const uint code, const bool isSysKey, const bool isDown)
	{
		PushUIEventQueue(kControlEventKeyboard, m_uiEventQueue.keyButton, std::make_pair(code, isDown));
	}

	void ModuleInterface::SetMouseButton(const uint code, const bool isDown)
	{
		PushUIEventQueue(kControlEventMouseButton, m_uiEventQueue.mouseButton, std::make_pair(code, isDown));
	}

	void ModuleInterface::SetMousePos(const int mouseX, const int mouseY, const WPARAM flags)
	{
		PushUIEventQueue(kControlEventMouseMove, m_uiEventQueue.mouseMove, ivec2(mouseX, mouseY));
	}

	void ModuleInterface::SetMouseWheel(const float angle)
	{
		PushUIEventQueue(kControlEventMouseWheel, m_uiEventQueue.mouseWheel, angle);
	}

	void ModuleInterface::SetClientSize(const int width, const int height)
	{
		m_clientWidth = width;
		m_clientHeight = height;

		m_clientToNormMatrix = mat3::Identity();
		m_clientToNormMatrix.i00 = 1.0f / height;
		m_clientToNormMatrix.i11 = 1.0f / height;
		m_clientToNormMatrix.i02 = -0.5f * float(width) / float(height);
		m_clientToNormMatrix.i12 = -0.5f;

		OnResizeClient();
	}

	void ModuleInterface::FocusChange(const bool isSet)
	{	
		// Notify the deriving class that the focus has changed so it can do clean-up
		OnFocusChange(isSet);

		// Reset the state and UI graph
		m_keyCodes.Clear();
		m_uiGraph.Reset();
		Log::Debug(isSet ? "Focus set" : "Focus lost");
	}

	void ModuleInterface::OnCommandsWaiting(CommandQueue& inbound) 
	{ 
		inbound.Clear(); 
	}
}
