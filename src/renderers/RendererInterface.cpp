#pragma once

#include "RendererInterface.h"

RendererInterface::RendererInterface()
{

}

RendererInterface::~RendererInterface()
{

}

void RendererInterface::SetCudaObjects(Cuda::AssetHandle<Cuda::Host::ImageRGBA>& compositeImage, cudaStream_t renderStream)
{
    m_compositeImage = compositeImage;
    m_renderStream = renderStream;
}

void RendererInterface::Start()
{
	Log::Write("Starting %s...\b", GetRendererName());

	m_threadSignal = kRenderManagerRun;
	m_managerThread = std::thread(std::bind(&RendererInterface::RunThread, this));

	m_renderStartTime = std::chrono::high_resolution_clock::now();

	Assert(m_managerThread.joinable());

	Log::Success("Okay!");
}

void RendererInterface::Stop()
{ 
	if (!m_managerThread.joinable() || m_threadSignal != kRenderManagerRun)
	{
		Log::Warning("Renderer %s is not running.", GetRendererName());
		return;
	}

	Log::Write("Halting %s...\r", GetRendererName());
	m_threadSignal.store(kRenderManagerHalt);
	m_managerThread.join();

	Log::Write("Done!\n");
}

void RendererInterface::RunThread()
{
	checkCudaErrors(cudaStreamSynchronize(m_renderStream));

	PreRender();

	int frameIdx = 0;
	try
	{
		while (m_threadSignal.load() == kRenderManagerRun)
		{
			Timer timer;

			Render();

			// Compute some stats on the framerate
			frameIdx++;
			m_frameTimes[frameIdx % m_frameTimes.size()] = timer.Get();
			m_meanFrameTime = 0.0f;
			for (const auto& ft : m_frameTimes)
			{
				m_meanFrameTime += ft;
			}
			m_meanFrameTime /= math::min(frameIdx, int(m_frameTimes.size()));
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

	PostRender();

	// Signal that the renderer has finished
	m_threadSignal.store(kRenderManagerIdle);
}