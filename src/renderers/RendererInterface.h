#pragma once

#include "generic/StdIncludes.h"
#include "CudaObjectManager.h"

enum RenderManagerRenderState : int
{
    kRenderManagerUndefined,
    kRenderManagerIdle,
    kRenderManagerRun,
    kRenderManagerHalt,
    kRenderManagerError
};

class RendererInterface
{
public:    
    virtual void Initialise() = 0;
    virtual void Start();
    virtual void Stop();
    virtual void Destroy();

    virtual void OnResizeClient() = 0;
   
    virtual std::string GetRendererName() const = 0;
    virtual const std::string& GetRendererDescription() const { return ""; }

    void SetCudaObjects(Cuda::AssetHandle<Cuda::Host::ImageRGBA>& compositeImage, cudaStream_t renderStream);

protected:
    RendererInterface();
    virtual ~RendererInterface();

    virtual void OnDestroy() {};
    virtual void OnPreRender() {};
    virtual void OnRender() {};
    virtual void OnPostRender() {};

private:
    void RunThread();

protected:
    Cuda::AssetHandle<Cuda::Host::ImageRGBA>        m_compositeImage;
    cudaStream_t                                    m_renderStream;

    std::atomic<int>	                            m_threadSignal;
    std::thread			                            m_managerThread;

    using                                           TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TimePoint					                    m_renderStartTime;
    std::vector<float>		                        m_frameTimes;
    float                                           m_meanFrameTime;
};