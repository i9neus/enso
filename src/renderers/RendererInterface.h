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
    virtual bool Poll(Json::Document& stateJson);

    virtual void OnResizeClient() = 0;
    virtual void OnMouseMove(const int clientX, const int clientY) {}
    virtual void OnMouseButton(const int code, const bool isDown) {}
    virtual void OnMouseWheel(const float degrees) {}
    virtual void OnKey(const int code, const bool isDown) {}
   
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
    float                                           m_lastFrameTime;
    int                                             m_frameIdx;

    std::mutex		                                m_jsonInputMutex;
    std::mutex			                            m_jsonOutputMutex;
};