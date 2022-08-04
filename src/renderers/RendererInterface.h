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

enum RenderManagerMouseButtons : int
{
    kMouseLButton = 1,
    kMouseRButton = 2,
    kMouseMButton = 4,
    kMouseXButton = 8
};

class RendererInterface
{
public:    
    friend class RendererManager;

    void Initialise(const UINT clientWidth, const UINT clientHeight);
    void Start();
    void Stop();
    void Destroy(); 

    virtual bool Poll(Json::Document& stateJson);

    void SetKey(const uint code, const bool isSysKey, const bool isDown);
    void SetMouseButton(const uint code, const bool isDown);
    void SetMousePos(const int mouseX, const int mouseY, const WPARAM flags);
    void SetMouseWheel(const float angle);
    void SetClientSize(const int width, const int height);
   
    virtual std::string GetRendererName() const = 0;
    virtual const std::string& GetRendererDescription() const { return ""; }

    void SetCudaObjects(Cuda::AssetHandle<Cuda::Host::ImageRGBA>& compositeImage, cudaStream_t renderStream);

protected:
    RendererInterface();
    virtual ~RendererInterface();

    virtual void OnMouseMove() {}
    virtual void OnMouseButton(const uint code, const bool isDown) {}
    virtual void OnMouseWheel() {}
    virtual void OnKey(const uint code, const bool isSysKey, const bool isDown) {}
    virtual void OnResizeClient() {}

    virtual void OnInitialise() {}
    virtual void OnDestroy() {};
    virtual void OnPreRender() {};
    virtual void OnRender() {};
    virtual void OnPostRender() {};

    bool IsKeyDown(const uint code) const;
    bool IsSysKeyDown(const uint code) const;
    bool IsMouseButtonDown(const uint code) const;

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
     
    struct
    {
        Cuda::ivec2                                 pos;
        Cuda::ivec2                                 prevPos;
        Cuda::ivec2                                 delta;
    }
    m_mouse;


    uint                                            m_mouseButtons;
    float                                           m_mouseWheelAngle;
    uint                                            m_keyCodes[4];
    uint                                            m_sysKeyCodes[4];
    int                                             m_clientWidth;
    int                                             m_clientHeight;

    Cuda::mat3                                      m_clientToNormMatrix;

    std::mutex		                                m_jsonInputMutex;
    std::mutex			                            m_jsonOutputMutex;
    std::mutex                                      m_resourceMutex;

    std::atomic<int>                                m_dirtyFlags;
};