#pragma once

#include "generic/StdIncludes.h"
#include "generic/Semaphore.h"
#include "CudaObjectManager.h"
#include "UIStateGraph.h"

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

enum RenderManagerSemaphore : uint
{
    kRenderManagerCompInProgress = 0,
    kRenderManagerCompFinished = 1,
    kRenderManagerD3DBlitInProgress = 2,
    kRenderManagerD3DBlitFinished = 3
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
    
    Semaphore& GetRenderSemaphore() { return m_renderSemaphore; }

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

    inline bool IsKeyDown(const uint code) const { return m_keyCodes.GetState(code) & (kButtonDown | kOnButtonDepressed); }

    #define IsDownImpl (m_mouse.codes.GetState(code) & (kButtonDown | kOnButtonDepressed))
    inline bool IsMouseButtonDown(const uint code) const { return IsDownImpl; }
    inline bool IsAnyMouseButtonDown(const uint code) const 
    {  
        return IsDownImpl; 
    }
    template<typename... Pack>
    inline bool IsAnyMouseButtonDown(const uint code, Pack... pack) const
    {
        return IsAnyMouseButtonDown(pack...) | IsDownImpl;
    }   
    #undef IsDownImpl

    inline void SetDirtyFlags(const uint code) { m_dirtyFlags = m_dirtyFlags | code; }
    inline void ClearDirtyFlags(const uint code) { m_dirtyFlags = m_dirtyFlags & ~code; }

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
        MouseButtonMap                              codes;
    }
    m_mouse;

    float                                           m_mouseWheelAngle;
    KeyboardButtonMap                               m_keyCodes;
    int                                             m_clientWidth;
    int                                             m_clientHeight;

    Cuda::mat3                                      m_clientToNormMatrix;

    std::mutex		                                m_jsonInputMutex;
    std::mutex			                            m_jsonOutputMutex;
    std::mutex                                      m_resourceMutex;

    std::atomic<uint>                               m_dirtyFlags;
    Semaphore                                       m_renderSemaphore;

    UIStateGraph                                    m_uiGraph;
private:
    
};