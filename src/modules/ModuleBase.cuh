#pragma once

#include "core/utils/Semaphore.h"
#include "CudaObjectManager.cuh"
#include "core/ui/StateGraph.h"
#include <mutex>
#include <atomic>
#include <thread>

#include "win/D3DHeaders.h"
#include "io/json/JsonUtils.h"
#include "core/assets/DirtinessGraph.cuh"
#include "core/ui/VirtualKeyStates.h"

namespace Enso
{
    enum RenderManagerRenderState : int
    {
        kRenderManagerUndefined,
        kRenderManagerIdle,
        kRenderManagerRun,
        kRenderManagerHalt,
        kRenderManagerError
    };

    enum RenderManagerSemaphore : uint
    {
        kRenderManagerCompInProgress = 0,
        kRenderManagerCompFinished = 1,
        kRenderManagerD3DBlitInProgress = 2,
        kRenderManagerD3DBlitFinished = 3
    };

    class CommandQueue;

    class ModuleBase : public Host::Dirtyable
    {
    public:
        friend class RendererManager;

        void Initialise(const UINT clientWidth, const UINT clientHeight, HWND hWnd);
        void Start();
        void Stop();
        void Destroy();

        virtual bool Poll(Json::Document& stateJson);

        void SetKey(const uint code, const bool isSysKey, const bool isDown);
        void SetMouseButton(const uint code, const bool isDown);
        void SetMousePos(const int mouseX, const int mouseY, const WPARAM flags);
        void SetMouseWheel(const float angle);
        void SetClientSize(const int width, const int height);
        void FocusChange(const bool isSet);

        virtual std::string GetRendererName() const = 0;
        virtual const std::string& GetRendererDescription() const { static std::string defaultDesc;  return defaultDesc; }

        Semaphore& GetRenderSemaphore() { return m_renderSemaphore; }

        void SetCudaObjects(AssetHandle<Host::ImageRGBA>& compositeImage, cudaStream_t renderStream);

        virtual bool Serialise(Json::Document& json, const int flags) = 0;
        void SetInboundCommandQueue(std::shared_ptr<CommandQueue> inQueue) { m_inboundCmdQueue = inQueue; }

    protected:
        ModuleBase(const Host::Asset::InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue);
        virtual ~ModuleBase();

        virtual void OnMouseMove() {}
        virtual void OnMouseButton(const uint code, const bool isDown) {}
        virtual void OnMouseWheel() {}
        virtual void OnKey(const uint code, const bool isSysKey, const bool isDown) {}
        virtual void OnResizeClient() {}
        virtual void OnFocusChange(const bool isSet) {}

        virtual void OnInitialise() {}
        virtual void OnDestroy() {};
        virtual void OnPreRender() {};
        virtual void OnRender() {};
        virtual void OnPostRender() {};

        virtual void OnCommandsWaiting(CommandQueue& inbound);

        inline bool IsKeyDown(const uint code) const { return m_keyCodes.GetState(code) & (kButtonDown | kOnButtonDepressed); }

        void        FlushUIEventQueue();

        // Returns the down status of the three mouse buttons as a bitmask
        inline uint GetMouseButtonDown() const
        {
            return (uint(IsDownImpl(KEY_LBUTTON)) * KEY_LBUTTON) |
                   (uint(IsDownImpl(KEY_MBUTTON)) * KEY_MBUTTON) |
                   (uint(IsDownImpl(KEY_RBUTTON)) * KEY_RBUTTON);
        }

        inline bool IsMouseButtonDown(const uint code) const { return IsDownImpl(code); }
        inline bool IsAnyMouseButtonDown() const
        {
            return GetMouseButtonDown() != 0u;
        }       

    private:
        void RunThread();

        template<typename T> T PopUIEventQueue(std::deque<T>& queue);
        template<typename T> void PushUIEventQueue(const int event, std::deque<T>& queue, const T& newItem);
        
        inline bool IsDownImpl(const uint code) const { return m_keyCodes.GetState(code) & (kButtonDown | kOnButtonDepressed); }

    protected:
        AssetHandle<Host::ImageRGBA>                    m_compositeImage;
        cudaStream_t                                    m_renderStream;
        HWND                                            m_parentWnd;

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
            ivec2                                       pos;
            ivec2                                       prevPos;
            ivec2                                       delta;
        }
        m_mouse;

        float                                           m_mouseWheelAngle;
        VirtualKeyMap                                   m_keyCodes;
        int                                             m_clientWidth;
        int                                             m_clientHeight;

        mat3                                            m_clientToNormMatrix;

        std::mutex		                                m_jsonInputMutex;
        std::mutex			                            m_jsonOutputMutex;
        std::mutex                                      m_commandMutex;
        std::mutex                                      m_uiEventQueueMutex;

        Semaphore                                       m_renderSemaphore;

        UIStateGraph                                    m_uiGraph;

        std::shared_ptr<CommandQueue>                   m_outboundCmdQueue;
        std::shared_ptr<CommandQueue>                   m_inboundCmdQueue;

    private:
        enum UIControlEventType : int { kControlEventUndefined = -1, kControlEventKeyboard, kControlEventMouseMove, kControlEventMouseButton, kControlEventMouseWheel };

        struct
        {
            int                                         maxEvents;
            int                                         autoFlushAfterEvents;
            std::deque<int>                             events;
            std::deque<std::pair<uint, bool>>           keyButton;
            std::deque<ivec2>                           mouseMove;
            std::deque<std::pair<uint, bool>>           mouseButton;
            std::deque<float>                           mouseWheel;
        }
        m_uiEventQueue;
    };
}