#pragma once

#include "win/D3DWindowInterface.h"
#include "win/DXSampleHelper.h"

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"

#include "renderers/RendererManager.h"

//#include "tools/MemoryMonitor.h"

#include "generic/HighResolutionTimer.h"

namespace Gui
{
    template<typename T>
    inline void SafeRelease(ComPtr<T>& resource)
    {
        if (resource)
        {
            resource.Reset();
        }
    }

    template<typename T>
    inline void SafeRelease(T*& resource)
    {
        if (resource)
        {
            resource->Release();
            resource = nullptr;
        }
    }
    
    class ComponentManager
    {
    public:
        ComponentManager(HWND hWnd, std::shared_ptr<RendererManager>& rendererManager);
        ~ComponentManager();

        void                            CreateD3DDeviceObjects(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames);
        void                            Construct();
        void                            PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx);
        void                            Destroy();

    private:
        void                            ConstructConsole();

        void                            PollRenderer();

    private:
        ComPtr<ID3D12DescriptorHeap>            m_srvHeap;
        HWND                                    m_hWnd;
        std::shared_ptr<RendererManager>        m_rendererManager;
        std::shared_ptr<RendererInterface>      m_activeRenderer;
        //MemoryMonitor                         m_memoryMonitor;

        bool                                    m_showConsole;

        int                                     m_frameIdx;
        float                                   m_meanFrameTime;

        Json::Document                          m_renderStateJson;
        Json::Document                          m_commandQueue;
        int                                     m_renderState;
        std::string                             m_renderStateFmt;
    };
}