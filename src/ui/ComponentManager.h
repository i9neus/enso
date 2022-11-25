#pragma once

//#include "tools/MemoryMonitor.h"
#include "components/ComponentInterface.h"

class RendererManager;
class RendererInterface;

namespace Gui
{
    class GI2DUI;
    
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
        void                            DispatchCommands();

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
        Json::CommandQueue                      m_commandQueue;
        int                                     m_renderState;
        std::string                             m_renderStateFmt;

        std::unique_ptr<GI2DUI>                 m_gi2DUI;
    };
}