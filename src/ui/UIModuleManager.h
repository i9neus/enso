#pragma once

//#include "tools/MemoryMonitor.h"
#include "modules/UIModuleInterface.h"
#include "win/WindowsHeaders.h"
#include "win/D3DHeaders.h"

#include <memory>

namespace Enso
{    
    class GI2DUI;
    class ModuleManager;
    class ModuleInterface;
    
    class UIModuleManager
    {
    public:
        UIModuleManager(HWND hWnd, std::shared_ptr<ModuleManager>& moduleManager);
        ~UIModuleManager();

        void                            CreateD3DDeviceObjects(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames);
        void                            Construct();
        void                            PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx);
        void                            Destroy();

        std::shared_ptr<CommandQueue>   GetOutboundCommandQueue() { return m_outboundCmdQueue; }
        void                            SetInboundCommandQueue(std::shared_ptr<CommandQueue> inbound);

    private:
        void                            ConstructConsole();

    private:
        ComPtr<ID3D12DescriptorHeap>            m_srvHeap;
        HWND                                    m_hWnd;
        std::shared_ptr<ModuleManager>          m_moduleManager;
        std::shared_ptr<ModuleInterface>        m_activeRenderer;
        //MemoryMonitor                         m_memoryMonitor;

        bool                                    m_showConsole;

        int                                     m_frameIdx;
        float                                   m_meanFrameTime;

        Json::Document                          m_renderStateJson;
        int                                     m_renderState;
        std::string                             m_renderStateFmt;

        std::unique_ptr<GI2DUI>                 m_gi2DUI;

        std::shared_ptr<CommandQueue>           m_inboundCmdQueue;
        std::shared_ptr<CommandQueue>           m_outboundCmdQueue;
    };
}