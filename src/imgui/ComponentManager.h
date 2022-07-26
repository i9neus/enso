#pragma once

#include "win/D3DWindowInterface.h"
#include "win/DXSampleHelper.h"

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"
#include "manager/RenderManager.h"

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
        ComponentManager();

        void                            Initialise(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames);
        void                            Build(HWND hWnd);
        void                            Rebuild();
        void                            Construct();
        void                            PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx);
        void                            Destroy();
        void                            CreateDeviceObjects();

    private:
        void                            ConstructConsole();

    private:
        ComPtr<ID3D12DescriptorHeap>    m_srvHeap;
        HWND                            m_hWnd;
        //MemoryMonitor                   m_memoryMonitor;
    };
}