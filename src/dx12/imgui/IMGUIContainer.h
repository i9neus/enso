#pragma once

#include "../D3DWindowInterface.h"
#include "../DXSampleHelper.h"

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"
#include "manager/RenderManager.h"

#include "shelves/IMGUIAbstractShelf.h"
#include "shelves/IMGUIShelfFactory.h"
#include "permutor/IMGUIStateManager.h"

#include "generic/HighResolutionTimer.h"

using namespace Cuda;

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

class IMGUIContainer
{
private:
    ComPtr<ID3D12DescriptorHeap>    m_srvHeap;
    HWND                            m_hWnd;
    RenderManager&                  m_cudaRenderer;
    IMGUIAbstractShelfMap           m_shelves;
    RenderObjectStateManager        m_stateManager;

    HighResolutionTimer             m_statsTimer;
    int                             m_frameIdx;
    float                           m_meanFrameTime;

public:
    IMGUIContainer(RenderManager& cudaRenderer);

    void Initialise(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames);
    void Build(HWND hWnd);
    void Rebuild();
    void Render();
    void PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx);
    void Destroy();
    void UpdateParameters();

    void ConstructRenderObjectShelves();

};