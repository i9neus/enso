#pragma once

#include "D3DWindowInterface.h"
#include "DXSampleHelper.h"

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"

#include "thirdparty/imgui/imgui.h"

template<typename T>
inline void SafeRelease(ComPtr<T>& resource)
{
    if (resource) { resource->Release(); }
    resource.Reset();
}

template<typename T>
inline void SafeRelease(T*& resource)
{
    if (resource) { resource->Release(); }
    resource = nullptr;
}

class IMGUIContainer
{
private:
    struct FrameResources
    {
        ID3D12Resource*     IndexBuffer;
        ID3D12Resource*     VertexBuffer;
        int                 IndexBufferSize;
        int                 VertexBufferSize;
    };

    struct VERTEX_CONSTANT_BUFFER
    {
        float   mvp[4][4];
    };

private:
    ComPtr<ID3D12PipelineState>  m_pipelineState;
    ComPtr<ID3D12RootSignature>  m_rootSignature;
    ComPtr<ID3D12Device>         m_device;

    std::vector<FrameResources> m_frameResources;
    int                         m_frameIdx;
    int                         m_numFramesInFlight;

    void InitialiseShaders();
    void SetupRenderState(ImDrawData* draw_data, ComPtr<ID3D12GraphicsCommandList>& comList, FrameResources& fr);

public:
    IMGUIContainer() = default;

    void Initialise(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device);
    void Render(ImDrawData* draw_data, ComPtr<ID3D12GraphicsCommandList>& commandList);
    void Destroy();
};