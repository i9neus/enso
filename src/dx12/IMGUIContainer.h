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
    resource.Reset(nullptr);
}

class IMGUIContainer
{
public:
    IMGUIContainer() = default;

    void Initialise(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device);
    void Destroy();

private:
    ComPtr<ID3D12PipelineState>  m_pipelineState;
    ComPtr<ID3D12RootSignature>  m_rootSignature;
    ComPtr<ID3D12Device>         m_device;

};