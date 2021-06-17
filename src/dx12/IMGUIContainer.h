#pragma once

#include "D3DWindowInterface.h"
#include "DXSampleHelper.h"

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"
#include "renderer/RenderManager.h"

#include "kernels/math/CudaMath.cuh"
#include "kernels/tracables/CudaKIFS.cuh"
#include "kernels/CudaPerspectiveCamera.cuh"

#include "thirdparty/imgui/imgui.h"

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
    ComPtr<ID3D12DescriptorHeap> m_srvHeap;
    HWND                        m_hWnd;

    struct Parameters
    {        
        Cuda::Device::PerspectiveCamera::Params     camera;
        Cuda::Device::KIFS::Params                  kifs;

        struct
        {
            ImVec4  colour;
        } 
        material;

        Parameters() { std::memset(this, 0, sizeof(Parameters)); }
        Parameters& operator=(const Parameters& other) = default;

        bool operator==(const Parameters& other) const
        {
            for (int i = 0; i < sizeof(Parameters); i++)
            {
                if (reinterpret_cast<const unsigned char*>(this)[i] != reinterpret_cast<const unsigned char*>(&other)[i]) { return false; }
            }
            return true;
        }
        inline bool operator!=(const Parameters& other) const { return operator==(other); }
    };
    
    Parameters                          m_parameters[2];

    void ConstructKIFSControls(Cuda::Device::KIFS::Params& params);
    void ConstructMaterialControls(Parameters& params);
    void ConstructCameraControls();

public:
    IMGUIContainer();

    void Initialise(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames);
    void Render();
    void PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx);
    void Destroy();
    void UpdateParameters(RenderManager& manager);

};