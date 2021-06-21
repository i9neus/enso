#include "IMGUIContainer.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

IMGUIContainer::IMGUIContainer()
{
    auto& params = m_parameters[1];
    
    params.kifs.rotate = vec3(0.0f);
    params.kifs.faceMask = 0xffffffffu;
    params.kifs.scale = vec2(0.0f, 0.0f);
    params.kifs.thickness = 0.5f;
    params.kifs.iterations = 1;
    
    params.material.colour = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
}

void IMGUIContainer::Initialise(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames)
{
    Assert(rootSignature);
    Assert(device);

    // Describe and create a shader resource view (SRV) heap for the texture.
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = 1;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvHeap)));

    Assert(ImGui::ImplDX12_Init(device.Get(), numConcurrentFrames, DXGI_FORMAT_R8G8B8A8_UNORM, m_srvHeap.Get(),
        m_srvHeap->GetCPUDescriptorHandleForHeapStart(), m_srvHeap->GetGPUDescriptorHandleForHeapStart()));

    Assert(ImGui::ImplDX12_CreateDeviceObjects());

    std::printf("IMGUI successfully initialised!\n");
}

void IMGUIContainer::Destroy()
{
    ImGui::ImplDX12_Shutdown();
    
    SafeRelease(m_srvHeap);

    std::printf("Destroyed IMGUI D3D objects.\n");
}

void IMGUIContainer::ConstructCameraControls()
{
    if (!ImGui::CollapsingHeader("Camera")) { return; }
   
}

void IMGUIContainer::ConstructKIFSControls(Cuda::Device::KIFS::Params& params)
{
    if (!ImGui::CollapsingHeader("KIFS")) { return; }
     
    ImGui::SliderFloat("Rotate X", &params.rotate.x, -1.0f, 1.0f); 
    ImGui::SliderFloat("Rotate Y", &params.rotate.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Rotate Z", &params.rotate.z, -1.0f, 1.0f);

    ImGui::SliderFloat("Scale A", &params.scale.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Scale B", &params.scale.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Isosurface thickness", &params.thickness, 0.0f, 1.0f);

    ImGui::SliderInt("Iterations ", &params.iterations, 0, kSDFMaxIterations);
}

void IMGUIContainer::ConstructMaterialControls(Parameters& params)
{
    if (!ImGui::CollapsingHeader("Material")) { return; }

    ImGui::ColorEdit3("Colour", (float*)&params.material.colour); // Edit 3 floats representing a color

}

void IMGUIContainer::UpdateParameters(RenderManager& manager)
{
    // If nothing's changed this frame, don't bother updating
    if (m_parameters[1] == m_parameters[0])
    {
        m_parameters[0] = m_parameters[1];
        return;
    }

    const auto& params = m_parameters[1];
   
    Json::Document document;
    document.AddArray("albedo", std::vector<float>( {params.material.colour.x, params.material.colour.y, params.material.colour.z} ));

    manager.OnJson(document);

    Log::Debug("Updated!\n");

    m_parameters[0] = m_parameters[1];
}

void IMGUIContainer::Render()
{
    auto& params = m_parameters[1];
    
    // Start the Dear ImGui frame
    ImGui::ImplDX12_NewFrame();
    ImGui::ImplWin32_NewFrame();

    ImGui::NewFrame();

    ImGui::Begin("Generator");   

    ConstructCameraControls();
    ConstructKIFSControls(params.kifs);
    ConstructMaterialControls(params);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    // Rendering
    ImGui::Render();
}

void IMGUIContainer::PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx)
{    
    Render();
    
    auto drawData = ImGui::GetDrawData();
    if (!drawData) { return; }
    
    ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap.Get() };
    commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

    ImGui::ImplDX12_RenderDrawData(drawData, commandList.Get());
}
