#include "IMGUIContainer.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

IMGUIContainer::IMGUIContainer()
{
    m_parameters[1].scale = 1.0f;
    m_parameters[1].colour = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
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

void IMGUIContainer::UpdateParameters(RenderManager& manager)
{
    // If nothing's changed this frame, don't bother updating
    if (m_parameters[1] == m_parameters[0])
    {
        m_parameters[0] = m_parameters[1];
        return;
    }

    const auto& p = m_parameters[1];
   
    Json::Document document;
    document.AddValue("scale", p.scale);
    document.AddArray("albedo", std::vector<float>( {p.colour.x, p.colour.y, p.colour.z} ));

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

    ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    ImGui::SliderFloat("Scale", &params.scale, 0.0f, 2.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::ColorEdit3("Colour", (float*)&params.colour); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);

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
