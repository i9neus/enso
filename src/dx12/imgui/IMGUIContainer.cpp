#include "IMGUIContainer.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

IMGUIContainer::IMGUIContainer(RenderManager& cudaRenderer) : 
    m_cudaRenderer(cudaRenderer)
{

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

    Log::Write("IMGUI successfully initialised!\n");
}

void IMGUIContainer::Build()
{
    Log::Indent indent("Building IMGUI components...\n", "Done!\n");
    m_shelves.clear();

    const Json::Document& json = m_cudaRenderer.GetSceneJSON();
    const Cuda::AssetHandle<Cuda::RenderObjectContainer> renderObjects = m_cudaRenderer.GetRenderObjectContainer();
    
    IMGUIShelfFactory shelfFactory;
    m_shelves = shelfFactory.Instantiate(json, *renderObjects);
}

void IMGUIContainer::Destroy()
{
    ImGui::ImplDX12_Shutdown();
    
    SafeRelease(m_srvHeap);

    Log::Write("Destroyed IMGUI D3D objects.\n");
}

void IMGUIContainer::UpdateParameters()
{
    for (const auto& shelf : m_shelves)
    {        
        std::string newJson;
        if (!shelf->Update(newJson)) { continue; }

        m_cudaRenderer.OnJson(shelf->GetDAGPath(), newJson);

        Log::Debug("Updated! %s, %s\n", shelf->GetID(), newJson);
        return;
    }
}

void IMGUIContainer::Render()
{
    // Start the Dear ImGui frame
    ImGui::ImplDX12_NewFrame();
    ImGui::ImplWin32_NewFrame();

    ImGui::NewFrame();

    ImGui::Begin("Generator"); 

    for (const auto& shelf : m_shelves)
    {
        ImGui::PushID(shelf->GetID().c_str());        
        shelf->Construct();
        ImGui::Separator();
        ImGui::PopID();
    }

    float renderFrameTime = -1.0f, renderMeanFrameTime = -1.0f;
    int deadRays = -1;
    m_cudaRenderer.GetRenderStats([&](const Json::Document& node)
        {
            node.GetValue("frameTime", renderFrameTime, Json::kSilent);
            node.GetValue("meanFrameTime", renderMeanFrameTime, Json::kSilent);
            node.GetValue("deadRays", deadRays, Json::kSilent);
        });

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Render %.3f ms/frame (%.1f FPS)", 1000.0f * renderMeanFrameTime, 1.0f / renderMeanFrameTime);
    ImGui::Text("Dead rays: %i", deadRays);

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
