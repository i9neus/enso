#include "IMGUIContainer.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

#include "shelves/IMGUIBxDFShelves.h"
#include "shelves/IMGUICameraShelves.h"
#include "shelves/IMGUIFilterShelves.h"
#include "shelves/IMGUIIntegratorShelves.h"
#include "shelves/IMGUILightShelves.h"
#include "shelves/IMGUIMaterialShelves.h"
#include "shelves/IMGUITracableShelves.h"

IMGUIContainer::IMGUIContainer(RenderManager& cudaRenderer) :
    m_cudaRenderer(cudaRenderer),
    m_stateManager(m_shelves, cudaRenderer),
    m_frameIdx(0),
    m_meanFrameTime(-1.0f)
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

void IMGUIContainer::Rebuild()
{
    Log::Indent indent("Building IMGUI components...\n", "Done!\n");
    m_shelves.clear();

    const Json::Document& sceneJson = m_cudaRenderer.GetSceneJSON();
    const Cuda::AssetHandle<Cuda::RenderObjectContainer> renderObjects = m_cudaRenderer.GetRenderObjectContainer();

    IMGUIShelfFactory shelfFactory;
    m_shelves = shelfFactory.Instantiate(sceneJson, *renderObjects);

    m_stateManager.Rebuild(sceneJson);
}

void IMGUIContainer::Build(HWND hwnd)
{    
    m_hWnd = hwnd;
    m_stateManager.Initialise(m_hWnd);

    Rebuild();
}

void IMGUIContainer::Destroy()
{
    ImGui::ImplDX12_Shutdown();
    
    SafeRelease(m_srvHeap);

    Log::Write("Destroyed IMGUI D3D objects.\n");
}

void IMGUIContainer::UpdateParameters()
{
    std::unique_ptr<Json::Document> patchJson;
    for (const auto& shelf : m_shelves)
    {        
        if (shelf.second->IsDirty())
        {
            if (!patchJson)
            {
                patchJson.reset(new Json::Document);
            }
            Json::Node shelfPatch = patchJson->AddChildObject(shelf.second->GetDAGPath());

            shelf.second->ToJson(shelfPatch);
            shelf.second->MakeClean();
        }
    }

    if (patchJson)
    {        
        m_cudaRenderer.OnJson(*patchJson);
    }
}

void IMGUIContainer::ConstructRenderObjectShelves()
{
    ImGui::Begin("Render Objects");

    // Only poll the render object manager occasionally
    if (m_statsTimer.Get() > 0.5f)
    {
        Json::Document statsDocument;
        m_cudaRenderer.GetRenderStats(statsDocument);

        statsDocument.GetValue("frameIdx", m_frameIdx, Json::kSilent);
        statsDocument.GetValue("meanFrameTime", m_meanFrameTime, Json::kSilent);

        for (const auto& shelf : m_shelves)
        {
            // Look to see if there are statistics associated with this shelf
            const Json::Node statsNode = statsDocument.GetChildObject(shelf.first, Json::kSilent | Json::kLiteralID);          
            if (statsNode)
            {
                Assert(shelf.second);
                shelf.second->OnUpdateRenderObjectStatistics(statsNode);
            }
        }

        m_statsTimer.Reset();
    }

    // Emit some statistics about the render
    ImGui::Text(tfm::format("Frame index: %i", m_frameIdx).c_str());
    if (m_meanFrameTime > 0.0f)
    {
        ImGui::Text(tfm::format("%.2f FPS (%.fms per frame)", 1.0f / m_meanFrameTime, m_meanFrameTime * 1e3f).c_str());
    }
    ImGui::Separator();

    // Construct the shelves
    int shelfIdx = 0;
    for (const auto& shelf : m_shelves)
    {
        UIStyle style(shelfIdx++);
        
        ImGui::PushID(shelf.second->GetID().c_str());

        shelf.second->Construct();
        ImGui::Separator();     

        ImGui::PopID();
    }    

    ImGui::End();
}

void IMGUIContainer::Render()
{
    if (m_stateManager.GetDirtiness() == IMGUIDirtiness::kSceneReload)
    {
        Rebuild();
    }
    m_stateManager.MakeClean();
    
    // Start the Dear ImGui frame
    ImGui::ImplDX12_NewFrame();
    ImGui::ImplWin32_NewFrame();

    ImGui::NewFrame();

    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.3f));

    m_stateManager.ConstructUI();

    ConstructRenderObjectShelves();

    m_stateManager.HandleBakeIteration();

    ImGui::PopStyleColor(1);

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