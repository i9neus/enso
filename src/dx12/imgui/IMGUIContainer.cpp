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
    m_stateManager(m_shelves, cudaRenderer, m_renderStateJson, m_commandQueue),
    m_memoryMonitor(m_renderStateJson, m_commandQueue),
    m_frameIdx(0),
    m_meanFrameTime(-1.0f),
    m_showCombinatorics(true),
    m_showRenderObjects(true),
    m_showConsole(false)
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

void IMGUIContainer::DispatchRenderCommands()
{
    std::unique_ptr<Json::Document> patchJson;
    Json::Document dispatchJson;
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
        // TODO: Don't use deep copy here.
        auto patchNode = dispatchJson.AddChildObject("patches");
        patchNode.DeepCopy(*patchJson);
    }

    // Copy any commands that may have been emitted by the GUI handlers
    if (m_commandQueue.NumMembers() > 0)
    {
        auto commandNode = dispatchJson.AddChildObject("commands");
        commandNode.DeepCopy(m_commandQueue);
        m_commandQueue.Clear();
    }

    // If there are commands waiting to be dispatched, push them to the renderer now.
    if (dispatchJson.NumMembers() > 0)
    {
        m_cudaRenderer.Dispatch(dispatchJson);
    }
}

void IMGUIContainer::ConstructConsole()
{
    ImGui::Begin("Console");

    m_renderStateFmt = m_renderStateJson.Stringify(true);
    ImGui::TextWrapped(m_renderStateFmt.c_str());

    ImGui::End();
}

void IMGUIContainer::ConstructRenderObjectShelves()
{
    ImGui::Begin("Render Objects", &m_showRenderObjects);

    // Only poll the render object manager occasionally
    if (m_renderStatsTimer.Get() > 0.5f)
    {
        // If we're waiting on a previous stats job, don't dispatch a new one
        if (!m_renderStateJson.GetChildObject("jobs/getRenderStats", Json::kSilent))
        {
            m_commandQueue.AddChildObject("getRenderStats");
            m_renderStatsTimer.Reset();
        }
    }

    const Json::Node statsJson = m_renderStateJson.GetChildObject("jobs/getRenderStats", Json::kSilent);
    if(statsJson)
    {
        int statsState;
        statsJson.GetValue("state", statsState, Json::kRequiredAssert);

        // If the stats gathering task has finished, it'll be accompanied by data for each render object that emits it
        if (statsState == kRenderManagerJobCompleted)
        {
            const Json::Node objectsJson = statsJson.GetChildObject("renderObjects", Json::kSilent);
            if (objectsJson)
            {
                for (const auto& shelf : m_shelves)
                {
                    // Look to see if there are statistics associated with this shelf
                    const Json::Node objectNode = objectsJson.GetChildObject(shelf.first, Json::kSilent | Json::kLiteralID);
                    if (objectNode)
                    {
                        Assert(shelf.second);
                        shelf.second->OnUpdateRenderObjectStatistics(objectNode, m_renderStateJson);
                    }
                }
            }
        }
    }

    // Emit some statistics about the render
    ImGui::Text(tfm::format("Frame index: %i", m_frameIdx).c_str());
    if (m_meanFrameTime > 0.0f)
    {
        ImGui::Text(tfm::format("%.2f FPS (%.2fms per frame)", 1.0f / m_meanFrameTime, m_meanFrameTime * 1e3f).c_str());
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

void IMGUIContainer::PollCudaRenderState()
{
    Assert(m_cudaRenderer.PollRenderState(m_renderStateJson));

    const Json::Node managerJson = m_renderStateJson.GetChildObject("renderManager", Json::kSilent | Json::kLiteralID);
    if (managerJson)
    {
        managerJson.GetValue("frameIdx", m_frameIdx, Json::kRequiredAssert);
        managerJson.GetValue("meanFrameTime", m_meanFrameTime, Json::kRequiredAssert);
        managerJson.GetValue("rendererStatus", m_renderState, Json::kRequiredAssert);
    }
}

void IMGUIContainer::Render()
{
    // Poll the CUDA renderer to get the latest stats and data
    PollCudaRenderState();
    
    if (m_renderState == kRenderManagerRun && m_stateManager.GetDirtiness() == IMGUIDirtiness::kSceneReload)
    {
        Rebuild();
    }
    m_stateManager.MakeClean();
    
    // Start the Dear ImGui frame
    ImGui::ImplDX12_NewFrame();
    ImGui::ImplWin32_NewFrame();
    ImGui::NewFrame();

    const auto kBaseSize = ImGui::CalcTextSize("A");
    ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, kBaseSize.x * 2.0f);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.3f));

    // Menu Bar
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Window"))
        {
            ImGui::MenuItem("Combinatorics", NULL, &m_showCombinatorics);
            ImGui::MenuItem("Render objects", NULL, &m_showRenderObjects);
            ImGui::Separator();
            ImGui::MenuItem("Console", NULL, &m_showConsole);
            ImGui::MenuItem("Memory monitor", NULL, &m_showMemoryMonitor);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (m_renderState == kRenderManagerRun)
    {
        // Construct the state manager UI
        if (m_showCombinatorics) { m_stateManager.ConstructUI(); }

        // Construct the render objects shelves
        if (m_showRenderObjects) { ConstructRenderObjectShelves(); }

        // Handle the bake iteration 
        m_stateManager.HandleBakeIteration();
    }

    if (m_showConsole) { ConstructConsole(); }

    if (m_showMemoryMonitor) { m_memoryMonitor.ConstructUI(); }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    // Rendering
    ImGui::Render();

    // Dispatch any commands that may have been generated
    DispatchRenderCommands();
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