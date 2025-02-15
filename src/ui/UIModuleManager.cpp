#include "UIModuleManager.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

#include "modules/gi2d/GI2DModule.h"
#include "modules/ModuleManager.cuh"

namespace Enso
{
    UIModuleManager::UIModuleManager(HWND hwnd, std::shared_ptr<ModuleManager>& moduleManager) :
        m_showConsole(false),
        m_outboundCmdQueue(std::make_shared<CommandQueue>())
    {       
        m_hWnd = hwnd;
        m_moduleManager = moduleManager;
        m_activeRenderer = m_moduleManager->GetRendererPtr();

        m_gi2DUI = std::make_unique<GI2DUI>(m_outboundCmdQueue);
    }

    UIModuleManager::~UIModuleManager()
    {
        Destroy();

        m_gi2DUI.reset();
    }

    void UIModuleManager::CreateD3DDeviceObjects(ComPtr<ID3D12RootSignature>& rootSignature, ComPtr<ID3D12Device>& device, const int numConcurrentFrames)
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

        Log::Success("IMGUI successfully initialised!\n");
    }

    void UIModuleManager::Destroy()
    {
        ImGui::ImplDX12_Shutdown();

        ReleaseResource(m_srvHeap);

        Log::Write("Destroyed IMGUI D3D objects.\n");
    }

    void UIModuleManager::ConstructConsole()
    {
        ImGui::Begin("Console");

        if (m_inboundCmdQueue)
        {
            m_renderStateFmt = m_inboundCmdQueue->Format();
        }
        ImGui::TextWrapped(m_renderStateFmt.c_str());

        ImGui::End();
    }

    void UIModuleManager::Construct()
    {
        // Poll the renderer for data to populate the UI
        //PollRenderer();

        // Start the Dear ImGui frame
        ImGui::ImplDX12_NewFrame();
        ImGui::ImplWin32_NewFrame();
        ImGui::NewFrame();

        const auto kBaseSize = ImGui::CalcTextSize("A");
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, kBaseSize.x * 2.0f);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.3f));

        if (m_showConsole) { ConstructConsole(); }       

        // Construct the active component
        m_gi2DUI->ConstructComponent();

        // Menu Bar
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("Renderer"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Window"))
            {
                //ImGui::MenuItem("Combinatorics", NULL, &m_showCombinatorics);
                //ImGui::MenuItem("Render objects", NULL, &m_showRenderObjects);
                //ImGui::Separator();
                ImGui::MenuItem("Console", NULL, &m_showConsole);
                //ImGui::MenuItem("Memory monitor", NULL, &m_showMemoryMonitor);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        // Rendering
        ImGui::Render();

        // Dispatch any JSON commands that may have been generated
        //DispatchCommands();
    }

    void UIModuleManager::PopulateCommandList(ComPtr<ID3D12GraphicsCommandList>& commandList, const int frameIdx)
    {
        Construct();

        auto drawData = ImGui::GetDrawData();
        if (!drawData) { return; }

        ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap.Get() };
        commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

        ImGui::ImplDX12_RenderDrawData(drawData, commandList.Get());
    }

    void UIModuleManager::SetInboundCommandQueue(std::shared_ptr<CommandQueue> inbound) 
    { 
        m_inboundCmdQueue = inbound;
        m_gi2DUI->SetInboundCommandQueue(m_inboundCmdQueue);
    }

}