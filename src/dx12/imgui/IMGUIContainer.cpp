#include "IMGUIContainer.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

IMGUIContainer::IMGUIContainer(RenderManager& cudaRenderer) : 
    m_cudaRenderer(cudaRenderer)
{
    auto& params = m_parameters[1];
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

void IMGUIContainer::Destroy()
{
    ImGui::ImplDX12_Shutdown();
    
    SafeRelease(m_srvHeap);

    Log::Write("Destroyed IMGUI D3D objects.\n");
}

void IMGUIContainer::ConstructQuadLightControls(Cuda::QuadLightParams& params)
{
    if (!ImGui::CollapsingHeader("Quad light", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::InputFloat3("Position", &params.position[0]);
    ImGui::InputFloat3("Orientation", &params.orientation[0]);
    ImGui::InputFloat3("Scale", &params.scale[0]);

    ImGui::ColorEdit3("Colour", &params.colour[0]);
    ImGui::SliderFloat("Intensity", &params.intensity, -10.0f, 10.0f);
}

void IMGUIContainer::ConstructCameraControls(Cuda::PerspectiveCameraParams& params)
{
    if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) { return; }
   
    ImGui::InputFloat3("Position", &params.position[0]);
    ImGui::InputFloat3("Look at", &params.lookAt[0]);
    
    ImGui::SliderFloat("F-stop", &params.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &params.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &params.focalPlane, 0.0f, 2.0f);
}

void IMGUIContainer::ConstructKIFSControls(Cuda::KIFSParams& params)
{
    if (!ImGui::CollapsingHeader("KIFS", ImGuiTreeNodeFlags_DefaultOpen)) { return; }
     
    ImGui::SliderFloat("Rotate X", &params.rotate.x, 0.0f, 1.0f); 
    ImGui::SliderFloat("Rotate Y", &params.rotate.y, 0.0f, 1.0f);
    ImGui::SliderFloat("Rotate Z", &params.rotate.z, 0.0f, 1.0f);

    ImGui::SliderFloat("Scale A", &params.scale.x, 0.0f, 1.0f);
    ImGui::SliderFloat("Scale B", &params.scale.y, 0.0f, 1.0f);

    ImGui::SliderFloat("Crust thickness", &params.crustThickness, 0.0f, 1.0f);
    ImGui::SliderFloat("Vertex scale", &params.vertScale, 0.0f, 1.0f);

    ImGui::SliderInt("Iterations ", &params.numIterations, 0, kSDFMaxIterations);
}

void IMGUIContainer::ConstructMaterialControls(Cuda::SimpleMaterialParams& params)
{
    if (!ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::ColorEdit3("Albedo", (float*)&params.albedo); 
    ImGui::ColorEdit3("Incandescence", (float*)&params.incandescence); 
}

void IMGUIContainer::UpdateParameters()
{
    // If nothing's changed this frame, don't bother updating
    if (m_parameters[1] == m_parameters[0])
    {
        m_parameters[0] = m_parameters[1];
        return;
    }

    const auto& params = m_parameters[1];
   
    // Construct a JSON dictionary with the new settings
    Json::Document document;
    {
        Json::Node childNode = document.AddChildObject("material");
        params.material.ToJson(childNode);
    }
    {
        Json::Node childNode = document.AddChildObject("kifs");
        params.kifs.ToJson(childNode);
    }
    {
        Json::Node childNode = document.AddChildObject("perspectiveCamera");
        params.perspectiveCamera.ToJson(childNode);
    }
    {
        Json::Node childNode = document.AddChildObject("quadLight");
        params.quadLight.ToJson(childNode);
    }

    m_cudaRenderer.OnJson(document);

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

    ConstructCameraControls(params.perspectiveCamera);
    ConstructKIFSControls(params.kifs);
    ConstructMaterialControls(params.material);
    ConstructQuadLightControls(params.quadLight);

    float renderFrameTime = -1.0f, renderMeanFrameTime = -1.0f;
    m_cudaRenderer.GetRenderStats([&](const Json::Document& node)
        {
            node.GetValue("frameTime", renderFrameTime, false);
            node.GetValue("meanFrameTime", renderMeanFrameTime, false);
        });

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Render %.3f ms/frame (%.1f FPS)", 1000.0f * renderMeanFrameTime, 1.0f / renderMeanFrameTime);
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
