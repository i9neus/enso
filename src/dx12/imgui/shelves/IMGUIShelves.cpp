#include "IMGUIShelves.h"

void SimpleMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::ColorEdit3(tfm::format("Albedo (%s)", m_id).c_str(), (float*)&p.albedo);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&p.incandescence);
    ImGui::Checkbox("Use grid", &p.useGrid);
}

void SimpleMaterialShelf::Randomise(int flags)
{
    const Cuda::vec2 randomRange = (flags & IMGUIAbstractShelf::kReset) ? Cuda::vec2(0.5f) : Cuda::vec2(0.0f, 1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void KIFSMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::SliderFloat3("HSL lower", &p.hslLower[0], 0.0f, 1.0f);
    ImGui::SliderFloat3("HSL upper", &p.hslUpper[0], 0.0f, 1.0f);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&p.incandescence);
}
void KIFSMaterialShelf::Randomise(int flags)
{
    const Cuda::vec2 randomRange = (flags & IMGUIAbstractShelf::kReset) ? Cuda::vec2(0.5f) : Cuda::vec2(0.0f, 1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void CornellMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::ColorEdit3("Albedo 1", (float*)&p.albedo[0]);
    ImGui::ColorEdit3("Albedo 2", (float*)&p.albedo[1]);
    ImGui::ColorEdit3("Albedo 3", (float*)&p.albedo[2]);
    ImGui::ColorEdit3("Albedo 4", (float*)&p.albedo[3]);
    ImGui::ColorEdit3("Albedo 5", (float*)&p.albedo[4]);
    ImGui::ColorEdit3("Albedo 6", (float*)&p.albedo[5]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void PlaneShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.tracable.transform, true);
    ImGui::Checkbox("Bounded", &p.isBounded);
}

void PlaneShelf::Randomise(int flags)
{
    const Cuda::vec2 randomRange = (flags & IMGUIAbstractShelf::kReset) ? Cuda::vec2(0.5f) : Cuda::vec2(0.0f, 1.0f);
    
    m_params[0].tracable.transform.Randomise(randomRange);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform, true);

    ImGui::Checkbox("Exclude from bake", &p.excludeFromBake);
}

void SphereShelf::Randomise(int flags)
{
    m_params[0].transform.Randomise((flags & IMGUIAbstractShelf::kReset) ? Cuda::vec2(0.5f) : Cuda::vec2(0.0f, 1.0f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void CornellBoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.tracable.transform, true);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

void QuadLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform, true);

    ImGui::ColorEdit3("Colour", &p.colour[0], ImGuiColorEditFlags_InputHSV);
    ImGui::SliderFloat("Intensity", &p.intensity, -10.0f, 10.0f);
}

void QuadLightShelf::Randomise(int flags)
{
    const Cuda::vec2 randomRange = (flags & IMGUIAbstractShelf::kReset) ? Cuda::vec2(0.5f) : Cuda::vec2(0.0f, 1.0f);
    m_params[0].transform.Randomise(randomRange);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void SphereLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform, true);

    ImGui::ColorEdit3("Colour", &p.colour[0]);
    ImGui::SliderFloat("Intensity", &p.intensity, -10.0f, 10.0f);
}

void SphereLightShelf::Randomise(int flags)
{
    m_params[0].transform.Randomise((flags & IMGUIAbstractShelf::kReset) ? Cuda::vec2(0.5f) : Cuda::vec2(0.0f, 1.0f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void EnvironmentLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::Text("[No attributes]");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void LambertBRDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::Text("[No attributes]");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void PerspectiveCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];

    ImGui::Checkbox("Active", &p.camera.isActive); SL;
    ImGui::Checkbox("Live", &p.camera.isLive);  SL;
    ImGui::Checkbox("Realtime", &p.isRealtime);
    ImGui::Checkbox("Mimic light probe", &p.mimicLightProbe);

    ImGui::DragFloat3("Position", &p.position[0], math::max(0.01f, cwiseMax(p.position) * 0.01f));
    ImGui::DragFloat3("Look at", &p.lookAt[0], math::max(0.01f, cwiseMax(p.lookAt) * 0.01f));

    ImGui::SliderFloat("F-stop", &p.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &p.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &p.focalPlane, 0.0f, 2.0f);
    ImGui::SliderFloat("Display gamma", &p.displayGamma, 0.01f, 5.0f);

    ImGui::SliderInt("Max path depth", &p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &p.camera.splatClamp, math::max(0.01f, p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
}

LightProbeCameraShelf::LightProbeCameraShelf(const Json::Node& json)
    : IMGUIShelf(json)
{
    m_swizzleLabels = { "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX" };
    m_pathData.resize(2048);
    m_pathData[0] = '\0';
}

void LightProbeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];

    ImGui::Checkbox("Active", &p.camera.isActive); SL;
    ImGui::Checkbox("Live", &p.camera.isLive);

    ConstructTransform(p.grid.transform, false);

    ImGui::InputInt3("Grid density", &p.grid.gridDensity[0]);
    ConstructComboBox("SH order", { "L0", "L1", "L2" }, p.grid.shOrder);
    ImGui::SliderInt("Max path depth", &p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &p.camera.splatClamp, math::max(0.01f, p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());

    ImGui::DragInt("Max samples", &p.maxSamples);

    ImGui::Checkbox("Debug PRef", &p.grid.debugOutputPRef); SL;
    ImGui::Checkbox("Debug validity", &p.grid.debugOutputValidity); SL;
    ImGui::Checkbox("Debug bake", &p.grid.debugBakePRef);    

    ConstructComboBox("Swizzle", m_swizzleLabels, p.grid.axisSwizzle);
    ImGui::Text("Invert axes"); SL;
    ImGui::Checkbox("X", &p.grid.invertX); SL;
    ImGui::Checkbox("Y", &p.grid.invertY); SL;
    ImGui::Checkbox("Z", &p.grid.invertZ);
}

void LightProbeCameraShelf::Reset()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////

void FisheyeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];

    ImGui::Checkbox("Active", &p.camera.isActive); SL;
    ImGui::Checkbox("Live", &p.camera.isLive);

    ConstructTransform(p.transform, false);

    ImGui::SliderInt("Override max path depth", &p.camera.overrides.maxDepth, -1, 20);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::SliderInt("Max path depth", &p.maxDepth, 0, 20);
    ImGui::ColorEdit3("Ambient radiance", &p.ambientRadiance[0]);
    ImGui::Checkbox("Debug normals", &p.debugNormals);
    ImGui::Checkbox("Debug shaders", &p.debugShaders);
    ConstructComboBox("Importance mode", { "MIS", "Lights", "BxDFs" }, p.importanceMode);
    ConstructComboBox("Trace mode", { "Wavefront", "Path" }, p.traceMode);
}