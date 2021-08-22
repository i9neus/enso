#include "IMGUIShelves.h"

void SimpleMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::ColorEdit3(tfm::format("Albedo (%s)", m_id).c_str(), (float*)&m_p.albedo);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&m_p.incandescence);
    ImGui::Checkbox("Use grid", &m_p.useGrid);
}

void SimpleMaterialShelf::Randomise(const Cuda::vec2 range)
{
    const Cuda::vec2 randomRange = range;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void KIFSMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::SliderFloat3("HSL lower", &m_p.hslLower[0], 0.0f, 1.0f);
    ImGui::SliderFloat3("HSL upper", &m_p.hslUpper[0], 0.0f, 1.0f);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&m_p.incandescence);
}
void KIFSMaterialShelf::Randomise(const Cuda::vec2 range)
{
    const Cuda::vec2 randomRange = range;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

CornellMaterialShelf::CornellMaterialShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_pickers({ IMGUIColourPicker(m_p.albedoHSV[0], "Colour 1"),
                IMGUIColourPicker(m_p.albedoHSV[1], "Colour 2"),
                IMGUIColourPicker(m_p.albedoHSV[2], "Colour 3"),
                IMGUIColourPicker(m_p.albedoHSV[3], "Colour 4"),
                IMGUIColourPicker(m_p.albedoHSV[4], "Colour 5"),
                IMGUIColourPicker(m_p.albedoHSV[5], "Colour 6") })
{
}

void CornellMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    for (int i = 0; i < 6; ++i) { m_pickers[i].Construct(); }
}

void CornellMaterialShelf::Randomise(const Cuda::vec2 range)
{
    for (int i = 0; i < 6; ++i) { m_p.albedoHSV[i].Randomise(range); }
    Update();
}

void CornellMaterialShelf::Update()
{
    for (int i = 0; i < 6; ++i) { m_pickers[i].Update(); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void PlaneShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructTransform(m_p.tracable.transform, true);
    ImGui::Checkbox("Bounded", &m_p.isBounded);
}

void PlaneShelf::Randomise(const Cuda::vec2 range)
{
    m_p.tracable.transform.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructTransform(m_p.transform, true);

    ImGui::Checkbox("Exclude from bake", &m_p.excludeFromBake);
}

void SphereShelf::Randomise(const Cuda::vec2 range)
{
    m_p.transform.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void CornellBoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructTransform(m_p.tracable.transform, true);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

void QuadLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructTransform(m_p.transform, true);

    ImGui::ColorEdit3("Colour", &m_p.colour[0], ImGuiColorEditFlags_InputHSV);
    ImGui::SliderFloat("Intensity", &m_p.intensity, -10.0f, 10.0f);
}

void QuadLightShelf::Randomise(const Cuda::vec2 range)
{
    const Cuda::vec2 randomRange = range;
    m_p.transform.Randomise(randomRange);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void SphereLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructTransform(m_p.transform, true);

    ImGui::ColorEdit3("Colour", &m_p.colour[0]);
    ImGui::SliderFloat("Intensity", &m_p.intensity, -10.0f, 10.0f);
}

void SphereLightShelf::Randomise(const Cuda::vec2 range)
{
    m_p.transform.Randomise(range);
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

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ImGui::Checkbox("Live", &m_p.camera.isLive);  SL;
    ImGui::Checkbox("Realtime", &m_p.isRealtime);
    ImGui::Checkbox("Mimic light probe", &m_p.mimicLightProbe);

    ImGui::DragFloat3("Position", &m_p.position[0], math::max(0.01f, cwiseMax(m_p.position) * 0.01f));
    ImGui::DragFloat3("Look at", &m_p.lookAt[0], math::max(0.01f, cwiseMax(m_p.lookAt) * 0.01f));

    ImGui::SliderFloat("F-stop", &m_p.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &m_p.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &m_p.focalPlane, 0.0f, 2.0f);
    ImGui::SliderFloat("Display gamma", &m_p.displayGamma, 0.01f, 5.0f);

    ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
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

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ImGui::Checkbox("Live", &m_p.camera.isLive);

    ConstructTransform(m_p.grid.transform, false);

    ImGui::InputInt3("Grid density", &m_p.grid.gridDensity[0]);
    ConstructComboBox("SH order", { "L0", "L1", "L2" }, m_p.grid.shOrder);
    ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());

    ImGui::DragInt("Max samples", &m_p.maxSamples);

    ImGui::Checkbox("Debug PRef", &m_p.grid.debugOutputPRef); SL;
    ImGui::Checkbox("Debug validity", &m_p.grid.debugOutputValidity); SL;
    ImGui::Checkbox("Debug bake", &m_p.grid.debugBakePRef);

    ConstructComboBox("Swizzle", m_swizzleLabels, m_p.grid.axisSwizzle);
    ImGui::Text("Invert axes"); SL;
    ImGui::Checkbox("X", &m_p.grid.invertX); SL;
    ImGui::Checkbox("Y", &m_p.grid.invertY); SL;
    ImGui::Checkbox("Z", &m_p.grid.invertZ);
}

void LightProbeCameraShelf::Reset()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////

void FisheyeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ImGui::Checkbox("Live", &m_p.camera.isLive);

    ConstructTransform(m_p.transform, false);

    ImGui::SliderInt("Override max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::SliderInt("Max path depth", &m_p.maxDepth, 0, 20);
    ImGui::ColorEdit3("Ambient radiance", &m_p.ambientRadiance[0]);
    ImGui::Checkbox("Debug normals", &m_p.debugNormals);
    ImGui::Checkbox("Debug shaders", &m_p.debugShaders);
    ConstructComboBox("Importance mode", { "MIS", "Lights", "BxDFs" }, m_p.importanceMode);
    ConstructComboBox("Trace mode", { "Wavefront", "Path" }, m_p.traceMode);
}