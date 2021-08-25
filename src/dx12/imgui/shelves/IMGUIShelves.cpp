#include "IMGUIShelves.h"

SimpleMaterialShelf::SimpleMaterialShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_albedoPicker(m_p.albedoHSV, "Albedo"),
    m_incandPicker(m_p.incandescenceHSV, "Incandescence")
{
}

void SimpleMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_albedoPicker.Construct();
    m_incandPicker.Construct();

    ImGui::Checkbox("Use grid", &m_p.useGrid);
}

void SimpleMaterialShelf::Randomise(const Cuda::vec2 range)
{
    m_p.albedoHSV.Randomise(range);
    m_p.incandescenceHSV.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

KIFSMaterialShelf::KIFSMaterialShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_albedoPicker(m_p.albedoHSV, "Albedo"),
    m_incandPicker(m_p.incandescenceHSV, "Incandescence")
{
}

void KIFSMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_albedoPicker.Construct();
    m_incandPicker.Construct();
}

void KIFSMaterialShelf::Randomise(const Cuda::vec2 range)
{
    m_p.albedoHSV.Randomise(range);
    m_p.incandescenceHSV.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

CornellMaterialShelf::CornellMaterialShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_pickers({ IMGUIJitteredColourPicker(m_p.albedoHSV[0], "Colour 1"),
                IMGUIJitteredColourPicker(m_p.albedoHSV[1], "Colour 2"),
                IMGUIJitteredColourPicker(m_p.albedoHSV[2], "Colour 3"),
                IMGUIJitteredColourPicker(m_p.albedoHSV[3], "Colour 4"),
                IMGUIJitteredColourPicker(m_p.albedoHSV[4], "Colour 5"),
                IMGUIJitteredColourPicker(m_p.albedoHSV[5], "Colour 6") })
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

PlaneShelf::PlaneShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.tracable.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void PlaneShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.tracable.transform, true);
    ImGui::Checkbox("Bounded", &m_p.isBounded);
}

void PlaneShelf::Randomise(const Cuda::vec2 range)
{
    m_p.tracable.renderObject.flags.Randomise(range);
    m_p.tracable.transform.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

SphereShelf::SphereShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({"Visible", "Exclude from bake"}));
}

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();    
    ConstructJitteredTransform(m_p.transform, true);
}

void SphereShelf::Randomise(const Cuda::vec2 range)
{
    m_p.renderObject.flags.Randomise(range);
    m_p.transform.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

CornellBoxShelf::CornellBoxShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.tracable.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void CornellBoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.tracable.transform, true);
}

void CornellBoxShelf::Randomise(const Cuda::vec2 range)
{
    m_p.tracable.renderObject.flags.Randomise(range);
    m_p.tracable.transform.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

QuadLightShelf::QuadLightShelf(const Json::Node& json) : 
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, "Light flags")
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void QuadLightShelf::Construct()
{   
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();    
    ConstructJitteredTransform(m_p.light.transform, true);
    m_colourPicker.Construct();
    m_intensity.Construct();
}

void QuadLightShelf::Randomise(const Cuda::vec2 range)
{
    m_p.light.transform.Randomise(range);
    m_p.light.colourHSV.Randomise(range);
    m_p.light.intensity.Randomise(range);
    m_p.light.renderObject.flags.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

SphereLightShelf::SphereLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, { "Light flags" })
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void SphereLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);
    m_colourPicker.Construct();
    m_intensity.Construct();
}

void SphereLightShelf::Randomise(const Cuda::vec2 range)
{
    m_p.light.transform.Randomise(range);
    m_p.light.colourHSV.Randomise(range);
    m_p.light.intensity.Randomise(range);
    m_p.light.renderObject.flags.Randomise(range);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void EnvironmentLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::Text("[No attributes]");
}

void EnvironmentLightShelf::Randomise(const Cuda::vec2 range)
{

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

    ConstructJitteredTransform(m_p.grid.transform, false);

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

///////////////////////////////////////////////////////////////////////////////////////////////////

void FisheyeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ImGui::Checkbox("Live", &m_p.camera.isLive);

    ConstructJitteredTransform(m_p.transform, false);

    ImGui::SliderInt("Override max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::SliderInt("Max path depth", &m_p.maxDepth, 0, 20);
    ImGui::ColorEdit3("Ambient radiance", &m_p.ambientRadiance[0]);
    ConstructComboBox("Importance mode", { "MIS", "Lights", "BxDFs" }, m_p.importanceMode);
    ConstructComboBox("Trace mode", { "Wavefront", "Path" }, m_p.traceMode);
    ConstructComboBox("Light selection mode", { "Naive", "Weighted" }, m_p.lightSelectionMode);
    ConstructComboBox("Shading mode", { "Full", "Simple", "Normals", "Debug" }, m_p.shadingMode);
}