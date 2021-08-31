#include "IMGUIShelves.h"
#include <random>

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

void SimpleMaterialShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteMaterials) || !(flags & kStatePermuteColours)) { return; }

    m_p.albedoHSV.Update(operation);
    m_p.incandescenceHSV.Update(operation);
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

void KIFSMaterialShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteMaterials) || !(flags & kStatePermuteColours)) { return; }

    m_p.albedoHSV.Update(operation);
    m_p.incandescenceHSV.Update(operation);
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

void CornellMaterialShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteMaterials) || !(flags & kStatePermuteColours)) { return; }

    for (int i = 0; i < 6; ++i) { m_p.albedoHSV[i].Update(operation); }
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

void PlaneShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.tracable.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.tracable.transform.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

SphereShelf::SphereShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.transform, true);
}

void SphereShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.transform.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

CornellBoxShelf::CornellBoxShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.tracable.renderObject.flags, "Object flags"),
    m_faceMask(m_p.faceMask, "Face mask"),
    m_cameraRayMask(m_p.cameraRayMask, "Camera ray mask")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
    m_faceMask.Initialise(std::vector<std::string>({ "1", "2", "3", "4", "5", "6" }));
    m_cameraRayMask.Initialise(std::vector<std::string>({ "1", "2", "3", "4", "5", "6" }));
}

void CornellBoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructJitteredTransform(m_p.tracable.transform, true);

    m_flags.Construct();
    m_faceMask.Construct();
    m_cameraRayMask.Construct();
}

void CornellBoxShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.tracable.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.tracable.transform.Update(operation); }

    m_p.faceMask.Update(operation);
    m_p.cameraRayMask.Update(operation);
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

void QuadLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
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

void SphereLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

DistantLightShelf::DistantLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, { "Light flags" })
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void DistantLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);

    m_colourPicker.Construct();
    m_intensity.Construct();

    ImGui::DragFloat("Angle", &m_p.angle, math::max(0.01f, m_p.angle * 0.01f), 0.0f, std::numeric_limits<float>::max());
}

void DistantLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

EnvironmentLightShelf::EnvironmentLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, { "Light flags" })
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void EnvironmentLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);
    m_colourPicker.Construct();
    m_intensity.Construct();
}

void EnvironmentLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}
///////////////////////////////////////////////////////////////////////////////////////////////////

void LambertBRDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::PushItemWidth(50);
    ImGui::SliderInt("Probe volume grid", &m_p.lightProbeGridIdx, 0, 1);
    ImGui::PopItemWidth();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void PerspectiveCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ImGui::Checkbox("Live", &m_p.camera.isLive);  SL;
    ImGui::Checkbox("Realtime", &m_p.isRealtime);

    ImGui::DragFloat3("Position", &m_p.position[0], math::max(0.01f, cwiseMax(m_p.position) * 0.01f));
    ImGui::DragFloat3("Look at", &m_p.lookAt[0], math::max(0.01f, cwiseMax(m_p.lookAt) * 0.01f));

    ImGui::SliderFloat("F-stop", &m_p.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &m_p.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &m_p.focalPlane, 0.0f, 2.0f);
    ImGui::SliderFloat("Display gamma", &m_p.displayGamma, 0.01f, 5.0f);

    ImGui::DragInt("Max samples", &m_p.camera.maxSamples, 1.0f, -1, std::numeric_limits<int>::max());
    ImGui::InputInt("Seed", &m_p.camera.seed);
    ImGui::Checkbox("Randomise seed", &m_p.camera.randomiseSeed);

    ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());

    ConstructComboBox("Emulate light probe", { "None", "All", "Direct only", "Indirect only" }, m_p.lightProbeEmulation);

    m_p.camera.seed = max(0, m_p.camera.seed);
}

void PerspectiveCameraShelf::Jitter(const uint flags, const uint operation)
{
    /*if (m_p.camera.randomiseSeed)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<> rng(0, std::numeric_limits<int>::max());

        m_p.camera.seed = rng(mt);
    }*/
}

LightProbeCameraShelf::LightProbeCameraShelf(const Json::Node& json)
    : IMGUIShelf(json)
{
    m_swizzleLabels = { "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX" };
}

void LightProbeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ImGui::Checkbox("Live", &m_p.camera.isLive);

    ConstructJitteredTransform(m_p.grid.transform, false);

    ImGui::InputInt3("Grid density", &m_p.grid.gridDensity[0]);
    ConstructComboBox("SH order", { "L0", "L1", "L2" }, m_p.grid.shOrder);
    ImGui::Checkbox("Validity compensation", &m_p.grid.useValidity);
    ConstructComboBox("Output mode", { "Irradiance", "Validity", "Harmonic mean", "pRef" }, m_p.grid.outputMode);
    ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
    ConstructComboBox("Lighting mode", { "All", "Direct + indirect" }, m_p.lightingMode);
    ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
    ImGui::SliderInt("Grid update interval", &m_p.gridUpdateInterval, 1, 200);

    ImGui::DragInt("Max samples", &m_p.camera.maxSamples, 1.0f, -1, std::numeric_limits<int>::max());
    ImGui::InputInt("Seed", &m_p.camera.seed);
    ImGui::Checkbox("Jitter seed", &m_p.camera.randomiseSeed);

    ConstructComboBox("Swizzle", m_swizzleLabels, m_p.grid.axisSwizzle);
    ImGui::Text("Invert axes"); SL;
    ImGui::Checkbox("X", &m_p.grid.invertX); SL;
    ImGui::Checkbox("Y", &m_p.grid.invertY); SL;
    ImGui::Checkbox("Z", &m_p.grid.invertZ);

    m_p.camera.seed = max(0, m_p.camera.seed);
}

void LightProbeCameraShelf::Randomise()
{
    if (m_p.camera.randomiseSeed)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<> rng(0, std::numeric_limits<int>::max());

        m_p.camera.seed = rng(mt);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

WavefrontTracerShelf::WavefrontTracerShelf(const Json::Node& json) :
    IMGUIShelf(json)
{}

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::SliderInt("Max path depth", &m_p.maxDepth, 0, 20);
    ImGui::DragFloat("Russian roulette", &m_p.russianRouletteThreshold, 0.001f, 0.0f, 1.0f, "%.4f");
    ConstructComboBox("Importance mode", { "MIS", "Lights", "BxDFs" }, m_p.importanceMode);
    ConstructComboBox("Trace mode", { "Wavefront", "Path" }, m_p.traceMode);
    ConstructComboBox("Light selection mode", { "Naive", "Weighted" }, m_p.lightSelectionMode);
    ConstructComboBox("Shading mode", { "Full", "Simple", "Normals", "Debug" }, m_p.shadingMode);
}

void WavefrontTracerShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteColours)) { return; }
}