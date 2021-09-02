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
    HelpMarker("Check to bound the plane in the range [-0.5, 0.5] in object space. Uncheck to trace to infinity in all directions.");
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

BoxShelf::BoxShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void BoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.transform, true, true);
}

void BoxShelf::Jitter(const uint flags, const uint operation)
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
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
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
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
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
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
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
    HelpMarker("The projected solid angle of the light in degrees.");
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
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
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
    HelpMarker("Selects the evaulated light probe grid depending on the bake setting. 0 = direct or combined, 1 = indirect");

    ImGui::PopItemWidth();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void PerspectiveCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::Checkbox("Active", &m_p.camera.isActive); SL;
    ToolTip("Active cameras are rendered in parallel by the ray tracer.");

    ImGui::Checkbox("Live", &m_p.camera.isLive);  SL;
    ToolTip("Live cameras are visible in the main viewport.");

    ImGui::Checkbox("Realtime", &m_p.isRealtime);
    ToolTip("Realtime cameras are continually reset after every rendered frame instead of slowly converging over time.");

    ImGui::DragFloat3("Position", &m_p.position[0], math::max(0.01f, cwiseMax(m_p.position) * 0.01f));
    HelpMarker("The position of the camera.");

    ImGui::DragFloat3("Look at", &m_p.lookAt[0], math::max(0.01f, cwiseMax(m_p.lookAt) * 0.01f));
    HelpMarker("The coordinate the camera is looking at.");

    ImGui::SliderFloat("F-number", &m_p.fStop, 0.0f, 1.0f);
    HelpMarker("The F-number of the camera.");

    ImGui::SliderFloat("Focal length", &m_p.fLength, 0.0f, 1.0f);
    HelpMarker("The focal length of the camera in mm.");

    ImGui::SliderFloat("Focal plane", &m_p.focalPlane, 0.0f, 2.0f);
    HelpMarker("The position of the focal plane as a functino of the distance between the camera position and its look-at vector.");

    ImGui::SliderFloat("Display gamma", &m_p.displayGamma, 0.01f, 5.0f);
    HelpMarker("The gamma value applied to the viewport window.");

    ImGui::DragInt("Max samples", &m_p.camera.maxSamples, 1.0f, -1, std::numeric_limits<int>::max());
    HelpMarker("The maximum number of samples per pixel. -1 = infinite.");

    ImGui::InputInt("Seed", &m_p.camera.seed);
    HelpMarker("The seed value used to see the random number generators.");

    ImGui::Checkbox("Jitter seed", &m_p.camera.randomiseSeed);
    HelpMarker("Whether the seed value should be randomised before every bake permutation.");

    ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
    HelpMarker("The maximum depth a ray can travel before it's terminated.");

    ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
    HelpMarker("Specifies the maximum value of a ray splat before it gets clipped. Setting this value too low will result in energy loss and bias.");

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
    ToolTip("Active cameras are rendered in parallel by the ray tracer.");

    ImGui::Checkbox("Live", &m_p.camera.isLive);  SL;
    ToolTip("Live cameras are visible in the main viewport.");

    ConstructJitteredTransform(m_p.grid.transform, false);

    ImGui::InputInt3("Grid density", &m_p.grid.gridDensity[0]);
    HelpMarker("The density of the light probe grid as width x height x depth");

    ConstructComboBox("SH order", { "L0", "L1", "L2" }, m_p.grid.shOrder);
    HelpMarker("The order of the spherical harmonic encoding");

    ImGui::Checkbox("Use validity", &m_p.grid.useValidity);
    HelpMarker("Detects invalid probes and exludes from the irradiance reconstruction.");

    ConstructComboBox("Output mode", { "Irradiance", "Validity", "Harmonic mean", "pRef" }, m_p.grid.outputMode);
    HelpMarker("Specifies the output of thee light probe evaluation. Use this to debug probe validity and other values.");

    ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
    HelpMarker("The maximum depth a ray can travel before it's terminated.");

    ConstructComboBox("Direct/indirect", { "Combined", "Separated" }, m_p.lightingMode);
    HelpMarker("Specifies whether direct and indirect illumination should be combined in a single pass or exported as two separate grids.");

    ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
    HelpMarker("Specifies the maximum value of a ray splat before it gets clipped. Setting this value too low will result in energy loss and bias.");

    ImGui::SliderInt("Grid update interval", &m_p.gridUpdateInterval, 1, 200);
    HelpMarker("Specifies the interval that the light probe grid is consolidated from the accumulation buffer. Grid updating is expensive so setting this value too low may slow down the bake.");

    ImGui::DragInt("Max samples", &m_p.camera.maxSamples, 1.0f, -1, std::numeric_limits<int>::max());
    HelpMarker("The maximum number of samples per probe.");

    ImGui::InputInt("Seed", &m_p.camera.seed);
    HelpMarker("The seed value used to see the random number generators.");

    ImGui::Checkbox("Jitter seed", &m_p.camera.randomiseSeed);
    HelpMarker("Whether the seed value should be randomised before every bake permutation.");

    ConstructComboBox("Swizzle", m_swizzleLabels, m_p.grid.axisSwizzle);
    HelpMarker("The swizzle factor applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

    ImGui::Text("Invert axes"); SL;
    ToolTip("Axis inverstion applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

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

    ImGui::DragFloat("Russian roulette", &m_p.russianRouletteThreshold, 0.001f, 0.0f, 1.0f, "%.4f");
    HelpMarker("The Russian roulette threshold specifies the minimum throughput weight below which rays are probabilsitically terminated.");

    ConstructComboBox("Direct sampling", { "MIS", "Lights", "BxDFs" }, m_p.importanceMode);
    HelpMarker("Specifies the sampling method for direct lights.");

    //ConstructComboBox("Trace mode", { "Wavefront", "Path" }, m_p.traceMode);
    //HelpMarker("Speciffies whether to use wavefront tracing or interative path tracing.");

    ConstructComboBox("Light selection mode", { "Naive", "Weighted" }, m_p.lightSelectionMode);
    HelpMarker("Specifies how direct lights should be selected. Naive mode selects lights at random. Weighted mode picks lights based on their estimated contribution.");

    ConstructComboBox("Shading mode", { "Default", "Simple", "Normals", "Debug" }, m_p.shadingMode);
    HelpMarker("Specifies how objects are shaded. Default performs full physically based shading. Simple reverts to a basic lighting model. Normals outputs the surface normals only. Debug outputs shader diagnostics.");
}

void WavefrontTracerShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteColours)) { return; }
}