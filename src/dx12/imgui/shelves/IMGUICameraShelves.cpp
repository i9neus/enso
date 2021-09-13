#include "IMGUICameraShelves.h"
#include <random>

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
    : IMGUIShelf(json),
    m_meanProbeValidity(0.0f),
    m_meanProbeDistance(0.0f),
    m_maxSamplesTaken(0),
    m_minSamplesTaken(0)
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

    ImGui::Text(tfm::format("Mean probe validity: %.2f%%", m_meanProbeValidity * 100.0f).c_str());
    ImGui::Text(tfm::format("Mean probe distance: %.5f", m_meanProbeDistance).c_str());
    ImGui::PlotHistogram("Distance histogram", m_distanceHistogramWidgetData.data(), m_distanceHistogramWidgetData.size(), 0, NULL, 0.0f, float(m_histogramMaxBucket), ImVec2(0, 80.0f));
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

void LightProbeCameraShelf::OnUpdateRenderObjectStatistics(const Json::Node& node)
{    
    node.GetValue("minSamples", m_minSamplesTaken, Json::kSilent);
    node.GetValue("maxSamples", m_minSamplesTaken, Json::kSilent);
    node.GetValue("meanProbeValidity", m_meanProbeValidity, Json::kSilent);
    node.GetValue("meanProbeDistance", m_meanProbeDistance, Json::kSilent);
    
    m_hasDistanceHistogram = false;    
    if (node.GetArrayValues("distanceHistogram", m_distanceHistogram, Json::kSilent))
    {
        m_distanceHistogramWidgetData.resize(m_distanceHistogram.size());
        m_histogramMaxBucket = 0;
        for (int idx = 0; idx < m_distanceHistogram.size(); ++idx) 
        { 
            m_histogramMaxBucket = max(m_histogramMaxBucket, m_distanceHistogram[idx]); 
            m_distanceHistogramWidgetData[idx] = m_distanceHistogram[idx];
        }

        m_hasDistanceHistogram = true;
    }


}
