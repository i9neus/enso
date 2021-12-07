#include "IMGUICameraShelves.h"
#include <random>

#include "kernels/cameras/CudaCamera.cuh"

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

    ImGui::SliderFloat("Display exposure", &m_p.displayExposure, -15.0f, 15.0f);
    HelpMarker("The exposure value in stops applied to the viewport window.");
    
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
    m_bakeProgress(-1.0f),
    m_bakeConvergence(-1.0f),
    m_frameIdx(-1),
    m_minMaxMSE(Cuda::kMinMaxReset)
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

    //ImGuiIO& io = ImGui::GetIO();
    //ImFontAtlas* atlas = io.Fonts; 
    //atlas->Fonts[0]->Scale = 0.95f;

    if (ImGui::TreeNodeEx("Encoding", ImGuiTreeNodeFlags_DefaultOpen))
    {        
        ImGui::DragInt3("Grid density", &m_p.grid.gridDensity[0], 1.0f, 2, 50);
        HelpMarker("The density of the light probe grid as width x height x depth");

        ConstructComboBox("SH order", { "L0", "L1", "L2" }, m_p.grid.shOrder);
        HelpMarker("The order of the spherical harmonic encoding");

        ImGui::SliderInt("Grid update interval", &m_p.gridUpdateInterval, 1, 200);
        HelpMarker("Specifies the interval that the light probe grid is consolidated from the accumulation buffer. Grid updating is expensive so setting this value too low may slow down the bake.");

        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Dilate", &m_p.grid.dilate);
        HelpMarker("Prevents shadow leaks by dilating valid regions into invalid regions.");

        ConstructComboBox("Output mode", { "Irradiance", "Irradiance Laplacian", "Harmonic mean", "pRef", "Convergence", "Relative MSE" }, m_p.grid.outputMode);
        HelpMarker("Specifies the output of thee light probe evaluation. Use this to debug probe validity and other values.");

        ConstructComboBox("Direct/indirect", { "Combined", "Separated" }, m_p.lightingMode);
        HelpMarker("Specifies whether direct and indirect illumination should be combined in a single pass or exported as two separate grids.");

        ImGui::SliderInt("Grid update interval", &m_p.gridUpdateInterval, 1, 200);
        HelpMarker("Specifies the interval that the light probe grid is consolidated from the accumulation buffer. Grid updating is expensive so setting this value too low may slow down the bake.");

        ImGui::TreePop();
    }    

    if (ImGui::TreeNodeEx("Sampling", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ConstructComboBox("Sampling mode", { "Fixed", "Adaptive (Relative)", "Adaptive (Absolute)" }, m_p.camera.samplingMode);
        HelpMarker("Specifies which sampling mode to use. Fixed takes a fixed number of samples up to the input value. Adaptive takes a noise threshold beyond which sampling is terminated.");

        ImGui::DragInt("Max samples", &m_p.camera.maxSamples, 1.0f, -1, std::numeric_limits<int>::max());
        HelpMarker("The maximum number of samples per probe. Set to -1 to take unlimited samples.");

        if (m_p.camera.samplingMode != Cuda::kCameraSamplingFixed)
        {
            ImGui::DragFloat("Noise treshold", &m_p.camera.sqrtErrorThreshold, math::max(m_p.camera.sqrtErrorThreshold * 1e-4f, 1e-4f), 0.0f, std::numeric_limits<float>::max());
            HelpMarker("The noise threshold below which sampling is terminated.");
        }

        ImGui::InputInt("Seed", &m_p.camera.seed);
        HelpMarker("The seed value used to see the random number generators.");

        ImGui::Checkbox("Jitter seed", &m_p.camera.randomiseSeed);
        HelpMarker("Whether the seed value should be randomised before every bake permutation.");

        ImGui::SliderInt("Max path depth", &m_p.camera.overrides.maxDepth, -1, 20);
        HelpMarker("The maximum depth a ray can travel before it's terminated.");

        ImGui::DragFloat("Splat clamp", &m_p.camera.splatClamp, math::max(0.01f, m_p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
        HelpMarker("Specifies the maximum value of a ray splat before it gets clipped. Setting this value too low will result in energy loss and bias.");

        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Data Export Attributes", 0))
    {
        ConstructComboBox("Swizzle", m_swizzleLabels, m_p.grid.axisSwizzle);
        HelpMarker("The swizzle factor applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

        ImGui::Text("Invert axes"); SL;
        ToolTip("Axis inverstion applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

        ImGui::Checkbox("X", &m_p.grid.invertX); SL;
        ImGui::Checkbox("Y", &m_p.grid.invertY); SL;
        ImGui::Checkbox("Z", &m_p.grid.invertZ);

        ImGui::TreePop();
    }

    m_p.camera.seed = max(0, m_p.camera.seed);

    if (m_bakeProgress >= 0.0f)     { ImGui::Text(tfm::format("Bake progress: %.2f%%", m_bakeProgress * 100.0f).c_str()); }
    if (m_bakeConvergence >= 0.0f)
    {
        ImGui::Text(tfm::format("Bake convergence: %.2f%%", m_bakeConvergence * 100.0f).c_str());
        ImGui::Text(tfm::format("Mean I: %.5f%%", m_meanI).c_str());
        ImGui::Text(tfm::format("MSE I: %.5f%%", m_MSE).c_str());
    }

    constexpr int kMSEPlotDensity = 10;
    std::string mseFormat = tfm::format("%.5f", m_MSE);
    ImGui::PlotLines("MSE", m_MSEData.data(), m_MSEData.size(), 0, mseFormat.c_str(), m_minMaxMSE.x, m_minMaxMSE.y, ImVec2(0, 80.0f));

    for (int i = 0; i < m_probeGridStatistics.size(); ++i)
    {
        if (!ImGui::TreeNodeEx(tfm::format("Grid statistics %i", i).c_str(), 0)) { continue; }

        auto& gridStats = m_probeGridStatistics[i];
        ImGui::PushID(gridStats.gridID.c_str());
        ImGui::Text(gridStats.gridID.c_str());
        ImGui::Text(tfm::format("Min/max samples: [%i, %i]", gridStats.minSamplesTaken, gridStats.maxSamplesTaken).c_str());
        ImGui::Text(tfm::format("Mean probe validity: %.2f%%", gridStats.meanProbeValidity * 100.0f).c_str());
        ImGui::Text(tfm::format("Mean probe distance: %.5f", gridStats.meanProbeDistance).c_str());       

        if (gridStats.hasHistogram)
        {
            for (const auto& histogramWidget : gridStats.histogramWidgetData)
            {
                ImGui::PlotHistogram("Distance histogram", histogramWidget.data.data(), histogramWidget.data.size(), 0, NULL, 0.0f, histogramWidget.maxValue, ImVec2(0, 50.0f));
            }
        }

        ImGui::Separator();
        ImGui::PopID();
        ImGui::TreePop();
    }
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

void LightProbeCameraShelf::OnUpdateRenderObjectStatistics(const Json::Node& baseNode)
{    
    baseNode.GetValue("bakeProgress", m_bakeProgress, Json::kSilent);
    baseNode.GetValue("bakeConvergence", m_bakeConvergence, Json::kSilent);    
    
    // Look for an MSE value
    if (baseNode.GetValue("mse", m_MSE, Json::kSilent) && m_MSE > 0.0f &&
        baseNode.GetValue("meanI", m_meanI, Json::kSilent) && m_meanI > 0.0f)
    {
        constexpr int kMaxMSEDataSize = 1000;
        int frameIdx = -1;
        baseNode.GetValue("frameIdx", frameIdx, Json::kSilent);

        // If the render has been reset, clear the stored data
        if (frameIdx <= m_frameIdx)
        {
            m_MSEData.clear();
            m_minMaxMSE = Cuda::vec2(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
        }
        // Add the data to the history
        if (m_MSEData.size() < kMaxMSEDataSize)
        {
            const float logMSE = std::log(1e-10f + m_MSE);
            m_minMaxMSE = Cuda::MinMax(m_minMaxMSE, logMSE);
            m_MSEData.push_back(logMSE);
        }

        m_frameIdx = frameIdx;
    }

    const Json::Node gridSetNode = baseNode.GetChildObject("grids", Json::kSilent);
    if (!gridSetNode) { return; }

    Assert(gridSetNode.IsObject());
    m_probeGridStatistics.resize(gridSetNode.NumMembers());

    int gridIdx = 0;
    std::vector<std::vector<uint>> histogramMatrix;
    for (::Json::Node::ConstIterator it = gridSetNode.begin(); it != gridSetNode.end(); ++it, ++gridIdx)
    {    
        const auto& gridNode = *it;
        auto& stats = m_probeGridStatistics[gridIdx];

        stats.gridID = it.Name();
        gridNode.GetValue("minSamples", stats.minSamplesTaken, Json::kSilent);
        gridNode.GetValue("maxSamples", stats.minSamplesTaken, Json::kSilent);
        gridNode.GetValue("meanProbeValidity", stats.meanProbeValidity, Json::kSilent);
        gridNode.GetValue("meanProbeDistance", stats.meanProbeDistance, Json::kSilent); 

        stats.hasHistogram = false;
        histogramMatrix.clear();
        if (gridNode.GetArray2DValues("coeffHistograms", histogramMatrix, Json::kSilent))
        {
            // Map the input data into something the widget can use
            stats.histogramWidgetData.resize(histogramMatrix.size());
            for (int histogramIdx = 0; histogramIdx < histogramMatrix.size(); ++histogramIdx)
            {
                const auto& inputData = histogramMatrix[histogramIdx];
                auto& outputData = stats.histogramWidgetData[histogramIdx];
                outputData.data.resize(inputData.size());
                outputData.maxValue = 0;
                for (int binIdx = 0; binIdx < inputData.size(); ++binIdx)
                {
                    //outputData.data[binIdx] = std::log(1.0f + inputData[binIdx]);
                    outputData.data[binIdx] = inputData[binIdx];
                    outputData.maxValue = max(outputData.maxValue, outputData.data[binIdx]);
                }
            }
            stats.hasHistogram = true;
        }
    }
}
