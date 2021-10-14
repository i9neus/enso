#include "IMGUIFilterShelves.h"
#include <random>

LightProbeKernelFilterShelf::LightProbeKernelFilterShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_linkAlphaK(m_p.nlm.alpha == m_p.nlm.K)
{}

void LightProbeKernelFilterShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructComboBox("Kernel type", { "Null", "Box", "Gaussian", "NLM", "NLM (fixed variance)" }, m_p.filterType);
    ImGui::SliderInt("Kernel radius", &m_p.kernelRadius, 0, 10);
    if (ImGui::SliderFloat("NLM alpha", &m_p.nlm.alpha, 0.0f, 1.0f)) 
    { 
        if (m_linkAlphaK) { m_p.nlm.K = m_p.nlm.alpha; } 
    }
    if (ImGui::SliderFloat("NLM k", &m_p.nlm.K, 0.0f, 1.0f)) 
    { 
        if (m_linkAlphaK) { m_p.nlm.alpha = m_p.nlm.K; }
    }
    ImGui::DragFloat("NLM sigma", &m_p.nlm.sigma, max(0.001f, m_p.nlm.sigma * 0.01f), 0.0f, std::numeric_limits<float>::max());
    ImGui::Checkbox("Link alpha/k", &m_linkAlphaK);
}

void LightProbeKernelFilterShelf::Reset()
{

}

///////////////////////////////////////////////////////////////////////////////////////

LightProbeRegressionFilterShelf::LightProbeRegressionFilterShelf(const Json::Node& json) :
    IMGUIShelf(json)
{}

void LightProbeRegressionFilterShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::SliderInt("Polynomial order", &m_p.polynomialOrder, 0, 3);
    ImGui::SliderInt("Regression radius", &m_p.regressionRadius, 0, 4);
    ImGui::DragInt("Regression iterations", &m_p.regressionIterations, 1.0f, 1, 100);
    ImGui::DragInt("Regression max iterations", &m_p.regressionMaxIterations, 1.0f, 1, std::numeric_limits<int>::max());
    ImGui::DragFloat("Tikhonov regularisation", &m_p.tikhonovCoeff, max(m_p.tikhonovCoeff * 0.01f, 0.001f), 0.0f, 20.0f);
    ImGui::DragFloat("Learning rate", &m_p.learningRate, m_p.learningRate * 0.01f, 1e-8f, 1.0f, "%.8f");
    ImGui::SliderInt("Regression min samples", &m_p.minSamples, 0, 1024);
    ImGui::SliderInt("Reconstruction radius", &m_p.reconstructionRadius, 0, 4);

    ConstructComboBox("Kernel type", { "Null", "Box", "Gaussian", "NLM", "NLM (fixed variance)"}, m_p.filterType);
    if (ImGui::SliderFloat("NLM alpha", &m_p.nlm.alpha, 0.0f, 1.0f))
    {
        if (m_linkAlphaK) { m_p.nlm.K = m_p.nlm.alpha; }
    }
    if (ImGui::SliderFloat("NLM k", &m_p.nlm.K, 0.0f, 1.0f))
    {
        if (m_linkAlphaK) { m_p.nlm.alpha = m_p.nlm.K; }
    }
    ImGui::DragFloat("NLM sigma", &m_p.nlm.sigma, max(0.001f, m_p.nlm.sigma * 0.01f), 0.0f, std::numeric_limits<float>::max());
    ImGui::Checkbox("Link alpha/k", &m_linkAlphaK);

    for (auto& gridStats : m_probeGridStatistics)
    {
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
    }
}

void LightProbeRegressionFilterShelf::Reset()
{

}

void LightProbeRegressionFilterShelf::OnUpdateRenderObjectStatistics(const Json::Node& baseNode)
{
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

///////////////////////////////////////////////////////////////////////////////////////

LightProbeIOShelf::LightProbeIOShelf(const Json::Node& json) :
    IMGUIShelf(json)
{}

void LightProbeIOShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    if (ImGui::Button("Batch Denoise")) { m_p.doBatch = true; }
    if (ImGui::Button("Next")) { m_p.doNext = true; } SL;
    if (ImGui::Button("Previous")) { m_p.doPrevious = true; } 

    ImGui::Checkbox("Export USD", &m_p.exportUSD);
}

void LightProbeIOShelf::Reset()
{
    m_p = Cuda::LightProbeIOParams();
}


