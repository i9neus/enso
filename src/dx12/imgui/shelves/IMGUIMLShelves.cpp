#include "IMGUIMLShelves.h"
#include <random>

FCNNProbeDenoiserShelf::FCNNProbeDenoiserShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_modelRootPath("Root path"),
    m_modelPreprocessPath("Pre-process path"),
    m_modelDenoiserPath("Denoiser path"),
    m_modelPostprocessPath("Post-process path")
{}

void FCNNProbeDenoiserShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    if (ImGui::TreeNodeEx("ONNX Paths", ImGuiTreeNodeFlags_DefaultOpen))
    {
        m_modelRootPath.Construct();
        m_modelPreprocessPath.Construct();
        m_modelDenoiserPath.Construct();
        m_modelPostprocessPath.Construct();
        
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Data Export Transform", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ConstructProbeDataTransform(m_p.dataTransform);

        ImGui::TreePop();
    }

    if (ImGui::Button("Evaluate"))
    {
        m_p.doEvaluate = true;
    }
}

void FCNNProbeDenoiserShelf::Reset()
{
    m_p.doEvaluate = false;
}


