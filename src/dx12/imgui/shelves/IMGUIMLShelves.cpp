#include "IMGUIMLShelves.h"
#include <random>

FCNNProbeDenoiserShelf::FCNNProbeDenoiserShelf(const Json::Node& json) :
    IMGUIShelf(json)
{}

void FCNNProbeDenoiserShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    if (ImGui::TreeNodeEx("Data Export Transform", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ConstructProbeDataTransform(m_p.dataTransform);

        ImGui::TreePop();
    }

    ConstructComboBox("Inference backend", { "CPU", "CUDA", "TensorRT" }, m_p.inferenceBackend);
    HelpMarker("Specifies the sampling method for direct lights.");

    if (ImGui::Button("Evaluate"))
    {
        m_p.doEvaluate = true;
    }     
    SL;

    if (ImGui::Button("Reload model"))
    {
        m_p.doReloadModel = true;
    }     
}

void FCNNProbeDenoiserShelf::Reset()
{
    m_p.doEvaluate = false;
    m_p.doReloadModel = false;
}


