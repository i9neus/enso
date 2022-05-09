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
        ConstructProbeDataTransform(m_p);

        ImGui::TreePop();
    }
}

void FCNNProbeDenoiserShelf::Reset()
{

}


