#include "IMGUIFilterShelves.h"
#include <random>

LightProbeKernelFilterShelf::LightProbeKernelFilterShelf(const Json::Node& json) :
    IMGUIShelf(json)
{}

void LightProbeKernelFilterShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructComboBox("Kernel type", { "Null", "Gaussian" }, m_p.filterType);
    ImGui::SliderFloat("Kernel radius", &m_p.radius, 0.0f, 20.0f);
    if (ImGui::Button("Update")) { m_p.trigger = true; }
}

void LightProbeKernelFilterShelf::Reset()
{
    m_p.trigger = false;
}
