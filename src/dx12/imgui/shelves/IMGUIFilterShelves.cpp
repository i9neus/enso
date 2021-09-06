#include "IMGUIFilterShelves.h"
#include <random>

LightProbeKernelFilterShelf::LightProbeKernelFilterShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_linkAlphaK(m_p.nlm.alpha == m_p.nlm.K)
{}

void LightProbeKernelFilterShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructComboBox("Kernel type", { "Null", "Box", "Gaussian", "Non-local Means" }, m_p.filterType);
    ImGui::SliderFloat("Kernel radius", &m_p.radius, 0.0f, 10.0f);
    if (ImGui::SliderFloat("NLM alpha", &m_p.nlm.alpha, 0.0f, 2.0f)) 
    { 
        if (m_linkAlphaK) { m_p.nlm.K = m_p.nlm.alpha; } 
    }
    if (ImGui::SliderFloat("NLM k", &m_p.nlm.K, 0.0f, 2.0f)) 
    { 
        if (m_linkAlphaK) { m_p.nlm.alpha = m_p.nlm.K; }
    }
    ImGui::Checkbox("Link alpha/k", &m_linkAlphaK);
    if (ImGui::Button("Update")) { m_p.trigger = true; }
}

void LightProbeKernelFilterShelf::Reset()
{
    m_p.trigger = false;
}
