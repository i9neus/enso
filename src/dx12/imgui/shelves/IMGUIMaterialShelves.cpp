#include "IMGUIMaterialShelves.h"
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