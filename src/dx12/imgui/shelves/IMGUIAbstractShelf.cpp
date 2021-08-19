#include "IMGUIAbstractShelf.h"
#include "generic/JsonUtils.h"

void IMGUIAbstractShelf::ConstructComboBox(const std::string& name, const std::vector<std::string>& labels, int& selected)
{
    std::string badLabel = "[INVALID VALUE]";
    const char* selectedLabel = (selected < 0 || selected >= labels.size()) ? badLabel.c_str() : labels[selected].c_str();

    if (ImGui::BeginCombo(name.c_str(), selectedLabel, 0))
    {
        for (int n = 0; n < labels.size(); n++)
        {
            const bool isSelected = (selected == n);
            if (ImGui::Selectable(labels[n].c_str(), isSelected))
            {
                selected = n;
            }

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void IMGUIAbstractShelf::ConstructTransform(Cuda::BidirectionalTransform& transform)
{
    if (ImGui::TreeNode("Transform"))
    {
        ImGui::DragFloat3("Position", &transform.trans[0], math::max(0.01f, cwiseMax(transform.trans) * 0.01f));
        ImGui::DragFloat3("Rotation", &transform.rot[0], math::max(0.01f, cwiseMax(transform.rot) * 0.01f));
        //ImGui::DragFloat3("Scale XYZ", &transform.scale[0], math::max(0.01f, cwiseMax(transform.scale) * 0.01f));
        ImGui::DragFloat("Scale XYZ", &transform.scale[0], math::max(0.01f, cwiseMax(transform.scale) * 0.01f));
        transform.scale = transform.scale[0];
        ImGui::TreePop();
    }
}

