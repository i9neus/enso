#include "IMGUIElement.h"
#include "generic/JsonUtils.h"

IMGUIListBox::IMGUIListBox(const std::string& id, const std::string& addLabel, const std::string& overwriteLabel, const std::string& deleteLabel) :
    m_currentIdx(-1),
    m_listBoxID(id),
    m_addLabel(addLabel),
    m_overwriteLabel(overwriteLabel),
    m_deleteLabel(deleteLabel)
{
    m_currentIdx = -1;
    m_newItemIDData.resize(2048);
    std::memset(m_newItemIDData.data(), '\0', sizeof(char) * m_newItemIDData.size());
}

void IMGUIListBox::Construct()
{
    ImGui::PushID(m_listBoxID.c_str());

    if (ImGui::BeginListBox(m_listBoxID.c_str()))
    {
        auto it = m_listItems.begin();
        for (int n = 0; n < m_listItems.size(); n++, ++it)
        {
            const bool isSelected = (m_currentIdx == n);
            if (ImGui::Selectable(it->c_str(), isSelected))
            {
                m_currentIdx = n;
            }
            if (isSelected) { ImGui::SetItemDefaultFocus(); }
        }
        ImGui::EndListBox(); 
    }

    // New element input control
    ImGui::InputText("", m_newItemIDData.data(), m_newItemIDData.size());

    // Save the current state to the container
    if (!m_addLabel.empty() && ImGui::Button(m_addLabel.c_str()))
    {
        std::string newElement = std::string(m_newItemIDData.data());
        if (!m_onAdd || m_onAdd(newElement))
        {
            Insert(newElement);
            std::memset(m_newItemIDData.data(), '\0', sizeof(char) * m_newItemIDData.size());
        }
    }
    SL;
    // Overwrite the currently selected state
    if (!m_overwriteLabel.empty() && ImGui::Button(m_overwriteLabel.c_str()) && m_currentIdx >= 0 && !m_listItems.empty())
    {
        std::list<std::string>::iterator it = m_listItems.begin();
        std::advance(it, m_currentIdx);
        if (it != m_listItems.end())
        {
            std::string overwriteElement = std::string(m_newItemIDData.data());
            if (!m_onAdd || m_onAdd(overwriteElement))
            {
                *it = overwriteElement;
                std::memset(m_newItemIDData.data(), '\0', sizeof(char) * m_newItemIDData.size());
            }
        }
    }
    SL;
    // Erase a saved state from the container
    if (!m_deleteLabel.empty() && ImGui::Button(m_deleteLabel.c_str()) && m_currentIdx >= 0 && !m_listItems.empty())
    {
        std::list<std::string>::iterator it = m_listItems.begin();
        std::advance(it, m_currentIdx);
        if (it != m_listItems.end())
        {
            if (!m_onDelete || m_onDelete(*it))
            {
                m_listItems.erase(it);
            }
        }
        m_currentIdx = -1;
    }
    /*SL;
    // Erase a saved state from the container
    if (ImGui::Button("Delete All"))
    {
        if (!m_onDeleteAll || m_onDeleteAll())
        {
            Clear();
        }
    }*/

    ImGui::PopID();
}

void IMGUIListBox::Clear()
{
    m_listItems.clear();
    m_currentIdx = -1;
}

void IMGUIListBox::Insert(const std::string& str)
{
    m_listItems.push_back(str);
    m_currentIdx = m_listItems.size() - 1;
}

std::string IMGUIListBox::GetCurrentlySelectedText() const
{
    if (m_currentIdx < 0) { return ""; }

    std::list<std::string>::const_iterator it = m_listItems.begin();
    std::advance(it, m_currentIdx);
    return (it != m_listItems.end()) ? *it : "";
    
}

void IMGUIElement::ConstructComboBox(const std::string& name, const std::vector<std::string>& labels, int& selected)
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

void IMGUIElement::ConstructTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable)
{
    if (ImGui::TreeNode("Transform"))
    {
        ImGui::PushID("p");
        ImGui::DragFloat3("Position", &transform.trans.p[0], math::max(0.01f, cwiseMax(transform.trans.p) * 0.01f));
        ImGui::DragFloat3("Rotation", &transform.rot.p[0], math::max(0.01f, cwiseMax(transform.rot.p) * 0.01f));
        ImGui::DragFloat("Scale", &transform.scale.p[0], math::max(0.01f, cwiseMax(transform.scale.p) * 0.01f));
        transform.scale.p = transform.scale.p[0];
        ImGui::PopID();

        if (isJitterable)
        {
            ImGui::PushID("dpdt");
            ImGui::DragFloat3("+/- Position", &transform.trans.dpdt[0], math::max(0.0001f, cwiseMax(transform.trans.dpdt) * 0.01f));
            ImGui::DragFloat3("+/- Rotation", &transform.rot.dpdt[0], math::max(0.0001f, cwiseMax(transform.rot.dpdt) * 0.01f));
            ImGui::DragFloat(" +/- Scale", &transform.scale.dpdt[0], math::max(0.0001f, cwiseMax(transform.scale.dpdt) * 0.01f));
            transform.scale.dpdt = transform.scale.dpdt[0];
            ImGui::PopID();

            ImGui::PushID("t");
            ImGui::SliderFloat3("~ Position", &transform.trans.t[0], 0.0f, 1.0f);
            ImGui::SliderFloat3("~ Rotation", &transform.rot.t[0], 0.0f, 1.0f);
            ImGui::SliderFloat("~ Scale", &transform.scale.t[0], 0.0f, 1.0f);
            transform.scale.t = transform.scale.t[0];
            ImGui::PopID();
        }

        ImGui::TreePop();
    }
} 

