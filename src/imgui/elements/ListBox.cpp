#include "IMGUIListBox.h"

IMGUIListBox::IMGUIListBox(const std::string& id, const std::string& addLabel, const std::string& overwriteLabel,
    const std::string& deleteLabel, const std::string& deleteAllLabel) :
    m_listBoxID(id),
    m_addLabel(addLabel),
    m_overwriteLabel(overwriteLabel),
    m_deleteLabel(deleteLabel),
    m_deleteAllLabel(deleteAllLabel),
    m_currentIdx(-1),
    m_lastIdx(-1)
{
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
                if (m_onSelectItem && m_currentIdx != m_lastIdx)
                {
                    m_onSelectItem(*it);
                    m_lastIdx = m_currentIdx;
                }
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
            if (m_onOverwrite)
            {
                m_onOverwrite(*it);
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
    SL;
    //Erase a saved state from the container
    if (!m_deleteAllLabel.empty() && ImGui::Button(m_deleteAllLabel.c_str()))
    {
        if (!m_onDeleteAll || m_onDeleteAll())
        {
            Clear();
        }
    }

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

void IMGUIListBox::SetListItems(const std::vector<std::string>& newItems)
{
    m_listItems.clear();
    m_listItems.insert(m_listItems.begin(), newItems.begin(), newItems.end());
}

std::string IMGUIListBox::GetCurrentlySelectedText() const
{
    if (m_currentIdx < 0) { return ""; }

    std::list<std::string>::const_iterator it = m_listItems.begin();
    std::advance(it, m_currentIdx);
    return (it != m_listItems.end()) ? *it : "";
}