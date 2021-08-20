#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"
#include "generic/StdIncludes.h"
#include "generic/JsonUtils.h"

#include "kernels/math/CudaMath.cuh"
#include "kernels/CudaAsset.cuh"

#define SL ImGui::SameLine()

namespace Json { class Document; class Node; }

class IMGUIListBox
{
public:
    IMGUIListBox(const std::string& id, const std::string& addLabel, const std::string& overwriteLabel, const std::string& deleteLabel);

    void    Construct();
    void    Clear();
    void    Insert(const std::string& str);

    bool    IsSelected() const { return m_currentIdx >= 0 && m_currentIdx < m_listItems.size(); }
    std::string GetCurrentlySelectedText() const;
    int     GetCurrentlySelectedIndex() const { return m_currentIdx; }

    void    SetOnAdd(const std::function<bool(const std::string&)>& onAdd) { m_onAdd = onAdd; }
    void    SetOnOverwrite(const std::function<bool(const std::string&)>& onOverwrite) { m_onOverwrite = onOverwrite; }
    void    SetOnDelete(const std::function<bool(const std::string&)>& onDelete) { m_onDelete = onDelete; }
    void    SetOnDeleteAll(const std::function<bool()>& onDeleteAll) { m_onDeleteAll = onDeleteAll; }

private:
    std::function<bool(const std::string&)>       m_onAdd;
    std::function<bool(const std::string&)>       m_onOverwrite;
    std::function<bool(const std::string&)>       m_onDelete;
    std::function<bool()>                         m_onDeleteAll;

    std::string                 m_addLabel;
    std::string                 m_overwriteLabel;
    std::string                 m_deleteLabel;

    std::string                 m_listBoxID;
    std::list<std::string>      m_listItems;
    std::vector<char>           m_newItemIDData;
    int                         m_currentIdx;
};

class IMGUIElement
{
public:
    IMGUIElement() = default;

protected:
    void ConstructTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable);
    void ConstructComboBox(const std::string& name, const std::vector<std::string>& elements, int& selected);
    
    template<typename T>
    void ConstructMappedListBox(const std::string& id, const std::map<const std::string, T>& container, std::string& selectedStr, int& selectedIdx)
    {
        if (!ImGui::BeginListBox(id)) { return; }

        StateMap::const_iterator it = container.begin();
        for (int n = 0; n < container.size(); n++, ++it)
        {
            const bool isSelected = (selectedIdx == n);
            if (ImGui::Selectable(it->first.c_str(), isSelected))
            {
                selectedStr = it->first;
                selectedIdx = n;
            }
            if (isSelected) { ImGui::SetItemDefaultFocus(); }
        }
        ImGui::EndListBox();
    }
};
