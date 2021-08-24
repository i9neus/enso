#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"
#include "generic/StdIncludes.h"
#include "generic/JsonUtils.h"

#include "kernels/math/CudaMath.cuh"
#include "kernels/CudaAsset.cuh"

#define SL ImGui::SameLine()

namespace Json { class Document; class Node; }

class UIStyle
{
public:
    UIStyle(const int shelfIdx);
    ~UIStyle();
};


class IMGUIListBox
{
public:
    IMGUIListBox(const std::string& id, const std::string& addLabel, const std::string& overwriteLabel, const std::string& deleteLabel);

    void    Construct();
    void    Clear();
    void    Insert(const std::string& str);
    const std::list<std::string>& GetListItems() const { return m_listItems; }

    bool    IsSelected() const { return m_currentIdx >= 0 && m_currentIdx < m_listItems.size(); }
    std::string GetCurrentlySelectedText() const;
    int     GetCurrentlySelectedIndex() const { return m_currentIdx; }

    void    SetOnAdd(const std::function<bool(const std::string&)>& onAdd) { m_onAdd = onAdd; }
    void    SetOnOverwrite(const std::function<bool(const std::string&)>& onOverwrite) { m_onOverwrite = onOverwrite; }
    void    SetOnDelete(const std::function<bool(const std::string&)>& onDelete) { m_onDelete = onDelete; }
    void    SetOnDeleteAll(const std::function<bool()>& onDeleteAll) { m_onDeleteAll = onDeleteAll; }
    void    SetOnSelect(const std::function<void(const std::string&)>& onSelectItem) { m_onSelectItem = onSelectItem; }

private:
    std::function<bool(const std::string&)>       m_onAdd;
    std::function<bool(const std::string&)>       m_onOverwrite;
    std::function<bool(const std::string&)>       m_onDelete;
    std::function<bool()>                         m_onDeleteAll;
    std::function<void(const std::string&)>       m_onSelectItem;

    std::string                 m_addLabel;
    std::string                 m_overwriteLabel;
    std::string                 m_deleteLabel;

    std::string                 m_listBoxID;
    std::list<std::string>      m_listItems;
    std::vector<char>           m_newItemIDData;
    int                         m_currentIdx;
    int                         m_lastIdx;
};

class IMGUIJitteredColourPicker
{
public:
    IMGUIJitteredColourPicker(Cuda::JitterableVec3& param, const std::string& id) : m_param(param), m_id(id) {}

    void Construct();
    void Update();

private:
    Cuda::vec3                  m_hsv[2];
    Cuda::vec3                  m_jitter;
    Cuda::JitterableVec3&       m_param;
    std::string                 m_id;
};

class IMGUIJitteredFlagArray
{
public: 
    IMGUIJitteredFlagArray(Cuda::JitterableFlags& param, const std::string& id);

    void Initialise(const std::vector<std::string>& flagLabels);
    void Construct();
    void Update();

private:
    enum StateFlag : int { kFlagOff, kFlagOn, kFlagRnd };

    std::vector<std::string>            m_flagLabels;
    std::array<std::vector<bool>, 2>    m_flags;
    float                               m_t;
    std::string                         m_id;
    std::string                         m_pId, m_dpdtId, m_tId;
    std::vector<int>                    m_states;
    int                                 m_numBits;
    const std::array<std::string, 3>    m_switchLabels;

    Cuda::JitterableFlags&              m_param;

};

class IMGUIJitteredParameterTable
{
public:
    struct Element
    {
        Element() = default;
        Element(const std::string& l, Cuda::JitterableFloat& p, const Cuda::vec3& r) : label(l), param(&p), range(r) {}

        std::string             label;
        Cuda::JitterableFloat*  param;
        Cuda::vec3              range;
    };
public:
    IMGUIJitteredParameterTable(const std::string& id) : m_id(id) {}

    void Push(const std::string& label, Cuda::JitterableFloat& param, const Cuda::vec3& range);
    void Construct();

private:
    std::vector<Element>                m_params;
    std::string                         m_id;
};

class IMGUIJitteredParameter : public IMGUIJitteredParameterTable
{
public:
    IMGUIJitteredParameter(const std::string& id) : IMGUIJitteredParameterTable(id) {}
    IMGUIJitteredParameter(Cuda::JitterableFloat& param, const std::string& label, const Cuda::vec3& range) : IMGUIJitteredParameterTable(label) 
    {
        IMGUIJitteredParameterTable::Push(label, param, range);
    }
};

class IMGUIElement
{
public:
    IMGUIElement() = default;

protected:
    void ConstructJitteredTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable);
    void ConstructJitteredFloat(Cuda::JitterableFloat& value);
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
