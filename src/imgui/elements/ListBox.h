#pragma once

#include "IMGUIUtils.h"

class IMGUIListBox
{
public:
    IMGUIListBox(const std::string& id, const std::string& addLabel = "", const std::string& overwriteLabel = "", const std::string& deleteLabel = "", const std::string& deleteAllLabel = "");

    void    Construct();
    void    Clear();
    void    Insert(const std::string& str);
    const std::list<std::string>& GetListItems() const { return m_listItems; }
    void SetListItems(const std::vector<std::string>& items);

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
    std::string                 m_deleteAllLabel;

    std::string                 m_listBoxID;
    std::list<std::string>      m_listItems;
    std::vector<char>           m_newItemIDData;
    int                         m_currentIdx;
    int                         m_lastIdx;
};