#pragma once

#include "shelves/IMGUIAbstractShelf.h"

class RenderObjectStateManager
{
public:
    using StateMap = std::map<const std::string, std::shared_ptr<Json::Document>>;

    RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves);

    void Initialise(const Json::Node& node);
    
    void ConstructIMGUI();

    void ReadJson();
    void WriteJson();

    void Insert(const std::string& id, const bool overwriteIfExists);
    void Erase(const std::string& id);
    void Restore(const std::string& id);

    const StateMap& GetStateMap() const { return m_stateMap; }

private:
    StateMap                m_stateMap;
    IMGUIAbstractShelfMap&  m_imguiShelves;

    int                     m_stateListCurrentIdx;
    std::string             m_stateListCurrentId;
    std::vector<char>       m_stateIDData;
    std::string             m_stateJsonPath;
};