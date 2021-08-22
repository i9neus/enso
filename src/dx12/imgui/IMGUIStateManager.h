#pragma once

#include "shelves/IMGUIAbstractShelf.h"

class RenderManager;

class RenderObjectStateManager : public IMGUIElement
{
public:
    using StateMap = std::map<const std::string, std::shared_ptr<Json::Document>>;

    RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager);

    void Initialise(const Json::Node& node);
    
    void ConstructUI();

    void ReadJson();
    void WriteJson();

    bool Insert(const std::string& id, const bool overwriteIfExists);
    bool Erase(const std::string& id);
    bool Restore(const std::string& id);
    void Clear();

    const StateMap& GetStateMap() const { return m_stateMap; }

private:
    void ConstructStateManagerUI();
    void ConstructBatchProcessorUI();

    RenderManager&          m_renderManager;

    StateMap                m_stateMap;
    IMGUIAbstractShelfMap&  m_imguiShelves;

    int                     m_numPermutations;
    bool                    m_isBaking;

    std::vector<char>       m_usdPathData;
    std::string             m_stateJsonPath;


    IMGUIListBox            m_sampleCountList;
    IMGUIListBox            m_stateList;
};