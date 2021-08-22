#pragma once

#include "shelves/IMGUIAbstractShelf.h"
#include <set>

class RenderManager;
class LightProbeCameraShelf;

class BakePermutor
{
public:
    BakePermutor(IMGUIAbstractShelfMap& imguiShelves);
    
    void Clear();
    std::set<int>& GetSampleCountSet() { return m_sampleCountSet; }
    void Prepare(const int numIterations, const std::string& templatePath);
    bool Advance();
    float GetProgress() const;
    std::string GenerateExportPath() const;

private:
    std::set<int>               m_sampleCountSet;  
    std::set<int>::iterator     m_sampleCountIt;
    int                         m_numIterations;
    int                         m_iterationIdx;
    int                         m_permutationIdx;
    int                         m_numPermutations;

    std::string                 m_templatePath;

    IMGUIAbstractShelfMap&      m_imguiShelves;
    std::shared_ptr<LightProbeCameraShelf> m_lightProbeCameraShelf;

};

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
    void HandleBakeIteration();
    void ToggleBake();

    RenderManager&          m_renderManager;

    StateMap                m_stateMap;
    IMGUIAbstractShelfMap&  m_imguiShelves;
    BakePermutor            m_permutor;

    std::vector<int>        m_sampleCounts;
    int                     m_sampleCountIdx;
    int                     m_numBakePermutations; 
    int                     m_bakePermutationIdx;
    bool                    m_isBaking;
    std::string             m_usdPathTemplate;

    std::vector<char>       m_usdPathData;
    std::string             m_stateJsonPath;

    IMGUIListBox            m_sampleCountListUI;
    IMGUIListBox            m_stateListUI;
};