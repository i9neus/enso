#pragma once

#include "shelves/IMGUIAbstractShelf.h"
#include <set>

class RenderManager;
class LightProbeCameraShelf;
class PerspectiveCameraShelf;

class RenderObjectStateMap
{
public:
    struct StateObject
    {
        StateObject() : isPermutable(true) {}
        StateObject(std::shared_ptr<Json::Document> json_, bool isPermutable_) : json(json_), isPermutable(isPermutable_) {}

        std::shared_ptr<Json::Document>     json;
        bool                                isPermutable;
    };

    using StateMap = std::map<const std::string, StateObject>;

    RenderObjectStateMap(IMGUIAbstractShelfMap& imguiShelves) : m_imguiShelves(imguiShelves) {}
    ~RenderObjectStateMap() = default;

    void FromJson(const Json::Node& node, const int flags);
    void ToJson(Json::Node& node) const;

    bool Insert(const std::string& id, const bool isPermutable, const bool overwriteIfExists);
    bool Erase(const std::string& id);
    bool Restore(const std::string& id);
    bool Restore(const std::pair<std::string, StateObject>& it);
    void Clear();
    int GetNumPermutableStates() const;
    RenderObjectStateMap::StateMap::const_iterator GetFirstPermutableState() const;

    inline const StateMap& GetStateData() const { return m_stateMap; }

private:
    StateMap                m_stateMap;  
    IMGUIAbstractShelfMap&  m_imguiShelves;
};

class BakePermutor
{
public:
    BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap);

    void Clear();
    std::set<int>& GetSampleCountSet() { return m_sampleCountSet; }
    void Prepare(const int numIterations, const std::string& templatePath, const bool disableLiveView);
    bool Advance();
    float GetProgress() const;
    std::vector<std::string> GenerateExportPaths() const;
    bool IsIdle() const { return m_isIdle; }

    float GetElapsedTime() const;
    float EstimateRemainingTime(const float bakeProgress) const;

private:
    std::set<int>               m_sampleCountSet;
    std::set<int>::const_iterator m_sampleCountIt;
    int                         m_sampleCountIdx;
    int                         m_numIterations;
    int                         m_iterationIdx;
    int                         m_permutationIdx;
    int                         m_numPermutations;
    bool                        m_isIdle;
    bool                        m_disableLiveView;
    RenderObjectStateMap::StateMap::const_iterator m_stateIt;
    int                         m_stateIdx;
    int                         m_numStates;

    std::string                 m_templatePath;
    std::vector<std::string>    m_templateTokens;

    IMGUIAbstractShelfMap&      m_imguiShelves;
    std::shared_ptr<LightProbeCameraShelf>  m_lightProbeCameraShelf;
    std::shared_ptr<PerspectiveCameraShelf> m_perspectiveCameraShelf;
    RenderObjectStateMap&       m_stateMap;

    int                         m_bakeLightingMode;
    int                         m_bakeSeed;
    std::string                 m_randomDigitString;

    std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
    int                         m_totalSamples;
    int                         m_completedSamples;
};

class RenderObjectStateManager : public IMGUIElement
{
public:
    RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager);
    ~RenderObjectStateManager();

    void Initialise(const Json::Node& node);
    
    void ConstructUI();

    void DeserialiseJson();
    void SerialiseJson() const;

private:
    void ConstructStateManagerUI();
    void ConstructBatchProcessorUI();
    void HandleBakeIteration();
    void ToggleBake();

    RenderObjectStateMap    m_stateMap;

    RenderManager&          m_renderManager;

    IMGUIAbstractShelfMap&  m_imguiShelves;
    BakePermutor            m_permutor;

    std::vector<int>        m_sampleCounts;
    int                     m_sampleCountIdx;
    int                     m_numBakePermutations; 
    int                     m_bakePermutationIdx;
    bool                    m_isBaking;
    std::string             m_usdPathTemplate;

    std::vector<char>       m_usdPathUIData;
    bool                    m_isPermutableUI;
    std::string             m_stateJsonPath;
    bool                    m_exportToUSD;
    bool                    m_disableLiveView;

    IMGUIListBox            m_sampleCountListUI;
    IMGUIListBox            m_stateListUI;
};