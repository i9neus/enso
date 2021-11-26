#pragma once

#include "BakePermutor.h"
#include "../shelves/IMGUIAbstractShelf.h"
#include <set>

class RenderObjectStateManager : public IMGUIElement
{
public:
    RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager, const Json::Document& renderStateJson, Json::Document& commandQueue);
    ~RenderObjectStateManager();

    void Rebuild(const Json::Node& node);
    void Initialise(HWND hwnd);
    
    void ConstructUI();
    void HandleBakeIteration();
    void UpdateRenderState(const Json::Document& json);

    void DeserialiseJson();
    void SerialiseJson() const;

    IMGUIDirtiness GetDirtiness() const { return m_dirtiness; }
    void MakeClean() { m_dirtiness = IMGUIDirtiness::kClean; }

private:
    void ConstructSceneManagerUI();
    void ConstructStateManagerUI();
    void ConstructBatchProcessorUI();
    void EnqueueExportViewport();
    void EnqueueBake();
    void ScanForSceneFiles();
    void ParseRenderStateJson();

    RenderObjectStateMap    m_stateMap;

    RenderManager&          m_renderManager;

    IMGUIAbstractShelfMap&  m_imguiShelves;
    BakePermutor            m_permutor;
    const Json::Document&   m_renderStateJson;
    Json::Document&         m_commandQueue;
    
    std::vector<std::string> m_sceneFilePathList;
    std::vector<std::string> m_sceneFileNameList;
    int                     m_sceneListIdx;

    int                     m_renderState;
    bool                    m_isBaking;
    bool                    m_isBatchRunning;

    std::vector<int>        m_sampleCounts;
    int                     m_sampleCountIdx;
    int                     m_numBakeIterations;
    int                     m_bakePermutationIdx;
    float                   m_bakeProgress;
    std::string             m_usdPathTemplate;

    std::vector<char>       m_usdPathUIData;
    uint                    m_stateFlags;
    std::string             m_stateJsonPath;
    bool                    m_exportToUSD;
    bool                    m_disableLiveView;
    bool                    m_startWithThisView;
    bool                    m_shutdownOnComplete;
    Cuda::ivec2             m_noisySampleRange;
    int                     m_referenceSamples;
    int                     m_numStrata;
    Cuda::vec2              m_gridValidityRange;
    Cuda::ivec2             m_kifsIterationRange;
    
    float                   m_gridFitness;

    IMGUIListBox            m_stateListUI;

    HWND                    m_hWnd;
    IMGUIDirtiness          m_dirtiness;
};