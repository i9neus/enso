#pragma once

#include "BakePermutor.h"
#include "../shelves/IMGUIAbstractShelf.h"
#include <set>

class RenderObjectStateManager : public IMGUIElement
{
public:
    RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager);
    ~RenderObjectStateManager();

    void Initialise(const Json::Node& node, HWND hwnd);
    
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
    int                     m_numBakeIterations;
    int                     m_bakePermutationIdx;
    bool                    m_isBaking;
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
    float                   m_minViableValidity;

    IMGUIListBox            m_stateListUI;

    HWND                    m_hWnd;
};