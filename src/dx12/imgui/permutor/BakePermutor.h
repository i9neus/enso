#pragma once

#include "RenderObjectStateMap.h"
#include "../shelves/IMGUIAbstractShelf.h"
#include <set>

class BakePermutor
{
public:
    BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap);

    void Clear();
    std::set<int>& GetSampleCountSet() { return m_sampleCountSet; }
    bool Prepare(const int numIterations, const std::string& templatePath, const bool disableLiveView, const bool startWithThisView);
    bool Advance();
    float GetProgress() const;
    std::vector<std::string> GenerateExportPaths() const;
    bool IsIdle() const { return m_isIdle; }

    float GetElapsedTime() const;
    float EstimateRemainingTime(const float bakeProgress) const;

private:
    void RandomiseScene();

    std::set<int>               m_sampleCountSet;
    std::set<int>::const_iterator m_sampleCountIt;
    int                         m_sampleCountIdx;
    int                         m_numIterations;
    int                         m_iterationIdx;
    int                         m_permutationIdx;
    int                         m_numPermutations;
    bool                        m_isIdle;
    bool                        m_disableLiveView;
    bool                        m_startWithThisView;
    RenderObjectStateMap::StateMap::const_iterator m_stateIt;
    int                         m_stateIdx;
    int                         m_numStates;

    std::string                 m_templatePath;
    std::vector<std::string>    m_templateTokens;

    IMGUIAbstractShelfMap& m_imguiShelves;
    std::shared_ptr<LightProbeCameraShelf>  m_lightProbeCameraShelf;
    std::shared_ptr<PerspectiveCameraShelf> m_perspectiveCameraShelf;
    std::shared_ptr<WavefrontTracerShelf> m_wavefrontTracerShelf;
    RenderObjectStateMap& m_stateMap;

    int                         m_bakeLightingMode;
    int                         m_bakeSeed;
    std::string                 m_randomDigitString;

    std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
    int                         m_totalSamples;
    int                         m_completedSamples;
};