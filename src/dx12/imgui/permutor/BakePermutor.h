#pragma once

#include "RenderObjectStateMap.h"
#include "../shelves/IMGUIAbstractShelf.h"
#include <set>

class BakePermutor
{
public:
    BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap);

    void                        Clear();
    std::set<int>&              GetSampleCountSet() { return m_sampleCountSet; }
    bool                        Prepare(const int numIterations, const Cuda::ivec2& noisyRange, const int referenceCount, const int numStrata, const Cuda::ivec2& kifsIterationRange);
    bool                        Initialise(const std::string& templatePath, const std::string& jsonRootPath, const bool disableLiveView, const bool startWithThisView);
    void                        ParseTemplatePath(const std::string& templatePath, const std::string& jsonRootPath);
    bool                        Advance();
    float                       GetProgress() const;
    std::vector<std::string>    GenerateExportPaths() const;
    bool                        IsIdle() const { return m_isIdle; }

    float                       GetElapsedTime() const;
    float                       EstimateRemainingTime(const float bakeProgress) const;

private:
    void                        RandomiseScene();
    int                         GenerateStratifiedSampleCountSet();

    std::set<int>               m_sampleCountSet;
    std::set<int>::const_iterator m_sampleCountIt;
    int                         m_sampleCountIdx;
    int                         m_numIterations;
    Cuda::ivec2                 m_noisyRange;
    Cuda::ivec2                 m_kifsIterationRange;
    int                         m_kifsIterationIdx;
    int                         m_referenceCount;
    int                         m_numStrata;
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
    std::string                 m_jsonRootPath;
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