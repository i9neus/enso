#pragma once

#include "RenderObjectStateMap.h"
#include "../shelves/IMGUIAbstractShelf.h"
#include <set>

class BakePermutor
{
public:
    struct Params
    {
        std::string probeGridTemplatePath;
        std::string renderTemplatePath; 
        std::string jsonRootPath;
        
        bool disableLiveView = false;
        bool startWithThisView = false;
        bool exportToUsd = false;
        bool exportToPng = false; 

        Cuda::vec2 gridValidityRange;
        Cuda::ivec2 kifsIterationRange;
        uint jitterFlags = 0;
        int numIterations = 0;
        int numStrata = 0; 

        Cuda::ivec2 noisyRange;
        Cuda::ivec2 minMaxReferenceSamples = Cuda::ivec2(0);
        int thumbnailCount = 0;
    };

    struct IterationData
    {
        IterationData(const int type, const Cuda::ivec2& minMax, const int mode, const bool filter) :
            bakeType(type), minMaxSamples(minMax), sampleMode(mode), filterGrids(filter) {}

        int             bakeType;
        Cuda::ivec2     minMaxSamples;
        int             sampleMode;
        bool            filterGrids;
    };

public:
    BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap, Json::Document& commandQueue);

    void                        Clear();
    std::vector<IterationData>& GetIterationList() { return m_iterationList; }
    bool                        Prepare(const Params& params);
    bool                        Advance(const bool lastBakeSucceeded);
    float                       GetProgress() const;
    bool                        IsIdle() const { return m_isIdle; }

    float                       GetElapsedTime() const;
    float                       EstimateRemainingTime(const float bakeProgress) const;

    std::string                 GeneratePNGExportPath(const std::string& renderTemplatePath, const std::string& jsonRootPath);
    std::vector<std::string>    GenerateGridExportPaths(const std::string& renderTemplatePath, const std::string& jsonRootPath);

private:
    std::vector<std::string>    TokeniseTemplatePath(const std::string& templatePath, const std::string& jsonRootPath);
    std::vector<std::string>    GenerateExportPaths(const std::vector<std::string>& templateTokens, const int numPaths) const;

    void                        RandomiseScene();
    int                         GenerateIterationList();

    Params                      m_params;

    std::vector<IterationData>                    m_iterationList;
    std::vector<IterationData>::const_iterator    m_iterationIt;
    int                         m_sampleCountIdx;
    int                         m_kifsIterationIdx;
    int                         m_iterationIdx;
    int                         m_permutationIdx;
    int                         m_numPermutations;
    bool                        m_isIdle;
    RenderObjectStateMap::StateMap::const_iterator m_stateIt;
    int                         m_stateIdx;
    int                         m_numStates;
    
    std::vector<std::string>    m_probeGridTemplateTokens;
    std::vector<std::string>    m_renderTemplateTokens;

    Json::Document&             m_commandQueue;

    IMGUIAbstractShelfMap&                      m_imguiShelves;
    std::shared_ptr<LightProbeCameraShelf>      m_lightProbeCameraShelf;
    std::shared_ptr<PerspectiveCameraShelf>     m_perspectiveCameraShelf;
    std::shared_ptr<WavefrontTracerShelf>       m_wavefrontTracerShelf;
    std::shared_ptr<KIFSShelf>                  m_kifsShelf;

    RenderObjectStateMap&                       m_stateMap;

    int                         m_bakeLightingMode;
    int                         m_bakeSeed;
    std::string                 m_randomDigitString;

    std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
    int                         m_totalSamples;
    int                         m_completedSamples;

    std::string                 m_jsonRootPath;
};