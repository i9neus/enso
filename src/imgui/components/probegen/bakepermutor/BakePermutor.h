#pragma once

#include "RenderObjectStateMap.h"
#include "../shelves/IMGUIAbstractShelf.h"
#include <set>
#include "kernels/math/CudaColourUtils.cuh"

class LambertBRDFShelf;

class BakePermutor
{
public:
    struct Params
    {
        std::string probeGridTrainTemplatePath;
        std::string probeGridTestTemplatePath;
        std::string renderTemplatePath; 
        std::string jsonRootPath;
        
        bool disableLiveView = false;
        bool startWithThisView = false;
        bool exportToUSD = false;
        bool exportToPNG = false; 

        Cuda::vec2 gridValidityRange;
        Cuda::ivec2 kifsIterationRange;
        uint jitterFlags = 0;
        int numIterations = 0;
        int startIteration = 0;
        Cuda::ivec2 trainTestRatio = Cuda::ivec2(4, 1);
        int numStrata = 0; 
        int noisyColourSpace = Cuda::kColourSpaceRGB;
        int referenceColourSpace = Cuda::kColourSpaceRGB;

        Cuda::ivec2 noisyRange;
        Cuda::ivec2 minMaxReferenceSamples = Cuda::ivec2(0);
        int thumbnailCount = 0;
    };

    struct BatchProgress
    {     
        bool isRunning = false;
        float totalProgress = 0.0;
        Cuda::ivec2 permutationRange;
        int numSucceeded = 0;
        int numFailed = 0;
        std::string bakeType;   
        struct
        {
            float elapsed = 0.0;
            float remaining = 0.0;
        }
        time;
    };

    struct IterationData
    {
        IterationData(const int type, const std::string& desc, const Cuda::ivec2& minMax, const int mode, const bool filter, const int colourSpace) :
            bakeType(type), description(desc), minMaxSamples(minMax), sampleMode(mode), filterGrids(filter), outputColourSpace(colourSpace) {}

        int             bakeType;
        std::string     description;
        Cuda::ivec2     minMaxSamples;
        int             sampleMode;
        int             outputColourSpace;
        bool            filterGrids;
    };

public:
    BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap, Json::Document& commandQueue);

    void                        Clear();
    std::vector<IterationData>& GetIterationList() { return m_iterationList; }
    bool                        Prepare(const Params& params);
    bool                        Advance(const bool lastBakeSucceeded);
    BatchProgress               GetBatchProgress(const float bakeProgress) const;
    bool                        IsIdle() const { return m_isIdle; }

    float                       GetElapsedTime() const;

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
    int                         m_numSucceeded, m_numFailed;
    int                         m_bakeType;
    
    std::vector<std::string>    m_probeGridTrainTemplateTokens;
    std::vector<std::string>    m_probeGridTestTemplateTokens;
    std::vector<std::string>    m_renderTemplateTokens;

    Json::Document&             m_commandQueue;

    IMGUIAbstractShelfMap&                      m_imguiShelves;
    std::shared_ptr<LightProbeCameraShelf>      m_lightProbeCameraShelf;
    std::shared_ptr<PerspectiveCameraShelf>     m_perspectiveCameraShelf;
    std::shared_ptr<WavefrontTracerShelf>       m_wavefrontTracerShelf;
    std::shared_ptr<LambertBRDFShelf>           m_lambertShelf;
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