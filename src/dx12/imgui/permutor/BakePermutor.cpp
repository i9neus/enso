#include "BakePermutor.h"
#include "generic/FilesystemUtils.h"
#include "manager/RenderManager.h"

#include "../shelves/IMGUIBxDFShelves.h"
#include "../shelves/IMGUICameraShelves.h"
#include "../shelves/IMGUIFilterShelves.h"
#include "../shelves/IMGUIIntegratorShelves.h"
#include "../shelves/IMGUILightShelves.h"
#include "../shelves/IMGUIMaterialShelves.h"
#include "../shelves/IMGUITracableShelves.h"

#include "kernels/cameras/CudaLightProbeCamera.cuh"
#include "kernels/CudaWavefrontTracer.cuh"

#include <random>

BakePermutor::BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap, Json::Document& commandQueue) :
    m_imguiShelves(imguiShelves),
    m_stateMap(stateMap),
    m_commandQueue(commandQueue)
{
    m_isIdle = true;
    
    Clear();
}

void BakePermutor::Clear()
{
    m_iterationList.clear();
    m_iterationIt = m_iterationList.end();
    m_stateIt = m_stateMap.GetStateData().end();
    m_iterationIdx = -1;
    m_kifsIterationIdx = 0;
    m_numPermutations = 0;
    m_permutationIdx = 0;
    m_totalSamples = 0;
    m_completedSamples = 0;
}

// Populate the sample count set with randomly stratified values
int BakePermutor::GenerateIterationList()
{
    m_iterationList.clear();
    int totalSamples = 0;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> rng(0, std::numeric_limits<int>::max());

    std::set<int> sampleCountSet;
    for (int idx = 0; idx < m_params.numStrata; ++idx)
    {
        const int stratumLower = m_params.noisyRange.x + (m_params.noisyRange.y - m_params.noisyRange.x) * idx / m_params.numStrata;
        const int stratumUpper = m_params.noisyRange.x + (m_params.noisyRange.y - m_params.noisyRange.x) * (idx + 1) / m_params.numStrata;
        const int stratumRange = stratumUpper - stratumLower;

        Assert(stratumRange >= 0);
        if (stratumRange == 0) { continue; }
        else if (stratumRange == 1) { m_iterationList.emplace_back(kBakeTypeProbeGrid, Cuda::ivec2(0, stratumLower), Cuda::kCameraSamplingFixed, false); continue; }
        
        // Generate some candidate samples and reject if they're already in the list.
        for (int tries = 0; tries < 10; ++tries)
        {
            const int candidate = stratumLower + rng(mt) % stratumRange;
            if (sampleCountSet.find(candidate) == sampleCountSet.end())
            {
                m_iterationList.emplace_back(kBakeTypeProbeGrid, Cuda::ivec2(0, candidate), Cuda::kCameraSamplingFixed, false);
                sampleCountSet.emplace(candidate);
                totalSamples += candidate;
                break;
            }
        }
    }
    
    // Add the reference sample count and the thumbnail render
    m_iterationList.emplace_back(kBakeTypeProbeGrid, m_params.minMaxReferenceSamples, Cuda::kCameraSamplingAdaptiveRelative, true);
    m_iterationList.emplace_back(kBakeTypeRender, Cuda::ivec2(0, m_params.thumbnailCount), Cuda::kCameraSamplingFixed, false);

    m_iterationIt = m_iterationList.cbegin();
    return totalSamples;
}

bool BakePermutor::Prepare(const BakePermutor::Params& params)
{
    m_params = params;

    // Decompose the template paths into tokens
    ParseTemplatePath(m_params.probeGridTemplatePath, m_params.jsonRootPath, m_probeGridTemplateTokens);
    ParseTemplatePath(m_params.renderTemplatePath, m_params.jsonRootPath, m_renderTemplateTokens);

    // Get the handles to the required shelves
    for (auto& shelf : m_imguiShelves)
    {
        if (!m_lightProbeCameraShelf)
        {
            m_lightProbeCameraShelf = std::dynamic_pointer_cast<LightProbeCameraShelf>(shelf.second);
        }
        if (!m_perspectiveCameraShelf)
        {
            m_perspectiveCameraShelf = std::dynamic_pointer_cast<PerspectiveCameraShelf>(shelf.second);
        }
        if (!m_wavefrontTracerShelf)
        {
            m_wavefrontTracerShelf = std::dynamic_pointer_cast<WavefrontTracerShelf>(shelf.second);
        }
        if (!m_kifsShelf)
        {
            m_kifsShelf = std::dynamic_pointer_cast<KIFSShelf>(shelf.second);
        }
    }

    if (!m_lightProbeCameraShelf)
    {
        Log::Warning("Warning: bake permutor was unable to find an instance of LightProbeCameraShelf.\n");
        return false;
    }
    if (!m_wavefrontTracerShelf)
    {
        Log::Warning("Warning: bake permutor was unable to find an instance of WavefrontTracerShelf.\n");
        return false;
    }
    if (!m_kifsShelf)
    {
        Log::Warning("Warning: bake permutor was unable to find an instance of KIFSShelf.\n");
        return false;
    }

    m_numStates = m_stateMap.GetNumPermutableStates();
    if (m_numStates == 0)
    {
        Log::Error("Error: can't start bake because no states are enabled.\n");
        return false;
    }

    // Populate the sample count set with a random collection of values
    m_totalSamples = GenerateIterationList();
    m_completedSamples = 0;
    m_iterationIdx = 0;
    m_stateIdx = 0;
    m_kifsIterationIdx = m_params.kifsIterationRange[0];
    m_permutationIdx = -1;
    m_stateIt = m_stateMap.GetFirstPermutableState();
    m_isIdle = false;

    m_numPermutations = m_iterationList.size() * m_params.numIterations * m_numStates * (1 + m_params.kifsIterationRange[1] - m_params.kifsIterationRange[0]);
    m_totalSamples *= m_params.numIterations * m_numStates;

    m_startTime = std::chrono::high_resolution_clock::now();

    // Restore the state pointed to by the state iterator
    m_stateMap.Restore(*m_stateIt);

    return true;
}

void BakePermutor::ParseTemplatePath(const std::string& templatePath, const std::string& jsonRootPath, std::vector<std::string>& templateTokens)
{
    m_jsonRootPath = jsonRootPath;
    
    // Parse the template path
    templateTokens.clear();
    Lexer lex(templatePath);
    while (lex)
    {
        std::string newToken;
        if (lex.ParseToken(newToken, [](const char c) { return c != '{'; }))
        {
            templateTokens.push_back(newToken);
        }
        if (lex && *lex == '{')
        {
            if (lex.ParseToken(newToken, [](const char c) { return c != '}'; }, Lexer::IncludeDelimiter))
            {
                templateTokens.push_back(newToken);
            }
        }
    }
}

void BakePermutor::RandomiseScene()
{
    if (m_params.startWithThisView && m_iterationIdx == 0) { return; }

    Assert(m_stateIt != m_stateMap.GetStateData().end());

    // Randomise the shelves depending on which types are selected
    for (auto& shelf : m_imguiShelves)
    {
        shelf.second->Jitter(m_stateIt->second.flags, Cuda::kJitterRandomise);
    }
}

bool BakePermutor::Advance(const bool lastBakeSucceeded)
{    
    // Can we advance? 
    if (m_isIdle ||
        m_stateIt == m_stateMap.GetStateData().end() ||
        m_iterationIt == m_iterationList.cend() ||
        m_iterationIdx >= m_params.numIterations ||
        !m_lightProbeCameraShelf || 
        !m_wavefrontTracerShelf ||
        !m_perspectiveCameraShelf)
    {
        if (m_params.disableLiveView)
        {
            m_perspectiveCameraShelf->GetParamsObject().camera.isActive = true;
        }
        return false;
    }

    // Don't increment on the first iteration as this was done at start-up
    if (m_permutationIdx >= 0)
    {
        // Samples set. If the last bake failed then abort the sequence and generate a new scene.
        m_completedSamples += m_iterationIt->minMaxSamples.y;
        if (++m_iterationIt == m_iterationList.cend() || !lastBakeSucceeded)
        {
            GenerateIterationList();

            // KIFS iteration set...
            if (++m_kifsIterationIdx == m_params.kifsIterationRange[1] + 1)
            {
                m_kifsIterationIdx = m_params.kifsIterationRange[0];                

                // Random iteration 
                if (++m_iterationIdx == m_params.numIterations)
                {
                    m_iterationIdx = 0;
                    do
                    {
                        ++m_stateIdx;
                        if (++m_stateIt == m_stateMap.GetStateData().end())
                        {
                            m_isIdle = true;
                            return false;
                        }
                    } while (!(m_stateIt->second.flags & kStateEnabled));

                    // Restore the next state in the sequence
                    m_stateMap.Restore(*m_stateIt);
                }

                // Randomise all the shelves
                RandomiseScene();
            }
        }
    }
    ++m_permutationIdx;
    
    // Get the handles to the camera objects
    auto& probeParams = m_lightProbeCameraShelf->GetParamsObject();
    auto& perspParams = m_perspectiveCameraShelf->GetParamsObject();

    // Set the sample counts depending on the type of bake we're doing
    if (m_iterationIt->bakeType == kBakeTypeProbeGrid)
    {
        // Set the sample count on the light probe camera          
        probeParams.camera.minMaxSamples = m_iterationIt->minMaxSamples;
        probeParams.camera.isActive = true;
        probeParams.camera.samplingMode = m_iterationIt->sampleMode;
        probeParams.filterGrids = m_iterationIt->filterGrids;

        // Always randomise the probe shelf to generate a new seed
        m_lightProbeCameraShelf->Randomise();

        // Deactivate the perspective camera if necessary
        perspParams.camera.isActive = !m_params.disableLiveView;
    }
    else if (m_iterationIt->bakeType == kBakeTypeRender)
    {
        perspParams.camera.minMaxSamples = m_iterationIt->minMaxSamples;
        perspParams.camera.overrides.maxDepth = 3;
        perspParams.camera.isActive = true;
        
        // Always deactivate the probe grid when baking
        probeParams.camera.isActive = false;
    }
    else { Assert(false); }

    // Set the iteration count on the KIFS object
    auto& kifsParams = m_kifsShelf->GetParamsObject(); 
    kifsParams.numIterations = m_kifsIterationIdx;

    // Set the shading mode to full illumination
    m_wavefrontTracerShelf->GetParamsObject().shadingMode = Cuda::kShadeFull;

    // Get some parameters that we'll need to generate the file paths
    m_bakeLightingMode == probeParams.lightingMode;
    m_bakeSeed = probeParams.camera.seed;   

    // Enqueue the bake
    Json::Node bakeJson = m_commandQueue.AddChildObject("bake");
    bakeJson.AddValue("action", "start");
    if (m_iterationIt->bakeType == kBakeTypeProbeGrid)
    {
        bakeJson.AddValue("type", "probeGrid");
        bakeJson.AddValue("exportToUSD", m_params.exportToUsd);
        bakeJson.AddValue("minGridValidity", m_params.gridValidityRange.x);
        bakeJson.AddValue("maxGridValidity", m_params.gridValidityRange.y);
        bakeJson.AddArray("usdExportPaths", GenerateExportPaths(m_probeGridTemplateTokens, 2));
    }
    else
    {
        bakeJson.AddValue("type", "render");
        bakeJson.AddValue("pngExportPath", GenerateExportPaths(m_renderTemplateTokens, 1)[0]);
    }         

    // Print some stats
    Log::Write("Starting bake permutation %i of %i...\n", m_permutationIdx + 1, m_numPermutations);
    Log::Indent indent;
    Log::Write("Sample count: %i", m_iterationIt->minMaxSamples.y);
    Log::Write("KIFS iteration: %i", m_kifsIterationIdx);
    Log::Write("Random iteration: %i", m_iterationIdx);
    Log::Write("State: %i of %i", m_stateIdx, m_stateMap.GetStateData().size());
    return true;
}

 std::string BakePermutor::GeneratePNGExportPath(const std::string& renderTemplatePath, const std::string& jsonRootPath)
 { 
     ParseTemplatePath(renderTemplatePath, jsonRootPath, m_renderTemplateTokens);

     return GenerateExportPaths(m_renderTemplateTokens, 1)[0]; 
 }

std::vector<std::string> BakePermutor::GenerateExportPaths(const std::vector<std::string>& templateTokens, const int numPaths) const
{
    // Naming convention:
    // SceneNameXXX.random10digits.6digitsSampleCount.usd

    AssertMsg(!templateTokens.empty(), "Can't generate paths. Token list is empty.");
    Assert(numPaths >= 1);

    std::vector<std::string> exportPaths(numPaths);
    for (int pathIdx = 0; pathIdx < numPaths; pathIdx++)
    {
        auto& path = exportPaths[pathIdx];
        for (const auto& token : templateTokens)
        {
            Assert(!token.empty());
            if (token[0] != '{')
            {
                path += token;
                continue;
            }

            if (token == "{$COMPONENT}")
            {
                path += (m_bakeLightingMode == Cuda::kBakeLightingCombined) ? "combined" : ((pathIdx == 0) ? "direct" : "indirect");
            }
            else if (token == "{$RANDOM_DIGITS}")
            {
                // 10 random digit string
                std::random_device rd;
                std::mt19937 mt(rd());
                std::uniform_int_distribution<> rng(0, std::numeric_limits<int>::max());
                path += tfm::format("%010.i", rng(mt) % 10000000000);
            }
            else if (token == "{$ITERATION}") { path += tfm::format("%06.i", m_iterationIdx); }
            else if (token == "{$KIFS}") { path += tfm::format("%i", m_kifsIterationIdx); }
            else if (token == "{$SAMPLE_COUNT}" && m_iterationIt != m_iterationList.end()) { path += tfm::format("%06.i", m_iterationIt->minMaxSamples.y); }
            else if (token == "{$PERMUTATION}") { path += tfm::format("%i", m_permutationIdx); }
            else if (token == "{$SEED}") { path += tfm::format("%.6i", m_bakeSeed % 1000000); }
            else if (token == "{$STATE}" && m_stateIt != m_stateMap.GetStateData().end()) 
            { 
                path += m_stateIt->first; 
            }
            else { path += token; }
        }

        if (!IsAbsolutePath(path))
        {
            path = MakeAbsolutePath(GetParentDirectory(m_jsonRootPath), path);
        }
    }

    return exportPaths;
}

float BakePermutor::GetProgress() const
{    
    return m_isIdle ? 0.0f : float(m_completedSamples) / float(m_totalSamples);
}

float BakePermutor::GetElapsedTime() const
{
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_startTime).count();
}

float BakePermutor::EstimateRemainingTime(const float bakeProgress) const
{
    const int currentSampleCount = (m_iterationIt == m_iterationList.end()) ? 0 : m_iterationIt->minMaxSamples.y;
    const float progress = (m_completedSamples + bakeProgress * currentSampleCount) / float(m_totalSamples);

    return GetElapsedTime() * (1.0f - progress) / max(1e-6f, progress);
}