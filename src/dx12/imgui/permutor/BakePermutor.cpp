#include "BakePermutor.h"
#include "generic/FilesystemUtils.h"
#include "manager/RenderManager.h"
#include "../shelves/IMGUIShelves.h"

#include "kernels/cameras/CudaLightProbeCamera.cuh"
#include "kernels/CudaWavefrontTracer.cuh"

#include <random>

BakePermutor::BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap) :
    m_imguiShelves(imguiShelves),
    m_stateMap(stateMap)
{
    Clear();
}

void BakePermutor::Clear()
{
    m_sampleCountSet.clear();
    m_stateIt = m_stateMap.GetStateData().end();
    m_numIterations = 0;
    m_iterationIdx = -1;
    m_numPermutations = 0;
    m_permutationIdx = 0;
    m_totalSamples = 0;
    m_completedSamples = 0;
    m_templateTokens.clear();
    m_isIdle = true;
    m_disableLiveView = true;
    m_startWithThisView = false;
}

bool BakePermutor::Prepare(const int numIterations, const std::string& templatePath, const bool disableLiveView, const bool startWithThisView)
{
    m_numStates = m_stateMap.GetNumPermutableStates();
    if (m_numStates == 0)
    {
        Log::Error("Error: can't start bake because no states are enabled.\n");
        return false;
    }

    m_totalSamples = m_completedSamples = 0;
    m_sampleCountIt = m_sampleCountSet.cbegin();
    m_totalSamples = 0;
    m_numIterations = numIterations;
    m_iterationIdx = 0;
    m_stateIdx = 0;
    m_numPermutations = m_sampleCountSet.size() * m_numIterations * m_numStates;
    m_permutationIdx = -1;
    m_templatePath = templatePath;
    m_stateIt = m_stateMap.GetFirstPermutableState();
    m_isIdle = false;
    m_disableLiveView = disableLiveView;
    m_startWithThisView = startWithThisView;

    for (auto element : m_sampleCountSet) { m_totalSamples += element; }

    m_templateTokens.clear();
    Lexer lex(templatePath);
    while (lex)
    {
        std::string newToken;
        if (lex.ParseToken(newToken, [](const char c) { return c != '{'; }))
        {
            m_templateTokens.push_back(newToken);
        }
        if (lex && *lex == '{')
        {
            if (lex.ParseToken(newToken, [](const char c) { return c != '}'; }, Lexer::IncludeDelimiter))
            {
                m_templateTokens.push_back(newToken);
            }
        }
    }

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
    }

    if (!m_lightProbeCameraShelf)
    {
        Log::Debug("Error: bake permutor was unable to find an instance of LightProbeCameraShelf.\n");
        return false;
    }
    if (!m_wavefrontTracerShelf)
    {
        Log::Debug("Error: bake permutor was unable to find an instance of WavefrontTracerShelf.\n");
        return false;
    }

    m_startTime = std::chrono::high_resolution_clock::now();

    // Restore the state pointed
    m_stateMap.Restore(*m_stateIt);

    return true;
}

void BakePermutor::RandomiseScene()
{
    if (m_startWithThisView && m_iterationIdx == 0) { return; }

    Assert(m_stateIt != m_stateMap.GetStateData().end());

    // Randomise the shelves depending on which types are selected
    for (auto& shelf : m_imguiShelves)
    {
        shelf.second->Jitter(m_stateIt->second.flags, Cuda::kJitterRandomise);
    }
}

bool BakePermutor::Advance()
{
    // Can we advance? 
    if (m_isIdle ||
        m_stateIt == m_stateMap.GetStateData().end() ||
        m_sampleCountIt == m_sampleCountSet.cend() ||
        m_iterationIdx >= m_numIterations ||
        !m_lightProbeCameraShelf || !m_wavefrontTracerShelf)
    {
        if (m_disableLiveView && m_perspectiveCameraShelf)
        {
            m_perspectiveCameraShelf->GetParamsObject().camera.isActive = true;
        }
        return false;
    }

    // Increment to the next permutation
    if (m_permutationIdx >= 0)
    {
        m_completedSamples += *m_sampleCountIt;
        if (++m_sampleCountIt == m_sampleCountSet.cend())
        {
            m_sampleCountIt = m_sampleCountSet.cbegin();
            if (++m_iterationIdx == m_numIterations)
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
    ++m_permutationIdx;

    // Set the sample count on the light probe camera
    auto& probeParams = m_lightProbeCameraShelf->GetParamsObject();
    probeParams.camera.maxSamples = *m_sampleCountIt;
    probeParams.camera.isActive = true;

    // Always randomise the probe shelf to generate a new seed
    m_lightProbeCameraShelf->Randomise();

    // Set the shading mode to full illumination
    m_wavefrontTracerShelf->GetParamsObject().shadingMode = Cuda::kShadeFull;

    // Get some parameters that we'll need to generate the file paths
    m_bakeLightingMode == probeParams.lightingMode;
    m_bakeSeed = probeParams.camera.seed;

    // Override some parameters on the other shelves
    if (m_perspectiveCameraShelf)
    {
        // Deactivate the perspective camera
        if (m_disableLiveView) { m_perspectiveCameraShelf->GetParamsObject().camera.isActive = false; }
    }

    Log::Write("Starting bake permutation %i of %i...\n", m_permutationIdx, m_numPermutations);
    return true;
}

float BakePermutor::GetProgress() const
{
    return m_isIdle ? 0.0f : (float(min(m_permutationIdx, m_numPermutations)) / float(max(1, m_numPermutations)));
}

std::vector<std::string> BakePermutor::GenerateExportPaths() const
{
    // Naming convention:
    // SceneNameXXX.random10digits.6digitsSampleCount.usd

    Assert(m_permutationIdx < m_numPermutations);

    std::vector<std::string> exportPaths(2);
    for (int pathIdx = 0; pathIdx < 2; pathIdx++)
    {
        auto& path = exportPaths[pathIdx];
        for (const auto& token : m_templateTokens)
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
                path += tfm::format("%10.i", rng(mt) % 10000000000);
            }
            else if (token == "{$ITERATION}") { path += tfm::format("%i", m_iterationIdx); }
            else if (token == "{$SAMPLE_COUNT}") { path += tfm::format("%i", *m_sampleCountIt); }
            else if (token == "{$PERMUTATION}") { path += tfm::format("%i", m_permutationIdx); }
            else if (token == "{$SEED}") { path += tfm::format("%.6i", m_bakeSeed % 1000000); }
            else if (token == "{$STATE}") { path += m_stateIt->first; }
            else { path += token; }
        }
    }

    return exportPaths;
}

float BakePermutor::GetElapsedTime() const
{
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_startTime).count();
}

float BakePermutor::EstimateRemainingTime(const float bakeProgress) const
{
    const float sampleBatchProgress = (m_completedSamples + *m_sampleCountIt * bakeProgress) / float(m_totalSamples);
    const float stateProgress = (m_iterationIdx + sampleBatchProgress) / float(m_numIterations);
    const float totalProgress = (m_stateIdx + stateProgress) / float(m_numStates);

    return GetElapsedTime() * (1.0f - totalProgress) / max(1e-6f, totalProgress);
}