#include "IMGUIStateManager.h"
#include "generic/FilesystemUtils.h"
#include "generic/GlobalStateAuthority.h"
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
#include <filesystem>

void CopyStringToVector(const std::string& str, std::vector<char>& vec)
{
    vec.resize(math::max(2048ull, str.length()));
    std::memset(vec.data(), '\0', sizeof(char) * vec.size());
    std::memcpy(vec.data(), str.data(), sizeof(char) * str.length());
}

RenderObjectStateManager::RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager, const Json::Document& renderStateJson, Json::Document& commandQueue) :
    m_imguiShelves(imguiShelves),
    m_renderManager(renderManager),
    m_stateListUI("Parameter states", "Add", "Overwrite", "Delete", ""),
    m_noisySampleRange(64, 2048),
    m_referenceSamples(100000),
    m_thumbnailSamples(128),
    m_numBakeIterations(1),
    m_isBaking(false),
    m_isBatchRunning(false),
    m_permutor(imguiShelves, m_stateMap, commandQueue),
    m_stateMap(imguiShelves),
    m_exportToUSD(false),
    m_disableLiveView(false),
    m_startWithThisView(false),
    m_shutdownOnComplete(false),
    m_jitterFlags(kStatePermuteAll),
    m_gridValidityRange(0.7f, 0.95f),
    m_kifsIterationRange(7, 9),
    m_numStrata(20),
    m_dirtiness(IMGUIDirtiness::kClean),
    m_gridFitness(0.0f),
    m_renderStateJson(renderStateJson),
    m_commandQueue(commandQueue)
{
    m_usdPathTemplate = "probeVolume.{$SAMPLE_COUNT}.{$ITERATION}.usd";
    m_pngPathTemplate = "probeVolume.{$ITERATION}.png";

    CopyStringToVector(m_usdPathTemplate, m_usdPathUIData);
    CopyStringToVector(m_pngPathTemplate, m_pngPathUIData);   
}

RenderObjectStateManager::~RenderObjectStateManager()
{
    SerialiseJson();
}

void RenderObjectStateManager::Rebuild(const Json::Node& node)
{
    m_stateJsonPath = node.GetRootDocument().GetOriginFilePath();
    std::string jsonStem = GetFileStem(m_stateJsonPath);
    ReplaceFilename(m_stateJsonPath, tfm::format("%s.states.json", jsonStem));
    
    // Set up some functors for the state list UI
    std::function<bool(const std::string&)> onAddState = [this](const std::string& id) -> bool
    {
        if (m_stateMap.Insert(id, m_jitterFlags, false))
        {
            SerialiseJson();
            return true;
        }
        return false;
    };
    m_stateListUI.SetOnAdd(onAddState);

    std::function<bool(const std::string&)> onOverwriteState = [this](const std::string& id) -> bool
    {
        if (m_stateMap.Insert(id, m_jitterFlags, true))
        {
            SerialiseJson();
            return true;
        }
        return false;
    };
    m_stateListUI.SetOnOverwrite(onOverwriteState);

    std::function<bool(const std::string&)> onDeleteState = [this](const std::string& id) -> bool
    {
        if (m_stateMap.Erase(id))
        {
            SerialiseJson();
            return true;
        }
        return false;
    };
    m_stateListUI.SetOnDelete(onDeleteState);

    std::function<bool()> onDeleteAllState = [this]() -> bool { return false;  };
    m_stateListUI.SetOnDeleteAll(onDeleteAllState);

    /*std::function<void(const std::string&)> onSelectItemState = [this](const std::string& id) -> void
    {
        auto it = m_stateMap.GetStateData().find(id);
        if (it != m_stateMap.GetStateData().end()) { m_jitterFlags = it->second.flags; }
    };
    m_stateListUI.SetOnSelect(onSelectItemState);*/

    // Load the JSON state dictionary
    DeserialiseJson();

    // Look for scene files populate the scene list
    ScanForSceneFiles();

    // Look for the light probe shelf so we can find the DAG path for the camera stats
    for (auto& shelf : m_imguiShelves)
    {
        std::shared_ptr<LightProbeCameraShelf> lightProbeCameraShelf = std::dynamic_pointer_cast<LightProbeCameraShelf>(shelf.second);
        if(lightProbeCameraShelf)
        {
            m_lightProbeCameraDAG = lightProbeCameraShelf->GetDAGPath();
            break;
        }
    }
}

void RenderObjectStateManager::Initialise(HWND hWnd)
{   
    m_hWnd = hWnd;
}

void RenderObjectStateManager::ScanForSceneFiles()
{
    const std::string& sceneDirectory = GSA().GetDefaultSceneDirectory();

    m_sceneFilePathList.clear();
    m_sceneFileNameList.clear();
    m_sceneListIdx = -1;

    namespace fs = std::filesystem;
    for (auto const& entry : fs::directory_iterator(sceneDirectory))
    {
        if (!entry.is_regular_file()) { continue; }

        // Only consider JSON files
        auto fullPath = entry.path();
        if(fullPath.extension().string() != ".json") { continue; }

        // Ignore state files with formats similar to 'xxxx.states.json'
        auto stemPath = fullPath.stem();
        if (!stemPath.extension().empty()) { continue; }

        // Store the scene path
        m_sceneFilePathList.push_back(fullPath.string());
        m_sceneFileNameList.push_back(fullPath.filename().string());
    }    
}

void RenderObjectStateManager::DeserialiseJson()
{
    Log::Debug("Trying to restore KIFS state library...\n");

    Json::Document rootDocument;
    try
    {
        rootDocument.Deserialise(m_stateJsonPath);
    }
    catch (const std::runtime_error& err)
    {
        Log::Debug("Failed: %s.\n", err.what());
        return;
    }

    const int jsonWarningLevel = Json::kRequiredWarn;

    // Rebuild the state map from the JSON dictionary
    Json::Node stateNode = rootDocument.GetChildObject("states", jsonWarningLevel);
    if (stateNode)
    {
        m_stateMap.FromJson(stateNode, jsonWarningLevel);

        m_stateListUI.Clear();
        for (auto element : m_stateMap.GetStateData())
        {
            m_stateListUI.Insert(element.first);
        }
    }

    // Rebuild the permutation settings
    Json::Node permNode = rootDocument.GetChildObject("permutations", jsonWarningLevel);
    if (permNode)
    {
        permNode.GetVector("noisySampleRange", m_noisySampleRange, jsonWarningLevel);
        permNode.GetValue("referenceSamples", m_referenceSamples, jsonWarningLevel);
        permNode.GetValue("thumbnailSamples", m_thumbnailSamples, jsonWarningLevel);
        permNode.GetValue("strata", m_numStrata, jsonWarningLevel);
        permNode.GetValue("iterations", m_numBakeIterations, jsonWarningLevel);
        permNode.GetVector("gridValidityRange", m_gridValidityRange, jsonWarningLevel);
        permNode.GetVector("kifsIterationRange", m_kifsIterationRange, jsonWarningLevel);
        if (permNode.GetValue("usdPathTemplate", m_usdPathTemplate, jsonWarningLevel))
        {
            CopyStringToVector(m_usdPathTemplate, m_usdPathUIData);
        }
        if (permNode.GetValue("pngPathTemplate", m_pngPathTemplate, jsonWarningLevel))
        {
            CopyStringToVector(m_pngPathTemplate, m_pngPathUIData);
        }
    }
}

void RenderObjectStateManager::SerialiseJson() const
{
    if (m_stateJsonPath.empty()) { return; }
    
    Json::Document rootDocument;
    
    Json::Node stateNode = rootDocument.AddChildObject("states");
    m_stateMap.ToJson(stateNode);    

    Json::Node permJson = rootDocument.AddChildObject("permutations");
    {        
        permJson.AddVector("noisySampleRange", m_noisySampleRange);
        permJson.AddValue("referenceSamples", m_referenceSamples);
        permJson.AddValue("thumbnailSamples", m_thumbnailSamples);
        permJson.AddValue("strata", m_numStrata);
        permJson.AddVector("gridValidityRange", m_gridValidityRange);
        permJson.AddVector("kifsIterationRange", m_kifsIterationRange);
        permJson.AddValue("iterations", m_numBakeIterations);
        permJson.AddValue("usdPathTemplate", std::string(m_usdPathUIData.data()));
        permJson.AddValue("pngPathTemplate", std::string(m_pngPathUIData.data()));
    }

    rootDocument.Serialise(m_stateJsonPath);
}

void RenderObjectStateManager::ConstructSceneManagerUI()
{
    UIStyle style(0);

    if (!ImGui::CollapsingHeader("Scene Manager", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::PushID("SceneManager");

    ConstructListBox("Local scenes", m_sceneFileNameList, m_sceneListIdx);

    if (ImGui::Button("Load"))
    {
        if(m_sceneListIdx >= 0 && m_sceneListIdx < m_sceneFilePathList.size())
        {
            m_renderManager.LoadScene(m_sceneFilePathList[m_sceneListIdx]);
            m_dirtiness = IMGUIDirtiness::kSceneReload;
        }
    }
    SL;
    if (ImGui::Button("Rescan"))
    {
        ScanForSceneFiles();
    }

    ImGui::PopID();
}

void RenderObjectStateManager::ConstructStateManagerUI()
{
    UIStyle style(0);
    
    if (!ImGui::CollapsingHeader("State Manager", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::PushID("StateManager");

    m_stateListUI.Construct();
    // Load a saved state to the UI
    SL;
    if (ImGui::Button("Load") && m_stateListUI.IsSelected())
    {        
        const std::string id = m_stateListUI.GetCurrentlySelectedText();
        
        auto it = m_stateMap.GetStateData().find(id);
        if (it != m_stateMap.GetStateData().end()) { m_jitterFlags = it->second.flags; }

        m_stateMap.Restore(id);
    }
    SL;
    if (ImGui::Button("Clone") && m_stateListUI.IsSelected())
    {
        //m_stateMap.Insert(tfm::format("%sm_stateMap.GetCurrentStateID()
    }

    auto FlaggedCheckbox = [this](const std::string& id, const uint flag)
    {
        bool checked = m_jitterFlags & flag;
        ImGui::Checkbox(id.c_str(), &checked);
        m_jitterFlags = (m_jitterFlags & ~flag) | (checked ? flag : 0);
    };

    FlaggedCheckbox("Enabled", kStateEnabled);

    FlaggedCheckbox("Lights", kStatePermuteLights); SL;
    FlaggedCheckbox("Geometry", kStatePermuteGeometry); SL;
    FlaggedCheckbox("Materials", kStatePermuteMaterials);
    
    FlaggedCheckbox("Transforms", kStatePermuteTransforms); SL;
    FlaggedCheckbox("Colours", kStatePermuteColours); SL;
    FlaggedCheckbox("Fractals", kStatePermuteFractals); SL;
    FlaggedCheckbox("Object flags", kStatePermuteObjectFlags); 

    // Jitter the current state to generate a new scene
    ImVec2 buttonSize = ImGui::GetItemRectSize();
    buttonSize.y *= 2;
    if (ImGui::Button("Shuffle", buttonSize))
    {
        for (auto& shelf : m_imguiShelves)
        {
            shelf.second->Jitter(m_jitterFlags, Cuda::kJitterRandomise);
        }
    } 
    SL;    
    // Reset all the jittered values to their midpoints
    if (ImGui::Button("Reset", buttonSize))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Jitter(m_jitterFlags, Cuda::kJitterReset); }
    }
    SL;
    // Bake the evaluated jittered values as the base parameters
    if (ImGui::Button("Flatten", buttonSize))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Jitter(m_jitterFlags, Cuda::kJitterFlatten); }
    }

    ImGui::PopID();
}

void RenderObjectStateManager::ParseRenderStateJson()
{    
    // Get the running state of the renderer
    m_renderStateJson.GetValue("renderManager/rendererStatus", m_renderState, Json::kRequiredAssert);
    
    // Bake job stats...
    const Json::Node& bakeJson = m_renderStateJson.GetChildObject("jobs/bake", Json::kSilent);    
    m_isBaking = bool(bakeJson);
    if(m_isBaking)
    {
        /*int state;
        bakeJson.GetValue("state", state, Json::kRequiredAssert | Json::kLiteralID);
        Log::Debug("Bake: %i", state);*/

        m_bakeProgress = 0.0f;
        m_lastBakeSucceeded = true;
        bakeJson.GetValue("progress", m_bakeProgress, Json::kRequiredAssert);
        bakeJson.GetValue("succeeded", m_lastBakeSucceeded, Json::kSilent);
    }   

    // Render object stats...
    const Json::Node& statsJson = m_renderStateJson.GetChildObject("jobs/getStats/renderObjects", Json::kSilent);
    if (statsJson)
    {
        if (!m_lightProbeCameraDAG.empty())
        {
            const Json::Node& cameraJson = statsJson.GetChildObject(m_lightProbeCameraDAG, Json::kSilent | Json::kLiteralID);
            if (cameraJson)
            {
                const Json::Node& gridListJson = cameraJson.GetChildObject("grids", Json::kRequiredAssert);                
                
                m_bakeGridValidity = -1.0f;
                for (const auto& gridJson : gridListJson)
                {
                    if (!gridJson.IsObject()) { continue; }
                    
                    float meanValidity, minSamples;
                    gridJson.GetValue("meanProbeValidity", meanValidity, Json::kRequiredAssert);
                    gridJson.GetValue("minSamples", minSamples, Json::kRequiredAssert);
                    
                    if (minSamples > 0.0f)
                    {
                        m_bakeGridSamples = minSamples;
                        if (meanValidity >= 0.0f)
                        {
                            m_bakeGridValidity = (m_bakeGridValidity < 0.0f) ? meanValidity : math::min(meanValidity, m_bakeGridValidity);
                        }
                    }
                }
            }
        }
    }
}

void RenderObjectStateManager::ConstructBatchProcessorUI()
{
    UIStyle style(1);
    
    if (!ImGui::CollapsingHeader("Batch Processor", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ParseRenderStateJson(); // Get some render state data so we know what's going on

    ImGui::PushID("BatchProcessor");

    if (ImGui::TreeNodeEx("Samples", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragInt2("Noisy sample range", &m_noisySampleRange[0], 1.0f, 1, 1e7);
        m_noisySampleRange.y = max(m_noisySampleRange.x + 1, m_noisySampleRange.y);

        ImGui::SliderInt("Strata", &m_numStrata, 1, 50);
        ImGui::DragInt("Reference samples", &m_referenceSamples, 1.0f, 1, 1e7);
        ImGui::DragInt("Thumbnail samples", &m_thumbnailSamples, 1.0f, 1, 1024);

        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Constraints", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragFloat2("Grid validity range", &m_gridValidityRange[0], 0.001f, 0.0f, 1.0f);

        ImGui::SliderInt2("KIFS interation range", &m_kifsIterationRange[0], 1, 10);
        m_kifsIterationRange[0] = math::min(m_kifsIterationRange[0], m_kifsIterationRange[1]);
        m_kifsIterationRange[1] = math::max(m_kifsIterationRange[0], m_kifsIterationRange[1]);

        ImGui::TreePop();
    }    
    
    if (ImGui::Button("Reset Defaults"))
    {
        m_noisySampleRange = Cuda::ivec2(64, 2048);
        m_referenceSamples = 100000;
        m_numStrata = 20;
        m_gridValidityRange = Cuda::vec2(0.7f, 0.95f);
        m_kifsIterationRange = Cuda::ivec2(7, 9);
    }
    
    // The number of iterations per permutation
    ImGui::DragInt("Iterations", &m_numBakeIterations, 1, 1, 100000);

    // The base paths that all export paths will be derived from
    ImGui::InputText("USD export path", m_usdPathUIData.data(), m_usdPathUIData.size());    
    ImGui::InputText("PNG export path", m_pngPathUIData.data(), m_pngPathUIData.size());

    ImVec2 size = ImGui::GetItemRectSize();
    size.y *= 2;
    const std::string actionText = m_isBatchRunning ? "Abort" : "Bake";
    if (ImGui::Button(actionText.c_str(), size))
    {
        // Start or abort the batch bake
        EnqueueBatch();
    }

    ImGui::Checkbox("Export to USD", &m_exportToUSD); SL;
    ImGui::Checkbox("Disable live view", &m_disableLiveView); SL;
    ImGui::Checkbox("Shutdown on complete", &m_shutdownOnComplete);

    if (ImGui::Button("Save PNG"))
    {        
        EnqueueExportViewport();
    }

    ImGui::ProgressBar(m_bakeProgress, ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Permutation %");
    ImGui::ProgressBar(m_permutor.GetProgress(), ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Bake %");
    ImGui::Text("%s elapsed", m_isBatchRunning ? FormatElapsedTime(m_permutor.GetElapsedTime()).c_str() : "00:00");
    ImGui::Text("%s remaining", m_isBatchRunning ? FormatElapsedTime(m_permutor.EstimateRemainingTime(m_bakeProgress)).c_str() : "00:00");

    bool gridOk = m_bakeGridValidity >= m_gridValidityRange[0] && m_bakeGridValidity <= m_gridValidityRange[1];
    ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(gridOk ? 0.33f : 0.0f, 0.6f, 0.6f));
    ImGui::Text("Bake grid validity: %.2f", m_bakeGridValidity);
    ImGui::PopStyleColor(1);

    ImGui::Text("Bake grid samples: %.2f", m_bakeGridSamples);

    ImGui::PopID();
}

void RenderObjectStateManager::EnqueueExportViewport()
{
    Json::Node commandNode = m_commandQueue.AddChildObject("exportViewport");
    Log::Error(std::string(m_pngPathUIData.data()));
    commandNode.AddValue("path", m_permutor.GeneratePNGExportPath(std::string(m_pngPathUIData.data()), m_stateJsonPath));
}

void RenderObjectStateManager::EnqueueBatch()
{
    // If a batch bake is already in progress, dispatch a command to stop it
    if (m_isBaking || m_isBatchRunning)
    {
        Json::Node commandJson = m_commandQueue.AddChildObject("bake");
        commandJson.AddValue("action", "abort");
        m_isBatchRunning = false;
        return;
    }

    // Initialise the permutor parameter structure
    BakePermutor::Params params;
    params.probeGridTemplatePath = std::string(m_usdPathUIData.data());
    params.renderTemplatePath = std::string(m_pngPathUIData.data());
    params.jsonRootPath = m_stateJsonPath;

    params.disableLiveView = m_disableLiveView;
    params.startWithThisView = true;
    params.exportToUsd = m_exportToUSD;
    params.exportToPng = true;

    params.gridValidityRange = m_gridValidityRange;
    params.kifsIterationRange = m_kifsIterationRange;
    params.jitterFlags = m_jitterFlags;
    params.numIterations = m_numBakeIterations;
    params.noisyRange = m_noisySampleRange;
    params.referenceCount = m_referenceSamples;
    params.thumbnailCount = m_thumbnailSamples;
    params.numStrata = m_numStrata;
    params.kifsIterationRange = m_kifsIterationRange;
    
    if(m_permutor.Prepare(params))
    {
        m_isBatchRunning = true;
    }
}

void RenderObjectStateManager::HandleBakeIteration()
{
    constexpr int kMinSamplesForEstimate = 8;
    
    if (!m_isBatchRunning || m_isBaking) { return; }  // Batch isn't running or a bake is in progress, so nothing to do

    // Advance to the next permutation. If this fails, we're done. 
    if (m_permutor.Advance(m_lastBakeSucceeded))
    {
        // Reset the statistics
        m_bakeGridValidity = -1.0f;
        m_bakeGridSamples = 0.0f;
    }
    else
    {
        m_isBatchRunning = false;
        if (m_shutdownOnComplete)
        {
            SendMessage(m_hWnd, WM_CLOSE, 0, 0);
        }
    }
}

void RenderObjectStateManager::ConstructUI()
{
    ImGui::Begin("Combinatorics");
    
    ConstructSceneManagerUI();    
    ConstructStateManagerUI();
    ConstructBatchProcessorUI();

    ImGui::End();
}