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

RenderObjectStateManager::RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager, const Json::Document& renderStateJson, Json::Document& commandJson) :
    m_imguiShelves(imguiShelves),
    m_renderManager(renderManager),
    m_stateListUI("Parameter states", "Add", "Overwrite", "Delete", ""),
    m_noisySampleRange(64, 2048),
    m_referenceSamples(100000),
    m_numBakeIterations(1),
    m_isBaking(false),
    m_isBatchRunning(false),
    m_permutor(imguiShelves, m_stateMap),
    m_stateMap(imguiShelves),
    m_exportToUSD(false),
    m_disableLiveView(true),
    m_startWithThisView(false),
    m_shutdownOnComplete(false),
    m_stateFlags(kStatePermuteAll),
    m_gridValidityRange(0.7f, 0.95f),
    m_kifsIterationRange(7, 9),
    m_numStrata(20),
    m_dirtiness(IMGUIDirtiness::kClean),
    m_gridFitness(0.0f),
    m_renderStateJson(renderStateJson),
    m_commandQueue(commandJson)
{
    m_usdPathTemplate = "probeVolume.{$SAMPLE_COUNT}.{$ITERATION}.usd";
    
    m_usdPathUIData.resize(2048);
    std::memset(m_usdPathUIData.data(), '\0', sizeof(char) * m_usdPathUIData.size());
    std::memcpy(m_usdPathUIData.data(), m_usdPathTemplate.data(), sizeof(char) * m_usdPathTemplate.length());
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

    std::function<bool(const std::string&)> onAddState = [this](const std::string& id) -> bool
    {
        if (m_stateMap.Insert(id, m_stateFlags, false))
        {
            SerialiseJson();
            return true;
        }
        return false;
    };
    m_stateListUI.SetOnAdd(onAddState);

    std::function<bool(const std::string&)> onOverwriteState = [this](const std::string& id) -> bool
    {
        if (m_stateMap.Insert(id, m_stateFlags, true))
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
        if (it != m_stateMap.GetStateData().end()) { m_stateFlags = it->second.flags; }
    };
    m_stateListUI.SetOnSelect(onSelectItemState);*/

    DeserialiseJson();

    ScanForSceneFiles();
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
        permNode.GetValue("strata", m_numStrata, jsonWarningLevel);
        permNode.GetValue("iterations", m_numBakeIterations, jsonWarningLevel);
        permNode.GetVector("gridValidityRange", m_gridValidityRange, jsonWarningLevel);
        permNode.GetVector("kifsIterationRange", m_kifsIterationRange, jsonWarningLevel);
        if (permNode.GetValue("usdPathTemplate", m_usdPathTemplate, jsonWarningLevel))
        {
            std::memset(m_usdPathUIData.data(), '\0', sizeof(char) * m_usdPathUIData.size());
            std::memcpy(m_usdPathUIData.data(), m_usdPathTemplate.data(), sizeof(char) * m_usdPathTemplate.length());
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
        permJson.AddValue("strata", m_numStrata);
        permJson.AddVector("gridValidityRange", m_gridValidityRange);
        permJson.AddVector("kifsIterationRange", m_kifsIterationRange);
        permJson.AddValue("iterations", m_numBakeIterations);
        permJson.AddValue("usdPathTemplate", std::string(m_usdPathUIData.data()));
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
        if (it != m_stateMap.GetStateData().end()) { m_stateFlags = it->second.flags; }

        m_stateMap.Restore(id);
    }
    SL;
    if (ImGui::Button("Clone") && m_stateListUI.IsSelected())
    {
        //m_stateMap.Insert(tfm::format("%sm_stateMap.GetCurrentStateID()
    }

    auto FlaggedCheckbox = [this](const std::string& id, const uint flag)
    {
        bool checked = m_stateFlags & flag;
        ImGui::Checkbox(id.c_str(), &checked);
        m_stateFlags = (m_stateFlags & ~flag) | (checked ? flag : 0);
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
            shelf.second->Jitter(m_stateFlags, Cuda::kJitterRandomise);
        }
    } 
    SL;    
    // Reset all the jittered values to their midpoints
    if (ImGui::Button("Reset", buttonSize))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Jitter(m_stateFlags, Cuda::kJitterReset); }
    }
    SL;
    // Bake the evaluated jittered values as the base parameters
    if (ImGui::Button("Flatten", buttonSize))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Jitter(m_stateFlags, Cuda::kJitterFlatten); }
    }

    ImGui::PopID();
}

void RenderObjectStateManager::ParseRenderStateJson()
{
    Log::Debug(m_renderStateJson.Stringify(true));
    
    m_renderStateJson.GetValue("renderManager/rendererStatus", m_renderState, Json::kRequiredAssert);
    
    const Json::Node& bakeJson = m_renderStateJson.GetChildObject("jobs/bake", Json::kSilent);
    m_isBaking = bool(bakeJson);
    m_bakeProgress = 0.0f;

    if(m_isBaking)
    {
        m_bakeProgress = 0.0f;
        bakeJson.GetValue("progress", m_bakeProgress, Json::kSilent);
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
    
    ImGui::DragInt("Iterations", &m_numBakeIterations, 1, 1, 100000);

    // New element input control
    ImGui::InputText("USD export path", m_usdPathUIData.data(), m_usdPathUIData.size());    

     // Reset all the jittered values to their midpoints
    ImVec2 size = ImGui::GetItemRectSize();
    size.y *= 2;
    const std::string actionText = m_isBatchRunning ? "Abort" : "Bake";
    if (ImGui::Button(actionText.c_str(), size))
    {
        EnqueueBake();
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

    ImGui::PopID();
}

void RenderObjectStateManager::EnqueueExportViewport()
{
    Json::Node commandNode = m_commandQueue.AddChildObject("exportViewport");
    std::string filePath(m_usdPathUIData.data());
    ReplaceFilename(filePath, "blah.png");
    commandNode.AddValue("path", filePath);
}

void RenderObjectStateManager::EnqueueBake()
{
    // If a batch bake is already in progress, dispatch a command to stop it
    if (m_isBaking || m_isBatchRunning)
    {
        Json::Node commandJson = m_commandQueue.AddChildObject("bake");
        commandJson.AddValue("action", "abort");
        m_isBatchRunning = false;
        return;
    }

    // Initialise the permutor and prepare it with the constraints
    if (m_permutor.Initialise(std::string(m_usdPathUIData.data()), m_stateJsonPath, m_disableLiveView, true))
    {
        if (m_permutor.Prepare(m_numBakeIterations, m_noisySampleRange, m_referenceSamples, m_numStrata, m_kifsIterationRange))
        {
            m_isBatchRunning = true;
        }
    }
}

void RenderObjectStateManager::HandleBakeIteration()
{
    // Start a new iteration only if a bake has been requested and the renderer is ready
    if (!m_isBatchRunning || m_isBaking) { return; }
    
    // Advance to the next permutation. If this fails, we're done. 
    if (m_permutor.Advance())
    {       
        Json::Node bakeJson = m_commandQueue.AddChildObject("bake");
        bakeJson.AddValue("action", "start");
        bakeJson.AddValue("exportToUSD", m_exportToUSD);
        bakeJson.AddValue("minGridValidity", m_gridValidityRange.x);
        bakeJson.AddValue("maxGridValidity", m_gridValidityRange.y);
        bakeJson.AddArray("usdExportPaths", m_permutor.GenerateExportPaths());
    }
    else
    {
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