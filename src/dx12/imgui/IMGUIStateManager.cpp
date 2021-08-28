#include "IMGUIStateManager.h"
#include "generic/FilesystemUtils.h"
#include "manager/RenderManager.h"
#include "shelves/IMGUIShelves.h"

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

void BakePermutor::Prepare(const int numIterations, const std::string& templatePath, const bool disableLiveView, const bool startWithThisView)
{
    m_totalSamples = m_completedSamples = 0;
    m_sampleCountIt = m_sampleCountSet.cbegin();
    m_totalSamples = 0;
    m_numIterations = numIterations;
    m_iterationIdx = 0;
    m_numStates = m_stateMap.GetNumPermutableStates();
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

    if (!m_lightProbeCameraShelf) { Log::Debug("Error: bake permutor was unable to find an instance of LightProbeCameraShelf.\n"); }
    if (!m_wavefrontTracerShelf) { Log::Debug("Error: bake permutor was unable to find an instance of WavefrontTracerShelf.\n"); }

    m_startTime = std::chrono::high_resolution_clock::now();

    // Restore the state pointed
    m_stateMap.Restore(*m_stateIt);
}

void BakePermutor::RandomiseScene()
{
    if (m_startWithThisView && m_iterationIdx == 0) { return; }
    
    Assert(m_stateIt != m_stateMap.GetStateData().end());
    
    // Randomise the shelves depending on which types are selected
    for (auto& shelf : m_imguiShelves)
    {
        shelf.second->Randomise(m_stateIt->second.flags, Cuda::vec2(0.0f, 1.0f));
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
    m_lightProbeCameraShelf->Randomise(m_stateIt->second.flags, Cuda::vec2(0.0f, 1.0f));

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
    for(int pathIdx = 0; pathIdx < 2; pathIdx++)
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
            else if (token == "{$ITERATION}")       { path += tfm::format("%i", m_iterationIdx); }
            else if (token == "{$SAMPLE_COUNT}")    { path += tfm::format("%i", *m_sampleCountIt); }
            else if (token == "{$PERMUTATION}")     { path += tfm::format("%i", m_permutationIdx); }
            else if (token == "{$SEED}")            { path += tfm::format("%.6i", m_bakeSeed % 1000000); }
            else if (token == "{$STATE}")           { path += m_stateIt->first; }            
            else                                    { path += token; }
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

void RenderObjectStateMap::FromJson(const Json::Node& node, const int jsonFlags)
{
    for (Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
    {
        std::string newId = it.Name();
        Json::Node versionNode = *it;

        auto& newState = m_stateMap[newId];
        newState.json.reset(new Json::Document());

        Json::Node patchNode = versionNode.GetChildObject("patch", jsonFlags);
        if (patchNode)
        {
            newState.json->DeepCopy(patchNode);
        }

        versionNode.GetValue("flags", newState.flags, jsonFlags);
    }
}

void RenderObjectStateMap::ToJson(Json::Node& node) const
{
    for (auto& state : m_stateMap)
    {
        Json::Node versionNode = node.AddChildObject(state.first);

        versionNode.AddValue("flags", state.second.flags);

        Json::Node patchNode = versionNode.AddChildObject("patch");
        patchNode.DeepCopy(*state.second.json);
    }
}

bool RenderObjectStateMap::Insert(const std::string& id, const int flags, bool overwriteIfExists)
{
    if (id.empty()) { Log::Error("Error: state ID must not be blank.\n");  return false; }

    auto it = m_stateMap.find(id);
    StateObject* stateObjectPtr = nullptr;
    if (it != m_stateMap.end())
    {
        if (!overwriteIfExists) { Log::Error("Error: state with ID '%s' already exists.\n", id); return false; }

        stateObjectPtr = &(it->second);
        Assert(stateObjectPtr);
        Assert(stateObjectPtr->json);
        stateObjectPtr->json->Clear();

        Log::Debug("Updated state '%s' in library.\n", id);
    }
    else
    {
        stateObjectPtr = &(m_stateMap[id]);
        stateObjectPtr->json.reset(new Json::Document());

        Log::Debug("Added KIFS state '%s' to library.\n", id);
    }

    stateObjectPtr->flags = flags;

    // Dump the data from each shelf into the new JSON node
    for (const auto& shelf : m_imguiShelves)
    {
        //if (shelf.second->IsJitterable())
        {
            Json::Node shelfNode = stateObjectPtr->json->AddChildObject(shelf.first);
            shelf.second->ToJson(shelfNode);
        }
    }

    m_currentStateID = id;
    return true;
}

bool RenderObjectStateMap::Erase(const std::string& id)
{
    auto it = m_stateMap.find(id);
    if (it == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return false; }

    m_stateMap.erase(it);

    Log::Debug("Removed state '%s' from library.\n", id);
    return true;
}

void RenderObjectStateMap::Clear()
{
    m_currentStateID = "";
}

bool RenderObjectStateMap::Restore(const std::string& id)
{
    auto stateIt = m_stateMap.find(id);
    if (stateIt == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return false; }

    m_currentStateID = stateIt->first;

    return Restore(*stateIt);
}

bool RenderObjectStateMap::Restore(const std::pair<std::string, StateObject>& element)
{    
    Assert(element.second.json);
    const auto& node = *(element.second.json);

    for (Json::Node::ConstIterator stateIt = node.begin(); stateIt != node.end(); ++stateIt)
    {
        const std::string dagPath = stateIt.Name();
        Json::Node childNode = *stateIt;

        auto shelfIt = m_imguiShelves.find(dagPath);

        if (shelfIt == m_imguiShelves.end())
        {
            Log::Debug("Error: trying to restore render object state '%s', but object does not exist in shelf map.\n", dagPath);
            continue;
        }

        shelfIt->second->FromJson(childNode, Json::kSilent, true);
        shelfIt->second->Update();        
    }

    Log::Debug("Restored state '%s' from library.\n", element.first);
    return true;
}

int RenderObjectStateMap::GetNumPermutableStates() const
{
    int numPermutableStates = 0;
    for (auto state : m_stateMap)
    {
        if (state.second.flags & kStateEnabled) { numPermutableStates++; }
    }
    return numPermutableStates;
}

RenderObjectStateMap::StateMap::const_iterator RenderObjectStateMap::GetFirstPermutableState() const
{
    RenderObjectStateMap::StateMap::const_iterator it = m_stateMap.cbegin();
    while (!(it->second.flags & kStateEnabled))
    {
        Assert(++it != m_stateMap.cend());
    } 
    return it;
}

RenderObjectStateManager::RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager) :
    m_imguiShelves(imguiShelves),
    m_renderManager(renderManager),
    m_stateListUI("Parameter states", "Add", "Overwrite", "Delete", ""),
    m_sampleCountListUI("Sample counts", " + ", "", " - ", "Clear"),
    m_numBakePermutations(1),
    m_isBaking(false),
    m_permutor(imguiShelves, m_stateMap),
    m_stateMap(imguiShelves),
    m_exportToUSD(false),
    m_disableLiveView(true),
    m_startWithThisView(false),
    m_shutdownOnComplete(false),
    m_stateFlags(kStatePermuteAll)
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

void RenderObjectStateManager::Initialise(const Json::Node& node, HWND hWnd)
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

    m_hWnd = hWnd;
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
        std::vector<int> sampleCounts;
        permNode.GetArrayValues("sampleCounts", sampleCounts, jsonWarningLevel);
        for (auto item : sampleCounts) { m_sampleCountListUI.Insert(tfm::format("%i", item)); }
        permNode.GetValue("iterations", m_numBakePermutations, jsonWarningLevel);
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
        std::vector<int> sampleCounts;
        for (const auto& item : m_sampleCountListUI.GetListItems()) { sampleCounts.push_back(std::atoi(item.c_str())); }
        permJson.AddArray("sampleCounts", sampleCounts);
        permJson.AddValue("iterations", m_numBakePermutations);
        permJson.AddValue("usdPathTemplate", std::string(m_usdPathUIData.data()));
    }

    rootDocument.Serialise(m_stateJsonPath);
}

void RenderObjectStateManager::ConstructStateManagerUI()
{
    UIStyle style(0);
    
    if (!ImGui::CollapsingHeader("State Manager", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

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
    if (ImGui::Button("Shuffle"))
    {
        for (auto& shelf : m_imguiShelves)
        {
            shelf.second->Randomise(m_stateFlags, Cuda::vec2(0.0f, 1.0f));
        }
    } 
    SL;    
    // Reset all the jittered values to their midpoints
    if (ImGui::Button("Reset"))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Randomise(kStatePermuteAll, Cuda::vec2(0.5f)); }
    }
}

void RenderObjectStateManager::ConstructBatchProcessorUI()
{
    UIStyle style(1);
    
    if (!ImGui::CollapsingHeader("Batch Processor", ImGuiTreeNodeFlags_DefaultOpen)) { return; }
    
    m_sampleCountListUI.Construct(); SL;
    
    if (ImGui::Button("Defaults"))
    {
        m_sampleCountListUI.Clear();
        for (int i = 32; i <= 65536; i <<= 1)
        {
            m_sampleCountListUI.Insert(tfm::format("%i", i));
        }
        m_sampleCountListUI.Insert("100000");
    }
    
    ImGui::DragInt("Permutations", &m_numBakePermutations, 1, 1, 100000);

    // New element input control
    ImGui::InputText("USD export path", m_usdPathUIData.data(), m_usdPathUIData.size());    

     // Reset all the jittered values to their midpoints
    const BakeStatus bakeStatus = m_renderManager.GetBakeStatus();
    ImVec2 size = ImGui::GetItemRectSize();
    size.y *= 2;
    const std::string actionText = (bakeStatus != BakeStatus::kReady) ? "Abort" : "Bake";
    if (ImGui::Button(actionText.c_str(), size))
    {
        ToggleBake();
    }

    ImGui::Checkbox("Export to USD", &m_exportToUSD); SL;
    ImGui::Checkbox("Disable live view", &m_disableLiveView); SL;
    ImGui::Checkbox("Shutdown on complete", &m_shutdownOnComplete);

    if (ImGui::Button("Save PNG"))
    {
        std::string filePath(m_usdPathUIData.data());
        ReplaceFilename(filePath, "blah.png");
        m_renderManager.ExportLiveViewport(filePath);
    }

    ImGui::ProgressBar(m_renderManager.GetBakeProgress(), ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Permutation %");
    ImGui::ProgressBar(m_permutor.GetProgress(), ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Bake %");
    ImGui::Text("%s elapsed", (bakeStatus != BakeStatus::kReady) ? FormatElapsedTime(m_permutor.GetElapsedTime()).c_str() : "00:00");
    ImGui::Text("%s remaining", (bakeStatus != BakeStatus::kReady) ? FormatElapsedTime(m_permutor.EstimateRemainingTime(m_renderManager.GetBakeProgress())).c_str() : "00:00");
}

void RenderObjectStateManager::ToggleBake()
{
    // If a bake is in progress, abort it
    if (m_isBaking)
    {
        m_renderManager.AbortBake();
        m_isBaking = false;
        return;
    }
    
    const auto& sampleList = m_sampleCountListUI.GetListItems();
    if (sampleList.empty())
    {
        Log::Error("ERROR: Can't start bake; no sample counts were specified.\n");
        return;
    }

    m_permutor.Clear();
    for (const auto& item : sampleList)
    {
        const int count = std::atoi(item.c_str());
        if (count >= 1)
        {
            m_permutor.GetSampleCountSet().insert(count);
        }
    }
    m_permutor.Prepare(m_numBakePermutations, std::string(m_usdPathUIData.data()), m_disableLiveView, true);

    m_isBaking = true;
}

void RenderObjectStateManager::HandleBakeIteration()
{
    // Start a new iteration only if a bake has been requested and the renderer is ready
    if (!m_isBaking || m_renderManager.GetBakeStatus() != BakeStatus::kReady) { return; }
    
    // Advance to the next permutation. If this fails, we're done. 
    if (m_permutor.Advance())
    {       
        m_renderManager.StartBake(m_permutor.GenerateExportPaths(), m_exportToUSD);
    }
    else
    {
        m_isBaking = false;
        if (m_shutdownOnComplete)
        {
            SendMessage(m_hWnd, WM_CLOSE, 0, 0);
        }
    }
}

void RenderObjectStateManager::ConstructUI()
{
    ImGui::Begin("Combinatorics");
    
    ConstructStateManagerUI();

    ConstructBatchProcessorUI();

    HandleBakeIteration();

    ImGui::End();
}