#include "IMGUIStateManager.h"
#include "generic/FilesystemUtils.h"
#include "manager/RenderManager.h"
#include "shelves/IMGUIShelves.h"

BakePermutor::BakePermutor(IMGUIAbstractShelfMap& imguiShelves, RenderObjectStateMap& stateMap) : 
    m_imguiShelves(imguiShelves),
    m_stateMap(stateMap)
{    
    Clear();
}

void BakePermutor::Clear()
{
    m_sampleCountSet.clear();
    m_sampleCountIt = m_sampleCountSet.end();
    m_stateIt = m_stateMap.GetStateData().end();
    m_numIterations = 0;
    m_iterationIdx = -1;
    m_numPermutations = 0;
    m_permutationIdx = 0;
    m_templateTokens.clear();
    m_isIdle = true;
    m_disableLiveView = true;
}

void BakePermutor::Prepare(const int numIterations, const std::string& templatePath, const bool disableLiveView)
{
    m_sampleCountIt = m_sampleCountSet.begin();
    m_numIterations = numIterations;
    m_iterationIdx = -1;
    m_numPermutations = m_sampleCountSet.size() * m_numIterations * m_stateMap.GetNumPermutableStates();
    m_permutationIdx = -1;
    m_templatePath = templatePath;
    m_stateIt = m_stateMap.GetFirstPermutableState();
    m_isIdle = false;
    m_disableLiveView = disableLiveView;

    const auto r = *(m_stateIt);

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
    }

    if (!m_lightProbeCameraShelf) { Log::Debug("Error: bake permutor was unable to find an instance of LightProbeCameraShelf.\n"); }
}

bool BakePermutor::Advance()
{
    if (m_stateIt == m_stateMap.GetStateData().end() || 
        m_sampleCountIt == m_sampleCountSet.end() ||
        m_iterationIdx >= m_numIterations ||
        !m_lightProbeCameraShelf) 
    { 
        if (m_disableLiveView && m_perspectiveCameraShelf)
        { 
            m_perspectiveCameraShelf->GetParamsObject().camera.isActive = true; 
        }
        return false; 
    }
    
    // Increment to the next permutation
    ++m_iterationIdx;
    if (m_iterationIdx == m_numIterations)
    {
        m_iterationIdx = 0;
        if (++m_sampleCountIt == m_sampleCountSet.end())
        {
            m_sampleCountIt = m_sampleCountSet.begin();
            do
            {
                if (++m_stateIt == m_stateMap.GetStateData().end()) 
                { 
                    m_isIdle = true;
                    return false; 
                }
            } 
            while (!m_stateIt->second.isPermutable);
        }
    }
    ++m_permutationIdx;

    // Restore the state pointed
    m_stateMap.Restore(*m_stateIt);

    if (m_iterationIdx > 0)
    {
        // Randomise all the shelves
        for (auto& shelf : m_imguiShelves)
        {
            shelf.second->Randomise(Cuda::vec2(0.0f, 1.0f));
        }
    }

    // Set the sample count on the light probe camera
    m_lightProbeCameraShelf->GetParamsObject().camera.maxSamples = *m_sampleCountIt;

    // Deactivate the perspective camera
    if (m_disableLiveView && m_perspectiveCameraShelf)
    { 
        m_perspectiveCameraShelf->GetParamsObject().camera.isActive = false; 
    }

    Log::Write("Starting bake permutation %i of %i...\n", m_permutationIdx, m_numPermutations);

    return true;
}

float BakePermutor::GetProgress() const
{
    return m_isIdle ? 0.0f : (float(min(m_permutationIdx, m_numPermutations)) / float(max(1, m_numPermutations)));
}

std::string BakePermutor::GenerateExportPath() const
{
    Assert(m_permutationIdx < m_numPermutations);
    
    std::string exportPath;
    for (const auto& token : m_templateTokens)
    {
        Assert(!token.empty());
        if (token[0] != '{') 
        {
            exportPath += token;
            continue;
        }

        if (token == "{$ITERATION}")            { exportPath += tfm::format("%i", m_iterationIdx); }
        else if (token == "{$SAMPLE_COUNT}")    { exportPath += tfm::format("%i", *m_sampleCountIt); }
        else if (token == "{$PERMUTATION}")     { exportPath += tfm::format("%i", m_permutationIdx); }
        else if (token == "{$STATE}")           { exportPath += m_stateIt->first; }
        else                                    { exportPath += token; }
    }

    return exportPath;
}

void RenderObjectStateMap::FromJson(const Json::Node& node, const int flags)
{
    for (Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
    {
        std::string newId = it.Name();
        Json::Node versionNode = *it;

        auto& newState = m_stateMap[newId];
        newState.json.reset(new Json::Document());

        Json::Node patchNode = versionNode.GetChildObject("patch", flags);
        if (patchNode)
        {
            newState.json->DeepCopy(patchNode);
        }

        versionNode.GetValue("isPermutable", newState.isPermutable, flags);
    }
}

void RenderObjectStateMap::ToJson(Json::Node& node) const
{
    for (auto& state : m_stateMap)
    {
        Json::Node versionNode = node.AddChildObject(state.first);

        versionNode.AddValue("isPermutable", state.second.isPermutable);

        Json::Node patchNode = versionNode.AddChildObject("patch");
        patchNode.DeepCopy(*state.second.json);
    }
}

bool RenderObjectStateMap::Insert(const std::string& id, const bool isPermutable, bool overwriteIfExists)
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

    stateObjectPtr->isPermutable = isPermutable;

    // Dump the data from each shelf into the new JSON node
    for (const auto& shelf : m_imguiShelves)
    {
        //if (shelf.second->IsJitterable())
        {
            Json::Node shelfNode = stateObjectPtr->json->AddChildObject(shelf.first);
            shelf.second->ToJson(shelfNode);
        }
    }

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

}

bool RenderObjectStateMap::Restore(const std::string& id)
{
    auto stateIt = m_stateMap.find(id);
    if (stateIt == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return false; }

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
        if (state.second.isPermutable) { numPermutableStates++; }
    }
    return numPermutableStates;
}

RenderObjectStateMap::StateMap::const_iterator RenderObjectStateMap::GetFirstPermutableState() const
{
    RenderObjectStateMap::StateMap::const_iterator it = m_stateMap.cbegin();
    while (!it->second.isPermutable)
    {
        Assert(++it != m_stateMap.cend());
    } 
    return it;
}

RenderObjectStateManager::RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager) :
    m_imguiShelves(imguiShelves),
    m_renderManager(renderManager),
    m_stateListUI("Parameter states", "Add state", "Overwrite state", "Delete state"),
    m_sampleCountListUI("Sample counts", "+", "", "-"),
    m_numBakePermutations(1),
    m_isBaking(false),
    m_permutor(imguiShelves, m_stateMap),
    m_isPermutableUI(true),
    m_stateMap(imguiShelves),
    m_exportToUSD(false),
    m_disableLiveView(true)
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

void RenderObjectStateManager::Initialise(const Json::Node& node)
{
    m_stateJsonPath = node.GetRootDocument().GetOriginFilePath();
    std::string jsonStem = GetFileStem(m_stateJsonPath);
    ReplaceFilename(m_stateJsonPath, tfm::format("%s.states.json", jsonStem));

    std::function<bool(const std::string&)> onAddState = [this](const std::string& id) -> bool 
    { 
        if (m_stateMap.Insert(id, m_isPermutableUI, false))
        {
            SerialiseJson();
            return true;
        }
        return false;
    };
    m_stateListUI.SetOnAdd(onAddState);

    std::function<bool(const std::string&)> onOverwriteState = [this](const std::string& id) -> bool 
    { 
        if (m_stateMap.Insert(id, m_isPermutableUI, true))
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

    std::function<void(const std::string&)> onSelectItemState = [this](const std::string& id) -> void  
    { 
        auto it = m_stateMap.GetStateData().find(id);
        if (it != m_stateMap.GetStateData().end()) { m_isPermutableUI = it->second.isPermutable; }
    };
    m_stateListUI.SetOnSelect(onSelectItemState);

    DeserialiseJson();
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
        m_stateMap.Restore(m_stateListUI.GetCurrentlySelectedText());
    }

    ImGui::Checkbox("Permutable", &m_isPermutableUI);

    // Jitter the current state to generate a new scene
    if (ImGui::Button("Randomise"))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Randomise(Cuda::vec2(0.0f, 1.0f)); }
    }
    SL;
    // Reset all the jittered values to their midpoints
    if (ImGui::Button("Reset jitter"))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Randomise(Cuda::vec2(0.5f)); }
    }
}

void RenderObjectStateManager::ConstructBatchProcessorUI()
{
    UIStyle style(1);
    
    if (!ImGui::CollapsingHeader("Batch Processor", ImGuiTreeNodeFlags_DefaultOpen)) { return; }
    
    m_sampleCountListUI.Construct();
    
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
    ImGui::Checkbox("Disable live view", &m_disableLiveView);

    ImGui::ProgressBar(m_renderManager.GetBakeProgress(), ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Permutation %");
    ImGui::ProgressBar(m_permutor.GetProgress(), ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Bake %");
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
    m_permutor.Prepare(m_numBakePermutations, std::string(m_usdPathUIData.data()), m_disableLiveView);

    m_isBaking = true;
}

void RenderObjectStateManager::HandleBakeIteration()
{
    // Start a new iteration only if a bake has been requested and the renderer is ready
    if (!m_isBaking || m_renderManager.GetBakeStatus() != BakeStatus::kReady) { return; }
    
    // Advance to the next permutation. If this fails, we're done. 
    if (m_permutor.Advance())
    {       
        m_renderManager.StartBake(m_permutor.GenerateExportPath(), m_exportToUSD);
    }
    else
    {
        m_isBaking = false;
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