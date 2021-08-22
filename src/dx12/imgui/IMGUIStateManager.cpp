#include "IMGUIStateManager.h"
#include "generic/FilesystemUtils.h"
#include "manager/RenderManager.h"

RenderObjectStateManager::RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves, RenderManager& renderManager) :
    m_imguiShelves(imguiShelves),
    m_renderManager(renderManager),
    m_stateList("Parameter states", "Add state", "Overwrite state", "Delete state"),
    m_sampleCountList("Sample counts", "+", "", "-"),
    m_numPermutations(1),
    m_isBaking(false)
{
    m_usdPathData.resize(2048);
    std::memset(m_usdPathData.data(), '\0', sizeof(char) * m_usdPathData.size());
    
    std::string defaultUSDPath = "probeVolume.#####.###.usd";
    std::memcpy(m_usdPathData.data(), defaultUSDPath.data(), sizeof(char) * defaultUSDPath.length());
}

void RenderObjectStateManager::Initialise(const Json::Node& node)
{
    m_stateJsonPath = node.GetRootDocument().GetOriginFilePath();
    std::string jsonStem = GetFileStem(m_stateJsonPath);
    ReplaceFilename(m_stateJsonPath, tfm::format("%s.states.json", jsonStem));

    std::function<bool(const std::string&)> onAddState = [this](const std::string& id) -> bool { return Insert(id, false); };
    m_stateList.SetOnAdd(onAddState);

    std::function<bool(const std::string&)> onOverwriteState = [this](const std::string& id) -> bool { return Insert(id, true); };
    m_stateList.SetOnOverwrite(onOverwriteState);

    std::function<bool(const std::string&)> onDeleteState = [this](const std::string& id) -> bool { return Erase(id); };
    m_stateList.SetOnDelete(onOverwriteState);

    std::function<bool()> onDeleteAllState = [this]() -> bool { return false;  };
    m_stateList.SetOnDeleteAll(onDeleteAllState);

    ReadJson();
}

void RenderObjectStateManager::ReadJson()
{
    Log::Debug("Trying to restore KIFS state library...\n");

    Json::Document rootDocument;
    try
    {
        rootDocument.Load(m_stateJsonPath);
    }
    catch (const std::runtime_error& err)
    {
        Log::Debug("Failed: %s.\n", err.what());
    }

    m_stateMap.clear();

    // Rebuild the state map from the JSON dictionary
    for (Json::Node::Iterator it = rootDocument.begin(); it != rootDocument.end(); ++it)
    {
        std::string newId = it.Name();
        Json::Node childNode = *it;

        auto& statePtr = m_stateMap[newId];
        statePtr.reset(new Json::Document());
        statePtr->DeepCopy(childNode);
    }

    m_stateList.Clear();
    for (auto element : m_stateMap)
    {
        m_stateList.Insert(element.first);
    }
}

void RenderObjectStateManager::WriteJson()
{
    Json::Document rootDocument;
    for (auto& state : m_stateMap)
    {
        Json::Node childNode = rootDocument.AddChildObject(state.first);
        childNode.DeepCopy(*state.second);
    }

    rootDocument.WriteFile(m_stateJsonPath);
}

bool RenderObjectStateManager::Insert(const std::string& id, bool overwriteIfExists)
{
    if (id.empty()) { Log::Error("Error: state ID must not be blank.\n");  return false; }

    auto it = m_stateMap.find(id);
    std::shared_ptr<Json::Document> jsonPtr;
    if (it != m_stateMap.end())
    {
        if (!overwriteIfExists) { Log::Error("Error: state with ID '%s' already exists.\n", id); return false; }

        Assert(it->second);
        jsonPtr = it->second;
        jsonPtr->Clear();

        Log::Debug("Updated state '%s' in library.\n", id);
    }
    else
    {
        auto& statePtr = m_stateMap[id];
        statePtr.reset(new Json::Document());
        jsonPtr = statePtr;

        Log::Debug("Added KIFS state '%s' to library.\n", id);
    }

    // Dump the data from each shelf into the new JSON node
    for (const auto& shelf : m_imguiShelves)
    {
        //if (shelf.second->IsJitterable())
        {
            Json::Node shelfNode = jsonPtr->AddChildObject(shelf.first);
            shelf.second->ToJson(shelfNode);
        }
    }

    WriteJson();
    return true;
}

bool RenderObjectStateManager::Erase(const std::string& id)
{
    auto it = m_stateMap.find(id);
    if (it == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return false; }

    m_stateMap.erase(it);
    WriteJson();

    Log::Debug("Removed state '%s' from library.\n", id); 
    return true;
}

void RenderObjectStateManager::Clear()
{
    
}

bool RenderObjectStateManager::Restore(const std::string& id)
{
    auto stateIt = m_stateMap.find(id);
    if (stateIt == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return false; }

    Assert(stateIt->second);
    const auto& node = *(stateIt->second);

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
    }

    Log::Debug("Restored state '%s' from library.\n", id);
}

void RenderObjectStateManager::ConstructStateManagerUI()
{
    if (!ImGui::CollapsingHeader("State Manager", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_stateList.Construct();

    // Load a saved state to the UI
    SL;
    if (ImGui::Button("Load") && m_stateList.IsSelected())
    {
        Restore(m_stateList.GetCurrentlySelectedText());
    }

    // Jitter the current state to generate a new scene
    if (ImGui::Button("Randomise"))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Randomise(0); }
    }
    SL;
    // Reset all the jittered values to their midpoints
    if (ImGui::Button("Reset jitter"))
    {
        for (auto& shelf : m_imguiShelves) { shelf.second->Randomise(IMGUIAbstractShelf::kReset); }
    }
}

void RenderObjectStateManager::ConstructBatchProcessorUI()
{
    if (!ImGui::CollapsingHeader("Batch Processor", ImGuiTreeNodeFlags_DefaultOpen)) { return; }
    
    m_sampleCountList.Construct();

    ImGui::DragInt("Permutations", &m_numPermutations, 1, 1, 100000);

    // New element input control
    ImGui::InputText("USD export path", m_usdPathData.data(), m_usdPathData.size());

    bool isBakeRunning = false;
    //float bakeProgress;
    //m_renderManager.GetBakeStatus(isBakeRunning, bakeProgress);
    
    // Reset all the jittered values to their midpoints
    ImVec2 size = ImGui::GetItemRectSize();
    size.y *= 2;
    std::string actionText = isBakeRunning ? "Abort" : "Bake";
    if (ImGui::Button(actionText.c_str(), size))
    {
        /*if (isBakeRunning)
        {
            m_renderManager.AbortBake();
        }
        else
        {
            const auto& sampleList = m_sampleCountList.GetListItems();
            if (sampleList.empty())
            {
                Log::Error("ERROR: Can't start bake; no sample counts were specified.\n");
            }
            else
            {
                RenderManager::BakeParams bakeParams;
                for (const auto& item : sampleList)
                {
                    const int count = std::atoi(item.c_str());
                    constexpr int kMaxSampleCount = 100000000;
                    if (count >= 1 && count <= kMaxSampleCount)
                    {
                        bakeParams.sampleCountList.push_back(count);
                    }
                }

                bakeParams.numPermutations = m_numPermutations;
                bakeParams.usdPathTemplate = std::string(m_usdPathData.data());

                m_renderManager.StartBake(bakeParams);
            }
        }*/
    }

    //ImGui::ProgressBar(bakeProgress, ImVec2(0.0f, 0.0f));
   
}

void RenderObjectStateManager::ConstructUI()
{
    ImGui::Begin("Combinatorics");
    
    ConstructStateManagerUI();

    ConstructBatchProcessorUI();

    ImGui::End();
}