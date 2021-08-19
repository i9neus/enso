#include "IMGUIStateManager.h"
#include "generic/FilesystemUtils.h"

RenderObjectStateManager::RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves) : m_imguiShelves(imguiShelves)
{
    m_stateListCurrentIdx = -1;
    m_stateIDData.resize(2048);
    std::memset(m_stateIDData.data(), '\0', sizeof(char) * m_stateIDData.size());

   
}

void RenderObjectStateManager::Initialise(const Json::Node& node)
{
    m_stateJsonPath = node.GetRootDocument().GetOriginFilePath();
    std::string jsonStem = GetFileStem(m_stateJsonPath);
    ReplaceFilename(m_stateJsonPath, tfm::format("%s.states.json", jsonStem));

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

void RenderObjectStateManager::Insert(const std::string& id, bool overwriteIfExists)
{
    if (id.empty()) { Log::Error("Error: state ID must not be blank.\n");  return; }

    auto it = m_stateMap.find(id);
    std::shared_ptr<Json::Document> jsonPtr;
    if (it != m_stateMap.end())
    {
        if (!overwriteIfExists) { Log::Error("Error: state with ID '%s' already exists.\n", id); return; }

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
        if (shelf.second->IsJitterable())
        {
            Json::Node shelfNode = jsonPtr->AddChildObject(shelf.first);
            shelf.second->ToJson(shelfNode);
        }
    }

    WriteJson();
}

void RenderObjectStateManager::Erase(const std::string& id)
{
    auto it = m_stateMap.find(id);
    if (it == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return; }

    m_stateMap.erase(it);
    WriteJson();

    Log::Debug("Removed state '%s' from library.\n", id);
}

void RenderObjectStateManager::Restore(const std::string& id)
{
    auto stateIt = m_stateMap.find(id);
    if (stateIt == m_stateMap.end()) { Log::Error("Error: state with ID '%s' does not exist.\n", id); return; }

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

        shelfIt->second->FromJson(childNode, Json::kSilent);
    }

    Log::Debug("Restored state '%s' from library.\n", id);
}

void RenderObjectStateManager::ConstructIMGUI()
{
    if (ImGui::BeginListBox("States"))
    {
        StateMap::const_iterator it = m_stateMap.begin();
        for (int n = 0; n < m_stateMap.size(); n++, ++it)
        {
            const bool isSelected = (m_stateListCurrentIdx == n);
            if (ImGui::Selectable(it->first.c_str(), isSelected))
            {
                m_stateListCurrentId = it->first;
                m_stateListCurrentIdx = n;
            }
            if (isSelected) { ImGui::SetItemDefaultFocus(); }
        }
        ImGui::EndListBox();
    }
    ImGui::InputText("State ID", m_stateIDData.data(), m_stateIDData.size());

    // Save the current state to the container
    if (ImGui::Button("New"))
    {
        Insert(std::string(m_stateIDData.data()), false);
        std::memset(m_stateIDData.data(), '\0', sizeof(char) * m_stateIDData.size());
    }
    SL;
    // Overwrite the currently selected state
    if (ImGui::Button("Overwrite"))
    {
        if (m_stateListCurrentIdx < 0) { Log::Warning("Select a state from the list to overwrite it.\n"); }
        else
        {
            Insert(m_stateListCurrentId, true);
        }
    }
    SL;
    // Load a saved state to the UI
    if (ImGui::Button("Load") && m_stateListCurrentIdx >= 0 && !m_stateMap.empty())
    {
        Restore(m_stateListCurrentId);
    }
    SL;
    // Erase a saved state from the container
    if (ImGui::Button("Erase") && m_stateListCurrentIdx >= 0 && !m_stateMap.empty())
    {
        Erase(m_stateListCurrentId);
    }
}