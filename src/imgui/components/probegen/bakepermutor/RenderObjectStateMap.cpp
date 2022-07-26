#include "RenderObjectStateMap.h"
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