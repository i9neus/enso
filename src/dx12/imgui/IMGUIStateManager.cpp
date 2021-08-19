#include "IMGUIStateManager.h"

RenderObjectStateManager::RenderObjectStateManager()
{

}

void RenderObjectStateManager::SetJsonPath(const std::string& filePath)
{
    m_jsonPath = filePath;
}

void RenderObjectStateManager::ReadJson()
{
    Log::Debug("Trying to restore KIFS state library...\n");

    Json::Document rootDocument;
    try
    {
        rootDocument.Load(m_jsonPath);
    }
    catch (const std::runtime_error& err)
    {
        Log::Debug("Failed: %s.\n", err.what());
    }

    for (Json::Node::Iterator it = rootDocument.begin(); it != rootDocument.end(); ++it)
    {
        std::string newId = it.Name();
        Json::Node childNode = *it;

        //Cuda::KIFSParams kifsParams(childNode, Json::kSilent);
        //Insert(newId, kifsParams, false);
    }
}

void RenderObjectStateManager::WriteJson()
{
    Json::Document rootDocument;
    for (auto& state : m_stateMap)
    {
        //Json::Node childNode = rootDocument.AddChildObject(state.first);
        //childNode.DeepCopy(*state.second);
    }

    rootDocument.WriteFile(m_jsonPath);
}

void RenderObjectStateManager::Insert(const std::string& id, const IMGUIAbstractShelfMap& shelves, bool overwriteIfExists)
{
   
}

void RenderObjectStateManager::Erase(const std::string& id)
{
   
}

void RenderObjectStateManager::Restore(const std::string& id, IMGUIAbstractShelfMap& shelves)
{
    
}

void RenderObjectStateManager::ToJson(Json::Document& document)
{
}