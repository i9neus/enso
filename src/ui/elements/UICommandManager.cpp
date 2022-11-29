#include "UICommandManager.h"
#include "io/json/JsonUtils.h"
#include "io/CommandQueue.h"

namespace Enso
{
    UICommandManager::UICommandManager(UIObjectContainer& container) :
        m_objectContainer(container)
    {
        RegisterEventHandler("OnCreateObject", this, &UICommandManager::OnCreateObject);
        RegisterEventHandler("OnUpdateObject", this, &UICommandManager::OnUpdateObject);
        RegisterEventHandler("OnDeleteObject", this, &UICommandManager::OnDeleteObject);
    }

    // Flushes the command queue and processes the commands contained within it
    void UICommandManager::Flush(CommandQueue& cmdQueue)
    {
        // Copy the data from the command queue and purge it
        Json::Document rootNode;
        if (!cmdQueue.Flush(rootNode)) { return; }

        Log::Debug(rootNode.Stringify(true));

        try
        {
            // Process each command in turn
            for (Json::Document::Iterator nodeIt = rootNode.begin(); nodeIt != rootNode.end(); ++nodeIt)
            {
                Json::Node childNode = *nodeIt;
                if (!childNode.IsObject()) { continue; }

                const auto functorIt = m_eventMap.find(nodeIt.Name());
                if (functorIt == m_eventMap.end())
                {
                    Log::Debug("Flush: command '%s' was not registered", nodeIt.Name());
                    continue;
                }

                // Call the event functor
                (functorIt->second)(childNode);
            }
        }
        catch (std::runtime_error& err)
        {
            Log::Error("Command error: %s", err.what());
        }
    }

    // Create a UI object that conforms to the specified schema
    void UICommandManager::OnCreateObject(const Json::Node& node)
    {
        AssertMsg(node.IsObject() && node.NumMembers() == 1, "OnCreateObject: Expecting an object with 1 member.");
        auto nodeIt = node.begin();
        
        AssertMsgFmt(!ObjectExists(nodeIt.Name()), "OnCreateObject: UI object with name '%s' already exists.", nodeIt.Name());

        // Get the class of the object
        const Json::Node objectJson = *nodeIt;
        std::string objectClass;       
        objectJson.GetValue("class", objectClass, Json::kRequiredAssert | Json::kNotBlank);

        // Get a handle to the schema
        auto schema = SerialisableObjectSchemaContainer::FindSchema(objectClass);
        AssertMsgFmt(schema, "OnCreateObject: schema for class '%s' has not been registered.", objectClass.c_str());

        // Create and emplace the new object
        m_objectContainer.emplace(nodeIt.Name(), std::make_shared<UIGenericObject>(nodeIt.Name(), *schema, objectJson));
        Log::Debug("OnCreateObject: Created new UI objects '%s'.", nodeIt.Name());
    }

    // Updates the properties of one or more objects
    void UICommandManager::OnUpdateObject(const Json::Node& node)
    {
        AssertMsg(node.IsObject(), "OnUpdateObject: Expected an object.");
        for (auto nodeIt = node.begin(); nodeIt != node.end(); ++nodeIt)
        {
            if (!ObjectExists(nodeIt.Name()))
            {
                Log::Warning("OnUpdateObject: UI object with name '%s' was not found.", nodeIt.Name());
                continue;
            }

            m_objectContainer[nodeIt.Name()]->Deserialise(*nodeIt);
        }
    }

    // Deletes one or more object
    void UICommandManager::OnDeleteObject(const Json::Node& node)
    {
        AssertMsg(node.IsObject(), "OnUpdateObject: Expected an object.");
        int numDeleted = 0;
        for (auto nodeIt = node.begin(); nodeIt != node.end(); ++nodeIt)
        {
            auto objectIt = m_objectContainer.find(nodeIt.Name());
            if(objectIt == m_objectContainer.end())
            {
                Log::Warning("OnUpdateObject: UI object with name '%s' was not found.", nodeIt.Name());
                continue;
            }
            else
            {
                m_objectContainer.erase(objectIt);
                numDeleted++;
            }
        }

        Log::Debug("OnDeleteObject: Deleted %i UI objects.", numDeleted);
    }

    bool UICommandManager::ObjectExists(const std::string& objectId) const
    {
        return m_objectContainer.find(objectId) != m_objectContainer.end();
    }
}