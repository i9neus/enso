#include "UICommandManager.h"
#include "io/json/JsonUtils.h"
#include "io/CommandQueue.h"

namespace Enso
{
    UICommandManager::UICommandManager(UIObjectContainer& container) :
        CommandManager(),
        m_objectContainer(container)
    {
        RegisterEventHandler("OnCreateObject", this, &UICommandManager::OnCreateObject);
        RegisterEventHandler("OnUpdateObject", this, &UICommandManager::OnUpdateObject);
        RegisterEventHandler("OnDeleteObject", this, &UICommandManager::OnDeleteObject);
    }

    // Create a UI object that conforms to the specified schema
    void UICommandManager::OnCreateObject(const Json::Node& node)
    {
        AssertMsg(node.IsObject(), "OnUpdateObject: Expected an object.");
        for (auto nodeIt = node.begin(); nodeIt != node.end(); ++nodeIt)
        {
            AssertMsgFmt(!ObjectExists(nodeIt.Name()), "OnCreateObject: UI object with name '%s' already exists.", nodeIt.Name().c_str());

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