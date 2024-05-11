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

    template<typename Lambda> void UICommandManager::IterateArrayObjects(const Json::Node& node, Lambda lambda)
    {
        AssertMsg(node.IsArray(), "Expected an array.");

        for (int idx = 0; idx < node.Size(); ++idx)
        {
            const Json::Node objectJson = node[idx];
            std::string id;
            objectJson.GetValue("id", id, Json::kRequiredAssert | Json::kNotBlank);

            lambda(objectJson, id);
        }
    }

    // Create a UI object that conforms to the specified schema
    void UICommandManager::OnCreateObject(const Json::Node& node)
    {       
        auto lambda = [&, this](const Json::Node& itemNode, const std::string& id) -> void
        {
            AssertMsgFmt(!ObjectExists(id), "OnCreateObject: UI object with name '%s' already exists.", id.c_str());

            // Get the class of the object
            std::string objectClass;
            itemNode.GetValue("class", objectClass, Json::kRequiredAssert | Json::kNotBlank);

            // Get a handle to the schema
            auto schema = SerialisableObjectSchemaContainer::FindSchema(objectClass);
            AssertMsgFmt(schema, "OnCreateObject: schema for class '%s' has not been registered.", objectClass.c_str());

            // Create and emplace the new object
            m_objectContainer.emplace(id, std::make_shared<UIGenericObject>(id, *schema, itemNode));
            Log::Debug("OnCreateObject: Created new UI objects '%s'.", id);
        };

        IterateArrayObjects(node, lambda);
    }

    // Updates the properties of one or more objects
    void UICommandManager::OnUpdateObject(const Json::Node& node)
    {
        auto lambda = [&, this](const Json::Node& itemNode, const std::string& id) -> void
        {
            if (!ObjectExists(id))
            {
                Log::Warning("OnUpdateObject: UI object with name '%s' was not found.", id);
                return;
            }

            m_objectContainer[id]->Deserialise(itemNode);
        };

        IterateArrayObjects(node, lambda);
    }

    // Deletes one or more object
    void UICommandManager::OnDeleteObject(const Json::Node& node)
    {
        int numDeleted = 0;
        auto lambda = [&, this](const Json::Node& itemNode, const std::string& id) -> void
        {
            auto objectIt = m_objectContainer.find(id);
            if (objectIt == m_objectContainer.end())
            {
                Log::Warning("OnUpdateObject: UI object with name '%s' was not found.", id);
                return;
            }
            else
            {
                m_objectContainer.erase(objectIt);
                numDeleted++;
            }
        };
       
        IterateArrayObjects(node, lambda);
        Log::Debug("OnDeleteObject: Deleted %i UI objects.", numDeleted);
    }

    bool UICommandManager::ObjectExists(const std::string& objectId) const
    {
        return m_objectContainer.find(objectId) != m_objectContainer.end();
    }
}