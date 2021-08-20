#pragma once

#include "IMGUIElement.h"

class IMGUIAbstractShelf : public IMGUIElement
{
public:
    enum RandomiseFlags : int { kReset = 1 };

    IMGUIAbstractShelf() = default;
    
    virtual void Construct() = 0;
    virtual void FromJson(const Json::Node& json, const int flags, bool dirtySceneGraph) = 0;
    virtual bool ToJson(std::string& newJson) = 0;
    virtual void ToJson(Json::Node& json) = 0;

    virtual void Randomise(int flags) {}

    const std::string& GetDAGPath() const { return m_dagPath; }
    const std::string& GetID() const { return m_id; }
    bool IsJitterable() const { return m_isJitterable; }

    void SetRenderObjectAttributes(const std::string& id, const std::string& dagPath, const bool isJitterable)
    {
        m_id = id;
        m_dagPath = dagPath;
        m_isJitterable = isJitterable;
    }

protected:
    std::string m_dagPath;
    std::string m_id;
    bool        m_isJitterable;
};

using IMGUIAbstractShelfMap = std::map<std::string, std::shared_ptr<IMGUIAbstractShelf>>;

template<typename ObjectType, typename ParamsType>
class IMGUIShelf : public IMGUIAbstractShelf
{
public:
    IMGUIShelf(const Json::Node& json)
    {
        FromJson(json, Json::kRequiredWarn, false);
    }

    virtual ~IMGUIShelf() = default;

    virtual void FromJson(const Json::Node& json, const int flags, bool dirtySceneGraph) override final
    {
        m_params[0].FromJson(json, flags);
        if (!dirtySceneGraph)
        {
            m_params[1] = m_params[0];
        }
    }

    virtual bool ToJson(std::string& newJson) override final
    {
        //if (m_params[0] == m_params[1]) { return false; }
        if (!IsDirty()) { return false; }
        m_params[1] = m_params[0];

        Json::Document newNode;
        m_params[0].ToJson(newNode);
        newJson = newNode.Stringify();

        Reset();

        return true;
    }

    virtual void ToJson(Json::Node& node) override final
    {
        m_params[0].ToJson(node);
    }

    virtual void Reset() {}

    bool IsDirty() const
    {
        //static_assert(std::is_standard_layout<ParamsType>::value, "ParamsType must be standard layout.");

        for (int i = 0; i < sizeof(ParamsType); i++)
        {
            if (reinterpret_cast<const unsigned char*>(&m_params[0])[i] != reinterpret_cast<const unsigned char*>(&m_params[1])[i]) { return true; }
        }
        return false;
    }

    std::string GetShelfTitle()
    {
        const auto& assetDescription = ObjectType::GetAssetDescriptionString();
        if (assetDescription.empty())
        {
            return tfm::format("%s", m_id);
        }

        return tfm::format("%s: %s", ObjectType::GetAssetDescriptionString(), m_id);
    }

protected:
    std::array<ParamsType, 2> m_params;
};