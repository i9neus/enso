#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"
#include "generic/StdIncludes.h"
#include "generic/JsonUtils.h"

#include "kernels/math/CudaMath.cuh"

#define SL ImGui::SameLine()

namespace Json { class Document; class Node; }

class IMGUIAbstractShelf
{
public:
    IMGUIAbstractShelf() = default;
    
    virtual void Construct() = 0;
    virtual bool ToJson(std::string& newJson) = 0;

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
    void ConstructTransform(Cuda::BidirectionalTransform& transform, const bool isDifferentiable);
    void ConstructComboBox(const std::string& name, const std::vector<std::string>& elements, int& selected);

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
        FromJson(json, Json::kRequiredWarn);
    }

    virtual ~IMGUIShelf() = default;

    void FromJson(const Json::Node& json, const int flags)
    {
        m_params[0].FromJson(json, flags);
        m_params[1] = m_params[0];
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