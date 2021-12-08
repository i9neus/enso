#pragma once

#include "IMGUIElement.h"

enum StateObjectFlags : uint
{
    kStateEnabled               = 1 << 0,
    kStatePermuteLights         = 1 << 1,
    kStatePermuteGeometry       = 1 << 2,
    kStatePermuteMaterials      = 1 << 3,
    kStatePermuteTransforms     = 1 << 4,
    kStatePermuteColours        = 1 << 5,
    kStatePermuteIntensities    = 1 << 6,
    kStatePermuteFractals       = 1 << 7,
    kStatePermuteObjectFlags    = 1 << 8,
    kStatePermuteAll = 0xffffffff
};

enum class IMGUIDirtiness : int
{
    kClean,
    kSceneReload
};

class IMGUIAbstractShelf : public IMGUIElement
{
public:
    enum RandomiseFlags : int { kReset = 1 };

    IMGUIAbstractShelf() :  m_isJitterable(false) {}
    
    virtual void Construct() = 0;
    virtual void FromJson(const Json::Node& json, const int flags, bool dirtySceneGraph) = 0;
    
    virtual void ToJson(Json::Node& json) = 0;
    virtual bool IsDirty() const = 0;
    virtual void MakeClean() = 0;
    virtual void MakeDirty() = 0;
    virtual Cuda::AssetType GetRenderObjectAssetType() const = 0;
    virtual void OnUpdateRenderObjectStatistics(const Json::Node& shelfNode, const Json::Node& rootNode) {}

    virtual void Jitter(const uint flags, const uint operation) = 0;
    virtual void Update() {}

    const std::string& GetDAGPath() const { return m_dagPath; }
    const std::string& GetID() const { return m_id; }
    bool IsJitterable() const { return m_isJitterable; }

    void SetRenderObjectAttributes(const std::string& id, const std::string& dagPath)
    {
        m_id = id;
        m_dagPath = dagPath;
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
    IMGUIShelf() : m_p(m_paramsBuffer[0]), m_isDirty(false) {}

    IMGUIShelf(const Json::Node& json) : IMGUIShelf()
    {
        FromJson(json, Json::kRequiredWarn, false);
    }

    virtual ~IMGUIShelf() = default;

    virtual void FromJson(const Json::Node& json, const int flags, bool dirtySceneGraph) override final
    {
        m_paramsBuffer[0].FromJson(json, flags);
        if (!dirtySceneGraph)
        {
            m_paramsBuffer[1] = m_paramsBuffer[0];
            m_isDirty = true;
        }
    }

    virtual void ToJson(Json::Node& json) override final
    {
        m_paramsBuffer[0].ToJson(json);
    }

    virtual bool IsDirty() const override final
    {
        if (m_isDirty) { return true; }
        //static_assert(std::is_standard_layout<ParamsType>::value, "ParamsType must be standard layout.");

        for (int i = 0; i < sizeof(ParamsType); i++)
        {
            if (reinterpret_cast<const unsigned char*>(&m_paramsBuffer[0])[i] != reinterpret_cast<const unsigned char*>(&m_paramsBuffer[1])[i]) { return true; }
        }
        return false;
    }

    virtual void MakeDirty() override final
    {
        m_isDirty = true;
    }

    virtual void MakeClean() override final
    {
        Reset();
        m_paramsBuffer[1] = m_paramsBuffer[0];
        m_isDirty = false;
    }

    virtual void Reset() { Update(); }

    virtual Cuda::AssetType GetRenderObjectAssetType() const override { return ObjectType::GetAssetStaticType(); }

    ParamsType& GetParamsObject() { return m_paramsBuffer[0]; }

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
    std::array<ParamsType, 2>   m_paramsBuffer;
    ParamsType&                 m_p;
    bool                        m_isDirty;
};