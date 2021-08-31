#pragma once

#include "../shelves/IMGUIAbstractShelf.h"
#include <set>

class RenderManager;
class LightProbeCameraShelf;
class PerspectiveCameraShelf;
class WavefrontTracerShelf;

class RenderObjectStateMap
{
public:
    struct StateObject
    {
        StateObject() : flags(kStateEnabled | kStatePermuteLights | kStatePermuteGeometry) {}
        StateObject(std::shared_ptr<Json::Document> json_, bool flags_) : json(json_), flags(flags_) {}

        std::shared_ptr<Json::Document>     json;
        uint                                flags;
    };

    using StateMap = std::map<const std::string, StateObject>;

    RenderObjectStateMap(IMGUIAbstractShelfMap& imguiShelves) : m_imguiShelves(imguiShelves) {}
    ~RenderObjectStateMap() = default;

    void FromJson(const Json::Node& node, const int flags);
    void ToJson(Json::Node& node) const;

    bool Insert(const std::string& id, const int flags, const bool overwriteIfExists);
    bool Erase(const std::string& id);
    bool Restore(const std::string& id);
    bool Restore(const std::pair<std::string, StateObject>& it);
    void Clear();
    int GetNumPermutableStates() const;
    const std::string& GetCurrentStateID() const { return m_currentStateID; }
    RenderObjectStateMap::StateMap::const_iterator GetFirstPermutableState() const;

    inline const StateMap& GetStateData() const { return m_stateMap; }

private:
    StateMap                m_stateMap;
    IMGUIAbstractShelfMap& m_imguiShelves;
    std::string             m_currentStateID;
};