#pragma once

#include "shelves/IMGUIAbstractShelf.h"

class RenderObjectStateManager
{
public:
    using StateMap = std::map<const std::string, std::shared_ptr<Json::Document>>;

    RenderObjectStateManager();

    void SetJsonPath(const std::string& filePath);
    void ReadJson();
    void WriteJson();

    void Insert(const std::string& id, const IMGUIAbstractShelfMap& shelves, const bool overwriteIfExists);
    void Erase(const std::string& id);
    void Restore(const std::string& id, IMGUIAbstractShelfMap& shelves);

    void ToJson(Json::Document& document);

    const StateMap& GetStateMap() const { return m_stateMap; }

private:
    StateMap    m_stateMap;

    std::string             m_jsonPath;
};