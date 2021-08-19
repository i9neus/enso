#pragma once

#include "IMGUIAbstractShelf.h"

#include "kernels/tracables/CudaKIFS.cuh"

class KIFSStateContainer
{
public:
    using StateMap = std::map<const std::string, std::shared_ptr<Json::Document>>;

    KIFSStateContainer();

    void SetJsonPath(const std::string& filePath);
    void ReadJson();
    void WriteJson();

    void Insert(const std::string& id, const Cuda::KIFSParams& kifsParams, const bool overwriteIfExists);
    void Erase(const std::string& id);
    void Restore(const std::string& id, Cuda::KIFSParams& kifsParams);

    void ToJson(Json::Document& document);  

    const StateMap& GetStateMap() const { return m_stateMap; }

private:
    StateMap    m_stateMap;

    std::string             m_jsonPath;
};

// KIFS tracable
class KIFSShelf : public IMGUIShelf<Cuda::Host::KIFS, Cuda::KIFSParams>
{
public:
    KIFSShelf(const Json::Node& json);
    virtual ~KIFSShelf();

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new KIFSShelf(json)); }
    virtual void            Construct() override final;
    virtual void            Reset() override final;

private:
    void                    JitterKIFSParameters();

    int                     m_stateListCurrentIdx;
    std::string             m_stateListCurrentId;
    std::vector<char>       m_stateIDData;
    KIFSStateContainer      m_stateContainer;
    std::string             m_stateJsonPath;
};