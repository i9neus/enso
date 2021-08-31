#pragma once

#include "IMGUIAbstractShelf.h"

#include "kernels/tracables/CudaKIFS.cuh"

// KIFS tracable
class KIFSShelf : public IMGUIShelf<Cuda::Host::KIFS, Cuda::KIFSParams>
{
public:
    KIFSShelf(const Json::Node& json);
    virtual ~KIFSShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new KIFSShelf(json)); }
    virtual void            Construct() override final;
    virtual void            Update() override final;

    virtual void            Jitter(const uint flags, const uint operation) override final;

private:
    void                    JitterKIFSParameters();

    int                     m_stateListCurrentIdx;
    std::string             m_stateListCurrentId;
    std::vector<char>       m_stateIDData;
    std::string             m_stateJsonPath;

    IMGUIJitteredFlagArray      m_faceFlags;
    IMGUIJitteredParameterTable m_jitteredParamTable;
};