#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"

#include "kernels/ml/CudaFCNNProbeDenoiser.cuh"

#include "IMGUIAbstractShelf.h"

// Light probe kernel filter
class FCNNProbeDenoiserShelf : public IMGUIShelf<Cuda::Host::FCNNProbeDenoiser, Cuda::LightProbeGridParams>
{
public:
    FCNNProbeDenoiserShelf(const Json::Node& json);
    virtual ~FCNNProbeDenoiserShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node & json) { return std::make_shared<FCNNProbeDenoiserShelf>(json); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final {}
    virtual void Reset() override final;

};

