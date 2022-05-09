#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"

#include "kernels/ml/CudaFCNNProbeDenoiser.cuh"

#include "IMGUIAbstractShelf.h"

// FCNN Probe Denoiser
class FCNNProbeDenoiserShelf : public IMGUIShelf<Cuda::Host::FCNNProbeDenoiser, Cuda::FCNNProbeDenoiserParams>
{
public:
    FCNNProbeDenoiserShelf(const Json::Node& json);
    virtual ~FCNNProbeDenoiserShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node & json) { return std::make_shared<FCNNProbeDenoiserShelf>(json); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final {}
    virtual void Reset() override final;

private:
    IMGUIInputText      m_modelRootPath;
    IMGUIInputText      m_modelPreprocessPath;
    IMGUIInputText      m_modelPostprocessPath;
    IMGUIInputText      m_modelDenoiserPath;
};

