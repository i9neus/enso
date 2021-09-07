#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"

#include "kernels/lightprobes/CudaLightProbeKernelFilter.cuh"
#include "kernels/lightprobes/CudaLightProbeRegressionFilter.cuh"

#include "IMGUIAbstractShelf.h"

// Light probe kernel filter
class LightProbeKernelFilterShelf : public IMGUIShelf<Cuda::Host::LightProbeKernelFilter, Cuda::LightProbeKernelFilterParams>
{
public:
    LightProbeKernelFilterShelf(const Json::Node& json);
    virtual ~LightProbeKernelFilterShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new LightProbeKernelFilterShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final {}
    virtual void Reset() override final;

private:
    bool m_linkAlphaK;
};

// Light probe regression filter
class LightProbeRegressionFilterShelf : public IMGUIShelf<Cuda::Host::LightProbeRegressionFilter, Cuda::LightProbeRegressionFilterParams>
{
public:
    LightProbeRegressionFilterShelf(const Json::Node& json);
    virtual ~LightProbeRegressionFilterShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new LightProbeRegressionFilterShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final {}
    virtual void Reset() override final;

private:
    bool m_linkAlphaK;
};