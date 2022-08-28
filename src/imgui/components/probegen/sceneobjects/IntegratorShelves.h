#pragma once

#include "thirdparty/imgui/imgui.h"

#include "kernels/CudaWavefrontTracer.cuh"

#include "IMGUIAbstractShelf.h"

// Wavefront tracer
class WavefrontTracerShelf : public IMGUIShelf<Cuda::Host::WavefrontTracer, Cuda::WavefrontTracerParams>
{
public:
    WavefrontTracerShelf(const Json::Node& json);
    virtual ~WavefrontTracerShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new WavefrontTracerShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
};