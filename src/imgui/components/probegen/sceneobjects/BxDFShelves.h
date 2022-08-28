#pragma once

#include "thirdparty/imgui/imgui.h"

#include "kernels/bxdfs/CudaLambert.cuh"

#include "IMGUIAbstractShelf.h"

// Lambertian BRDF
class LambertBRDFShelf : public IMGUIShelf<Cuda::Host::LambertBRDF, Cuda::LambertBRDFParams>
{
public:
    LambertBRDFShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~LambertBRDFShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new LambertBRDFShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final {}
};