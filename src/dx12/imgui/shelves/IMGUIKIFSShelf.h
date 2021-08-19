#pragma once

#include "IMGUIAbstractShelf.h"

#include "kernels/tracables/CudaKIFS.cuh"

// KIFS tracable
class KIFSShelf : public IMGUIShelf<Cuda::Host::KIFS, Cuda::KIFSParams>
{
public:
    KIFSShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~KIFSShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new KIFSShelf(json)); }
    virtual void Construct() override final;
    virtual void Reset() override final;

private:

};