#pragma once

#include "thirdparty/imgui/imgui.h"

#include "kernels/materials/CudaSimpleMaterial.cuh"
#include "kernels/materials/CudaCornellMaterial.cuh"
#include "kernels/materials/CudaKIFSMaterial.cuh"

#include "IMGUIAbstractShelf.h"

// Simple material
class SimpleMaterialShelf : public IMGUIShelf<Cuda::Host::SimpleMaterial, Cuda::SimpleMaterialParams>
{
public:
    SimpleMaterialShelf(const Json::Node& json);
    virtual ~SimpleMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SimpleMaterialShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredColourPicker                   m_albedoPicker;
    IMGUIJitteredColourPicker                   m_incandPicker;
};

// KIFS code material
class KIFSMaterialShelf : public IMGUIShelf<Cuda::Host::KIFSMaterial, Cuda::KIFSMaterialParams>
{
public:
    KIFSMaterialShelf(const Json::Node& json);
    virtual ~KIFSMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new KIFSMaterialShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredColourPicker                   m_albedoPicker;
    IMGUIJitteredColourPicker                   m_incandPicker;
};

// Cornell material
class CornellMaterialShelf : public IMGUIShelf<Cuda::Host::CornellMaterial, Cuda::CornellMaterialParams>
{
public:
    CornellMaterialShelf(const Json::Node& json);
    virtual ~CornellMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new CornellMaterialShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;
    virtual void Update() override final;

private:
    std::array<IMGUIJitteredColourPicker, 6>    m_pickers;
};