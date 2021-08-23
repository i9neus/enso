#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"

#include "kernels/tracables/CudaSphere.cuh"
#include "kernels/tracables/CudaPlane.cuh"
#include "kernels/tracables/CudaCornellBox.cuh"

#include "kernels/lights/CudaQuadLight.cuh"
#include "kernels/lights/CudaSphereLight.cuh"
#include "kernels/lights/CudaEnvironmentLight.cuh"

#include "kernels/bxdfs/CudaLambert.cuh"

#include "kernels/materials/CudaSimpleMaterial.cuh"
#include "kernels/materials/CudaCornellMaterial.cuh"
#include "kernels/materials/CudaKIFSMaterial.cuh"

#include "kernels/cameras/CudaPerspectiveCamera.cuh"
#include "kernels/cameras/CudaLightProbeCamera.cuh"
#include "kernels/cameras/CudaFisheyeCamera.cuh"

#include "kernels/CudaWavefrontTracer.cuh"

#include "IMGUIAbstractShelf.h"

// Simple material
class SimpleMaterialShelf : public IMGUIShelf<Cuda::Host::SimpleMaterial, Cuda::SimpleMaterialParams>
{
public:
    SimpleMaterialShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~SimpleMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SimpleMaterialShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
};

// KIFS code material
class KIFSMaterialShelf : public IMGUIShelf<Cuda::Host::KIFSMaterial, Cuda::KIFSMaterialParams>
{
public:
    KIFSMaterialShelf(const Json::Node& json);
    virtual ~KIFSMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new KIFSMaterialShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
private:
    IMGUIColourPicker                   m_albedoPicker;
    IMGUIColourPicker                   m_incandPicker;
};

// Cornell material
class CornellMaterialShelf : public IMGUIShelf<Cuda::Host::CornellMaterial, Cuda::CornellMaterialParams>
{
public:
    CornellMaterialShelf(const Json::Node& json);
    virtual ~CornellMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new CornellMaterialShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
    virtual void Update() override final;

private:
    std::array<IMGUIColourPicker, 6>    m_pickers;
};

// Plane tracable
class PlaneShelf : public IMGUIShelf<Cuda::Host::Plane, Cuda::PlaneParams>
{
public:
    PlaneShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~PlaneShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new PlaneShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
};

// Sphere tracable
class SphereShelf : public IMGUIShelf<Cuda::Host::Sphere, Cuda::TracableParams>
{
public:
    SphereShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~SphereShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SphereShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
};

// Quad 
class QuadLightShelf : public IMGUIShelf<Cuda::Host::QuadLight, Cuda::QuadLightParams >
{
public:
    QuadLightShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~QuadLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new QuadLightShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
};

// Sphere light
class SphereLightShelf : public IMGUIShelf<Cuda::Host::SphereLight, Cuda::SphereLightParams >
{
public:
    SphereLightShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~SphereLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SphereLightShelf(json)); }
    virtual void Construct() override final;
    virtual void Randomise(const Cuda::vec2 range) override final;
};

// Environment light
class EnvironmentLightShelf : public IMGUIShelf<Cuda::Host::EnvironmentLight, Cuda::EnvironmentLightParams>
{
public:
    EnvironmentLightShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~EnvironmentLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new EnvironmentLightShelf(json)); }
    virtual void Construct() override final;
};

// Lambertian BRDF
class LambertBRDFShelf : public IMGUIShelf<Cuda::Host::LambertBRDF, Cuda::NullParams>
{
public:
    LambertBRDFShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~LambertBRDFShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new LambertBRDFShelf(json)); }
    virtual void Construct() override final;
};

// Wavefront tracer
class WavefrontTracerShelf : public IMGUIShelf<Cuda::Host::WavefrontTracer, Cuda::WavefrontTracerParams>
{
public:
    WavefrontTracerShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~WavefrontTracerShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new WavefrontTracerShelf(json)); }
    virtual void Construct() override final;
};

// Perspective camera
class PerspectiveCameraShelf : public IMGUIShelf<Cuda::Host::PerspectiveCamera, Cuda::PerspectiveCameraParams>
{
public:
    PerspectiveCameraShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~PerspectiveCameraShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new PerspectiveCameraShelf(json)); }
    virtual void Construct() override final;
};

// Light probe camera
class LightProbeCameraShelf : public IMGUIShelf<Cuda::Host::LightProbeCamera, Cuda::LightProbeCameraParams>
{
public:
    LightProbeCameraShelf(const Json::Node& json);
    virtual ~LightProbeCameraShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new LightProbeCameraShelf(json)); }
    virtual void Construct() override final;

private:
    std::vector<std::string>    m_swizzleLabels;
    std::vector<char>           m_pathData;
    std::string                 m_usdExportPath[2];
};

// Fisheye camera
class FisheyeCameraShelf : public IMGUIShelf<Cuda::Host::FisheyeCamera, Cuda::FisheyeCameraParams>
{
public:
    FisheyeCameraShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~FisheyeCameraShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new FisheyeCameraShelf(json)); }
    virtual void Construct() override final;
};

// Cornell box
class CornellBoxShelf : public IMGUIShelf<Cuda::Host::CornellBox, Cuda::CornellBoxParams>
{
public:
    CornellBoxShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~CornellBoxShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new CornellBoxShelf(json)); }
    virtual void Construct() override final;
};