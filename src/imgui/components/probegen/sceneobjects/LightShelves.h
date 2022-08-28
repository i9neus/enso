#pragma once

#include "thirdparty/imgui/imgui.h"

#include "kernels/lights/CudaQuadLight.cuh"
#include "kernels/lights/CudaSphereLight.cuh"
#include "kernels/lights/CudaEnvironmentLight.cuh"
#include "kernels/lights/CudaDistantLight.cuh"

#include "IMGUIAbstractShelf.h"

// Quad light
class QuadLightShelf : public IMGUIShelf<Cuda::Host::QuadLight, Cuda::QuadLightParams >
{
public:
    QuadLightShelf(const Json::Node& json);
    virtual ~QuadLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new QuadLightShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredColourPicker   m_colourPicker;
    IMGUIJitteredParameter      m_intensity;
    IMGUIJitteredFlagArray      m_flags;
};

// Sphere light
class SphereLightShelf : public IMGUIShelf<Cuda::Host::SphereLight, Cuda::SphereLightParams >
{
public:
    SphereLightShelf(const Json::Node& json);
    virtual ~SphereLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SphereLightShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredColourPicker   m_colourPicker;
    IMGUIJitteredParameter      m_intensity;
    IMGUIJitteredFlagArray      m_flags;
};

// Distant light
class DistantLightShelf : public IMGUIShelf<Cuda::Host::DistantLight, Cuda::DistantLightParams >
{
public:
    DistantLightShelf(const Json::Node& json);
    virtual ~DistantLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new DistantLightShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredColourPicker   m_colourPicker;
    IMGUIJitteredParameter      m_intensity;
    IMGUIJitteredFlagArray      m_flags;
};

// Environment light
class EnvironmentLightShelf : public IMGUIShelf<Cuda::Host::EnvironmentLight, Cuda::EnvironmentLightParams >
{
public:
    EnvironmentLightShelf(const Json::Node& json);
    virtual ~EnvironmentLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new EnvironmentLightShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredColourPicker   m_colourPicker;
    IMGUIJitteredParameter      m_intensity;
    IMGUIJitteredFlagArray      m_flags;
};