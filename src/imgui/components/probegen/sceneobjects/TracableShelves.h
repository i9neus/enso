#pragma once

#include "thirdparty/imgui/imgui.h"

#include "kernels/tracables/CudaSphere.cuh"
#include "kernels/tracables/CudaPlane.cuh"
#include "kernels/tracables/CudaCornellBox.cuh"
#include "kernels/tracables/CudaBox.cuh"
#include "kernels/tracables/CudaKIFS.cuh"
#include "kernels/tracables/CudaSDF.cuh"

#include "IMGUIAbstractShelf.h"

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
    int                     m_stateListCurrentIdx;
    std::string             m_stateListCurrentId;
    std::vector<char>       m_stateIDData;
    std::string             m_stateJsonPath;

    IMGUIJitteredFlagArray      m_faceFlags;
    IMGUIJitteredFlagArray      m_sdfFlags;
    IMGUIJitteredParameterTable m_jitteredParamTable;
};

// SDF tracable
class SDFShelf : public IMGUIShelf<Cuda::Host::SDF, Cuda::SDFParams>
{
public:
    SDFShelf(const Json::Node& json);
    virtual ~SDFShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SDFShelf(json)); }
    virtual void            Construct() override final;
    virtual void            Update() override final;

    virtual void            Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredParameterTable m_jitteredParamTable;
};

// Plane tracable
class PlaneShelf : public IMGUIShelf<Cuda::Host::Plane, Cuda::PlaneParams>
{
public:
    PlaneShelf(const Json::Node& json);
    virtual ~PlaneShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new PlaneShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredFlagArray      m_flags;
};

// Sphere tracable
class SphereShelf : public IMGUIShelf<Cuda::Host::Sphere, Cuda::TracableParams>
{
public:
    SphereShelf(const Json::Node& json);
    virtual ~SphereShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new SphereShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredFlagArray  m_flags;
};

// Box tracable
class BoxShelf : public IMGUIShelf<Cuda::Host::Box, Cuda::TracableParams>
{
public:
    BoxShelf(const Json::Node& json);
    virtual ~BoxShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new BoxShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredFlagArray  m_flags;
};

// Cornell box
class CornellBoxShelf : public IMGUIShelf<Cuda::Host::CornellBox, Cuda::CornellBoxParams>
{
public:
    CornellBoxShelf(const Json::Node& json);
    virtual ~CornellBoxShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new CornellBoxShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;

private:
    IMGUIJitteredFlagArray  m_flags;
    IMGUIJitteredFlagArray  m_faceMask;
    IMGUIJitteredFlagArray  m_cameraRayMask;
};