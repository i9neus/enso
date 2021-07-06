#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"

#include "kernels/tracables/CudaKIFS.cuh"
#include "kernels/tracables/CudaSphere.cuh"
#include "kernels/tracables/CudaPlane.cuh"
#include "kernels/tracables/CudaCornellBox.cuh"

#include "kernels/lights/CudaQuadLight.cuh"
#include "kernels/lights/CudaEnvironmentLight.cuh"

#include "kernels/bxdfs/CudaLambert.cuh"

#include "kernels/materials/CudaSimpleMaterial.cuh"
#include "kernels/materials/CudaCornellMaterial.cuh"

#include "kernels/CudaPerspectiveCamera.cuh"

#include "kernels/CudaWavefrontTracer.cuh"

namespace Json { class Document; class Node; }

class IMGUIAbstractShelf
{
public:
    IMGUIAbstractShelf() = default;
    virtual void Construct() = 0;
    virtual bool Update(std::string& newJson) = 0;
    const std::string& GetDAGPath() const { return m_dagPath; }
    const std::string& GetID() const { return m_id; }
    void SetIDAndDAGPath(const std::string& id, const std::string& dagPath)
    {
        m_id = id;
        m_dagPath = dagPath;
    }          

protected:
    std::string m_dagPath;
    std::string m_id;
};

template<typename ObjectType, typename ParamsType>
class IMGUIShelf : public IMGUIAbstractShelf
{
public:
    IMGUIShelf(const Json::Node& json)
    {
        m_params[0].FromJson(json, Json::kRequiredWarn);
        m_params[1] = m_params[0];
    }

    virtual ~IMGUIShelf() = default;

    virtual bool Update(std::string& newJson) override final
    {
        //if (m_params[0] == m_params[1]) { return false; }
        if (!IsDirty()) { return false; }
        m_params[1] = m_params[0];

        Json::Document newNode;
        m_params[0].ToJson(newNode);
        newJson = newNode.Stringify();

        return true;
    }

    void ConstructTransform(Cuda::BidirectionalTransform& transform)
    {
        if (ImGui::TreeNode("Transform"))
        {
            ImGui::DragFloat3("Position", &transform.trans[0], math::max(0.01f, cwiseMax(transform.trans) * 0.01f));
            ImGui::DragFloat3("Rotation", &transform.rot[0], math::max(0.01f, cwiseMax(transform.rot) * 0.01f));            
            //ImGui::DragFloat3("Scale XYZ", &transform.scale[0], math::max(0.01f, cwiseMax(transform.scale) * 0.01f));
            ImGui::DragFloat("Scale XYZ", &transform.scale[0], math::max(0.01f, cwiseMax(transform.scale) * 0.01f));
            transform.scale = transform.scale[0];
            ImGui::TreePop();
        }
    }

    bool IsDirty() const
    {
        static_assert(std::is_standard_layout<ParamsType>::value, "ParamsType must be standard layout.");

        for (int i = 0; i < sizeof(ParamsType); i++)
        {
            if (reinterpret_cast<const unsigned char*>(&m_params[0])[i] != reinterpret_cast<const unsigned char*>(&m_params[1])[i]) { return true; }
        }
        return false;
    }

    std::string GetShelfTitle()
    {
        const auto& assetDescription = ObjectType::GetAssetDescriptionString();
        if (assetDescription.empty())
        {
            return tfm::format("%s", m_id);
        }

        return tfm::format("%s: %s", ObjectType::GetAssetDescriptionString(), m_id);
    }

protected:
    std::array<ParamsType, 2> m_params;
};

// Simple material
class SimpleMaterialShelf : public IMGUIShelf<Cuda::Host::SimpleMaterial, Cuda::SimpleMaterialParams>
{
public:
    SimpleMaterialShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~SimpleMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json)  {  return std::shared_ptr<IMGUIShelf>(new SimpleMaterialShelf(json)); }
    virtual void Construct() override final;
};

// Simple material
class CornellMaterialShelf : public IMGUIShelf<Cuda::Host::CornellMaterial, Cuda::CornellMaterialParams>
{
public:
    CornellMaterialShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~CornellMaterialShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new CornellMaterialShelf(json)); }
    virtual void Construct() override final;
};

// Plane tracable
class PlaneShelf : public IMGUIShelf<Cuda::Host::Plane, Cuda::PlaneParams>
{
public:
    PlaneShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~PlaneShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new PlaneShelf(json)); }
    virtual void Construct() override final;
};

// Sphere tracable
class SphereShelf : public IMGUIShelf<Cuda::Host::Sphere, Cuda::TracableParams>
{
public:
    SphereShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~SphereShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node & json) { return std::shared_ptr<IMGUIShelf>(new SphereShelf(json)); }
    virtual void Construct() override final;
};

// KIFS tracable
class KIFSShelf : public IMGUIShelf<Cuda::Host::KIFS, Cuda::KIFSParams>
{
public:
    KIFSShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~KIFSShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new KIFSShelf(json)); }
    virtual void Construct() override final;
};

// Quad light
class QuadLightShelf : public IMGUIShelf<Cuda::Host::QuadLight, Cuda::QuadLightParams >
{
public:
    QuadLightShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~QuadLightShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new QuadLightShelf(json)); }
    virtual void Construct() override final;
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

// Cornell box
class CornellBoxShelf : public IMGUIShelf<Cuda::Host::CornellBox, Cuda::CornellBoxParams>
{
public:
    CornellBoxShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~CornellBoxShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new CornellBoxShelf(json)); }
    virtual void Construct() override final;
};

class IMGUIShelfFactory
{
public:
    IMGUIShelfFactory();

    std::vector<std::shared_ptr<IMGUIAbstractShelf>> Instantiate(const Json::Document& document, const Cuda::RenderObjectContainer& objectContainer);

private:
    std::map<std::string, std::function<std::shared_ptr<IMGUIAbstractShelf>(const ::Json::Node&)>>    m_instantiators;
};