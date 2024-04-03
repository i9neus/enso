#pragma once

#include "Camera2D.cuh"
#include "../FwdDecl.cuh"
#include "../primitives/Ellipse.cuh"

namespace Enso
{
    class LineSegment;

    struct PerspectiveCameraObjects
    {
        __host__ __device__ PerspectiveCameraObjects() {}

        __host__ __device__ PerspectiveCameraObjects& operator=(const PerspectiveCameraObjects& other)
        {            
            return *this;
        }
    };

    struct PerspectiveCameraParams
    {
        __host__ __device__ PerspectiveCameraParams() : m_lightRadius(0.0f), m_lightColour(1.0f), m_lightIntensity(0.0f) {}

        float   m_lightRadius;

        vec3    m_lightColour;
        float   m_lightIntensity;
    };

    namespace Host
    {
        class PerspectiveCamera;
    }

    namespace Device
    {
        class PerspectiveCamera : public Device::Camera2D,
                                  public PerspectiveCameraParams,
                                  public PerspectiveCameraObjects
        {
            friend class Host::PerspectiveCamera;

        public:
            __device__ PerspectiveCamera() {}

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __host__ __device__ virtual uint            OnMouseClick(const UIViewCtx& viewCtx) const override final;
            __host__ __device__ virtual vec4            EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;

            __host__ __device__ virtual void            OnSynchronise(const int) override final;

        private:
            Ellipse m_primitive;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class PerspectiveCamera : public Host::Camera2D
        {
        public:
            __host__ PerspectiveCamera(const std::string& id);
            __host__ virtual ~PerspectiveCamera();

            __host__ virtual void       OnDestroyAsset() override final;

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;
            //__host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx);

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&);
            __host__ static const std::string  GetAssetClassStatic() { return "omnilight"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ void               Synchronise(const int syncType);

            __host__ virtual Device::PerspectiveCamera* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override final;

        protected:
            __host__ virtual BBox2f     RecomputeObjectSpaceBoundingBox() override final;

        private:
            Device::PerspectiveCamera* cu_deviceInstance = nullptr;
            Device::PerspectiveCamera           m_hostInstance;

            struct
            {
                bool isCentroidSet;
            }
            m_onCreate;
        };
    }
}