#pragma once

#include "Camera.cuh"
#include "../SceneObject.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    class LineSegment;
    class Ellipse;
    class UIHandle;

    struct PerspectiveCameraObjects
    {
        __host__ __device__ PerspectiveCameraObjects() {}

        __host__ __device__ void Validate() const
        {
            CudaAssert(accumBuffer);
            CudaAssert(ui.lineSegments);
            CudaAssert(ui.handles);
        }
        
        Device::AccumulationBuffer* accumBuffer = nullptr;
        struct
        {
            Generic::Vector<LineSegment>* lineSegments = nullptr;
            Generic::Vector<UIHandle>* handles = nullptr;
        }
        ui;
    };

    struct PerspectiveCameraParams
    {
        __host__ __device__ PerspectiveCameraParams() : cameraPos(0.0f), fov(50.0f) {}
        __device__ void Validate() const {}

        vec2    cameraPos;          // Position of the camera
        vec2    cameraAxis;         // View axis of the camera
        float   fov;                // Field of view in degrees

        mat2    fwdBasis;
        mat2    invBasis;
    };

    namespace Host
    {
        class PerspectiveCamera;
    }

    namespace Device
    {
        class PerspectiveCamera : public Device::Camera
        {
            friend class Host::PerspectiveCamera;
        public:
            __device__ PerspectiveCamera() {}

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __host__ __device__ virtual vec4            EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;
            __device__ void                             Synchronise(const PerspectiveCameraParams& params) { m_params = params; }
            __device__ void                             Synchronise(const PerspectiveCameraObjects& objects) { m_objects = objects; }

        private:
            PerspectiveCameraParams m_params;
            PerspectiveCameraObjects m_objects;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class PerspectiveCamera : public Host::Camera
                                  
        {
        public:
            __host__ PerspectiveCamera(const Asset::InitCtx& initCtx, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ virtual ~PerspectiveCamera() noexcept;

            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;
            //__host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            __host__ virtual bool       OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       Rebuild() override final;

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ static const std::string  GetAssetClassStatic() { return "perspectivecamera"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual void              Synchronise(const uint syncType) override final;

            __host__ virtual Device::PerspectiveCamera* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;

            __host__ virtual BBox2f     GetObjectSpaceBoundingBox() override final { return m_objectSpaceBBox; }

        protected:
            __host__ virtual bool       OnCreateSceneObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;

            __host__ void               UpdateObjectSpaceBoundingBox();

        private:
            __host__ void               ConstructUIWireframes();
            __host__ void               ConstructUIHandlesFromAxis(const vec2& cameraAxis);

            __host__ UIHandle&          GetOriginHandle();
            __host__ UIHandle&          GetAxisHandle();

            Device::PerspectiveCamera*  cu_deviceInstance = nullptr;
            Device::PerspectiveCamera   m_hostInstance;
            PerspectiveCameraObjects    m_deviceObjects;

            BBox2f                      m_objectSpaceBBox;

            struct
            {
                AssetHandle<Host::Vector<LineSegment>>  hostLineSegments;
                AssetHandle<Host::Vector<UIHandle>>     hostUIHandles;
            }
            m_ui;

            struct
            {
                bool isCentroidSet;
            }
            m_onCreate;
        };
    }
}