#pragma once

#include "DirtinessFlags.cuh"

#include "Transform2D.cuh"
#include "Ctx.cuh"

#include "core/GenericObject.cuh"
#include "core/math/Math.cuh"
#include "FwdDecl.cuh"

namespace Enso
{   
    enum SceneObjectFlags : uint
    {
        kSceneObjectSelected = 1u,
        kSceneObjectInteractiveElement = 2u
    };

    enum SceneObjectSelectType : int
    {
        kSceneObjectInvalidSelect = 0,
        kSceneObjectPrecisionDrag = 1,
        kSceneObjectDelegatedAction = 2
    };

    struct SceneObjectParams
    {
        __host__ __device__ SceneObjectParams() {}
        __device__ void Validate() const {}

        BBox2f                      objectBBox;
        BBox2f                      worldBBox;

        BidirectionalTransform2D    transform;
        uint                        attrFlags;
    };

    namespace Host { class SceneObject; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class SceneObject : public Device::GenericObject
        {
            friend Host::SceneObject;

        public:
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const { return vec4(0.f); }

            __host__ __device__ const BBox2f&   GetObjectSpaceBoundingBox() const { return m_params.objectBBox; };
            __host__ __device__ const BBox2f&   GetWorldSpaceBoundingBox() const { return m_params.worldBBox; };

            __device__ void                     Synchronise(const SceneObjectParams& params) { m_params = params; }
            
            __host__ __device__ const BidirectionalTransform2D& GetTransform() const { return m_params.transform; }
            __host__ __device__ const BBox2f&            GetWorldBBox() const { return m_params.worldBBox; }
            __host__ __device__ const BBox2f&            GetObjectBBox() const { return m_params.objectBBox; }            SceneObjectParams m_params;


        protected:
            __host__ __device__  SceneObject() {}

            __device__ bool                     EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;
            __host__ __device__ BidirectionalTransform2D& GetTransform() { return m_params.transform; }

            __host__ __device__ __forceinline__ vec2 ToObjectSpace(const vec2& pWorld) const
            {
                return pWorld - m_params.transform.trans;
            }

        private:
            BBox2f m_handleInnerBBox;
        };
    }

    namespace Host
    {               
        class SceneObject : public Host::GenericObject
        {
        public:
            //__host__ virtual bool       Finalise() = 0;

            __host__ virtual bool       Rebuild() { return true; }
            __host__ virtual bool       Prepare();
            
            __host__ bool               OnCreate(const std::string& stateID, const UIViewCtx& viewCtx);
            __host__ virtual bool       OnMove(const std::string& stateID, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);
            __host__ virtual bool       OnSelect(const bool isSelected);
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const { return kSceneObjectInvalidSelect; }
            __host__ virtual bool       OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) { return false; }
            __host__ virtual bool       IsTransformable() const { return true; }

            __host__ virtual bool       IsFinalised() const { return m_isFinalised; }
            __host__ virtual bool       IsSelected() const { return m_hostInstance.m_params.attrFlags & kSceneObjectSelected; }
            __host__ virtual bool       IsConstructed() const { return m_isConstructed; }
            __host__ virtual bool       HasOverlay() const { return false; }
            //__host__ virtual const Host::SceneObject& GetSceneObject() const { return *this; }
            __host__ virtual Device::SceneObject* GetDeviceInstance() const = 0;
            
            __host__ const BBox2f& GetObjectSpaceBoundingBox() { return m_hostInstance.m_params.objectBBox; }
            __host__ const BBox2f& GetWorldSpaceBoundingBox() { return m_hostInstance.m_params.worldBBox; }

            __host__ static uint        GetInstanceFlags() { return 0u; }

            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual bool Deserialise(const Json::Node& rootNode, const int flags) override;

            /*__host__ Device::SceneObject* GetDeviceInstance() const
            {
                AssertMsgFmt(cu_deviceSceneObjectInterface, "Device::SceneObject::cu_deviceSceneObjectInterface has not been initialised by its inheriting class '%s'", GetAssetID().c_str());
                return cu_deviceSceneObjectInterface;
            }*/

            __host__ virtual void SetAttributeFlags(const uint flags, bool isSet = true)
            {
                //DeviceSuper::m_attrFlags = 1;
                /*if (SetGenericFlags(m_attrFlags, flags, isSet))
                {
                    SetDirtyFlags(kGI2DDirtyUI, true);
                }*/
            }

            //__host__ virtual uint GetStaticAttributes() const { return StaticAttributes; }

        protected:
            __host__ SceneObject(const Asset::InitCtx& initCtx, Device::SceneObject* hostInstance, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ void SetDeviceInstance(Device::SceneObject* deviceInstance);
            __host__ virtual void       Synchronise(const uint flags) override;

            __host__ virtual bool       OnCreateSceneObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) = 0;

            __host__ void               SetTransform(const vec2& trans);
           
            __host__ BidirectionalTransform2D& GetTransform() { return m_hostInstance.m_params.transform; }
            __host__ const BidirectionalTransform2D& GetTransform() const { return m_hostInstance.m_params.transform; }

            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() = 0;
            __host__ void               RecomputeBoundingBoxes();

        protected:
            Device::SceneObject&                        m_hostInstance;
            AssetHandle<const Host::SceneContainer>     m_scene;

        private:
            Device::SceneObject*                        cu_deviceInstance = nullptr;

            struct
            {
                vec2 dragAnchorDelta;
            } 
            m_onMove;
        };
    }
}
