#pragma once

#include "core/assets/DirtinessFlags.cuh"

#include "Transform2D.cuh"
#include "core/2d/Ctx.cuh"

#include "core/assets/GenericObject.cuh"
#include "core/math/Math.cuh"
#include "core/ui/UIButtonMap.h"
#include "core/ui/VirtualKeyStates.h"

namespace Enso
{   
    enum DrawableObjectFlags : uint
    {
        kDrawableObjectSelected = 1u,
        kDrawableObjectInteractiveElement = 2u
    };

    enum DrawableObjectSelectType : int
    {
        kDrawableObjectInvalidSelect = 0,
        kDrawableObjectPrecisionDrag = 1,
        kDrawableObjectDelegatedAction = 2
    };

    struct DrawableObjectParams
    {
        __host__ __device__ DrawableObjectParams() :
            attrFlags(0)
        {}
        __device__ void Validate() const {}

        BBox2f                      objectBBox;
        BBox2f                      worldBBox;

        BidirectionalTransform2D    transform;
        uint                        attrFlags;
    };

    namespace Host { class DrawableObject; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class DrawableObject : public Device::GenericObject
        {
            friend Host::DrawableObject;

        public:
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const { return vec4(0.f); }

            __host__ __device__ const BBox2f&   GetObjectSpaceBoundingBox() const { return m_params.objectBBox; };
            __host__ __device__ const BBox2f&   GetWorldSpaceBoundingBox() const { return m_params.worldBBox; };

            __device__ void                     Synchronise(const DrawableObjectParams& params) { m_params = params; }
            
            __host__ __device__ const BidirectionalTransform2D& GetTransform() const { return m_params.transform; }
            __host__ __device__ const BBox2f&            GetWorldBBox() const { return m_params.worldBBox; }
            __host__ __device__ const BBox2f&            GetObjectBBox() const { return m_params.objectBBox; }            


        protected:
            __host__ __device__                 DrawableObject() {}

            __device__ bool                     EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;
            __host__ __device__ BidirectionalTransform2D& GetTransform() { return m_params.transform; }

            __host__ __device__ __forceinline__ vec2 ToObjectSpace(const vec2& pWorld) const
            {
                return pWorld - m_params.transform.trans;
            }

        protected:
            DrawableObjectParams    m_params;

        private:
            BBox2f                  m_handleInnerBBox;
        };
    }

    namespace Host
    {               
        class DrawableObject : public Host::GenericObject
        {
        public:
            //__host__ virtual bool       Finalise() = 0;

            __host__ virtual bool       Rebuild() override final;
            __host__ virtual void       Synchronise(const uint syncFlags) override final;
            __host__ void               Verify() const;
            
            __host__ bool               OnCreate(const std::string& stateID, const UIViewCtx& viewCtx);
            __host__ virtual bool       OnMove(const std::string& stateID, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);
            __host__ virtual bool       OnSelect(const bool isSelected);
            __host__ virtual bool       OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) { return false; }
            __host__ virtual void       OnPreDraw(const UIViewCtx& viewCtx) {}
            __host__ virtual bool       HasBoundingBox() const { return true; }

            __host__ virtual bool       IsFinalised() const { return m_isFinalised; }
            __host__ virtual bool       IsSelected() const { return m_hostInstance.m_params.attrFlags & kDrawableObjectSelected; }
            __host__ virtual bool       IsConstructed() const { return m_isConstructed; }
            __host__ virtual bool       HasOverlay() const { return false; }
            __host__ virtual bool       IsClickablePoint(const UIViewCtx& viewCtx) const = 0;
            __host__ virtual bool       IsDelegatable() const = 0;
            __host__ uint               OnMouseClick(const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) const;

            //__host__ virtual const Host::DrawableObject& GetDrawableObject() const { return *this; }
            __host__ virtual Device::DrawableObject* GetDeviceInstance() const = 0;
            
            __host__ const BBox2f& GetObjectSpaceBoundingBox() const { return m_hostInstance.m_params.objectBBox; }
            __host__ const BBox2f& GetWorldSpaceBoundingBox() const { return m_hostInstance.m_params.worldBBox; }

            __host__ static uint        GetInstanceFlags() { return 0u; }

            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual bool Deserialise(const Json::Node& rootNode, const int flags) override;

            /*__host__ Device::DrawableObject* GetDeviceInstance() const
            {
                AssertMsgFmt(cu_deviceDrawableObjectInterface, "Device::DrawableObject::cu_deviceDrawableObjectInterface has not been initialised by its inheriting class '%s'", GetAssetID().c_str());
                return cu_deviceDrawableObjectInterface;
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
            __host__ DrawableObject(const Asset::InitCtx& initCtx, Device::DrawableObject* hostInstance);
            
            __host__ void               SetDeviceInstance(Device::DrawableObject* deviceInstance);

            __host__ virtual bool       OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) = 0;
            __host__ virtual bool       OnRebuildDrawableObject() = 0;
            __host__ virtual void       OnSynchroniseDrawableObject(const uint syncFlags) = 0;

            __host__ void               SetTransform(const vec2& trans);
           
            __host__ BidirectionalTransform2D& GetTransform() { return m_hostInstance.m_params.transform; }
            __host__ const BidirectionalTransform2D& GetTransform() const { return m_hostInstance.m_params.transform; }

            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() = 0;
            __host__ void               RecomputeBoundingBoxes();

        protected:


        private:
            Device::DrawableObject&                         m_hostInstance;
            Device::DrawableObject*                         cu_deviceInstance = nullptr;
             
            struct
            {
                vec2 dragAnchorDelta;
            } 
            m_onMove;
        };
    }
}
