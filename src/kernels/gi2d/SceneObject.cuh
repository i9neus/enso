#pragma once

#include "Common.cuh"

#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "kernels/CudaRenderObject.cuh"

using namespace Cuda;

template<typename ObjectType, typename ParamsType>
__global__ static void KernelSynchroniseInheritedClass(ObjectType* cu_object, const size_t hostParamsSize, const ParamsType* cu_params, const int syncFlags)
{
    // Check that the size of the object in the device matches that of the host. Empty base optimisation can bite us here. 
    assert(cu_object);
    assert(cu_params);
    assert(sizeof(ParamsType) == hostParamsSize);

    ParamsType& cast = static_cast<ParamsType&>(*cu_object);
    cast = *cu_params;

    cu_object->OnSynchronise(syncFlags);
}

template<typename ParamsType, typename ObjectType>
__host__ void SynchroniseInheritedClass(ObjectType* cu_object, const ParamsType& params, const int syncFlags)
{
    //AssertIsTransferrableType<ParamsType>();
    Assert(cu_object);
    static_assert(std::is_standard_layout< ParamsType>::value, "SynchroniseInheritedClass: ParamsType is not standard layout type");
    static_assert(std::is_base_of<ParamsType, ObjectType>::value, "SynchroniseInheritedClass: ObjectType not derived from ParamsType");
    //static_assert(std::is_base_of<Device::RenderObject, ObjectType>::value, "cu_object not derived from Device::SceneObject");

    ParamsType* cu_params;
    IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
    IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

    IsOk(cudaDeviceSynchronize());
    KernelSynchroniseInheritedClass << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params, syncFlags);
    IsOk(cudaDeviceSynchronize());

    IsOk(cudaFree(cu_params));
}

enum AssetSyncType : int { kSyncObjects = 1, kSyncParams = 2 };

namespace GI2D
{
    enum SceneObjectFlags : uint
    {
        kSceneObjectSelected                = 1u,
        kSceneObjectInteractiveElement      = 2u
    };

    struct SceneObjectParams
    {
        __host__ __device__ SceneObjectParams() {}

        BBox2f                      m_objectBBox;
        BBox2f                      m_worldBBox;

        BidirectionalTransform2D    m_transform;
        uint                        m_attrFlags;
    };

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class SceneObject : public SceneObjectParams
        {
        public:
            __device__ virtual vec4             EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx) const { return vec4(0.0f); }

            __host__ __device__ const BBox2f&   GetObjectSpaceBoundingBox() const { return m_objectBBox; };
            __host__ __device__ const BBox2f&   GetWorldSpaceBoundingBox() const { return m_worldBBox; };
            
            __device__ virtual void OnSynchronise(const int) {}

        protected:
            __device__ SceneObject() {}

            __device__ bool                                     EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;

            __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
            {
                return m_transform.RayToObjectSpace(world);
            }

        private:
            BBox2f m_handleInnerBBox;
        };
    }

    namespace Host
    {
        // Interface that hides the templated SceneObject<> class and permits us to use polymorphism
        class SceneObjectInterface
        {
        public:            
            SceneObjectInterface() = default;

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) = 0;
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) { return kGI2DClean;  };
            __host__ virtual uint       OnSelect(const bool isSelected) { return kGI2DClean; }
            __host__ virtual bool       Finalise() = 0;
            __host__ virtual bool       IsConstructed() const = 0;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual uint       GetDirtyFlags() const = 0;
            __host__ virtual bool       IsFinalised() const = 0;
            __host__ virtual bool       IsSelected() const = 0;
            __host__ virtual void       SetAttributeFlags(const uint flags, bool isSet = true) = 0;
            __host__ virtual const Cuda::Host::RenderObject& GetRenderObject() const = 0;
            __host__ virtual const      BBox2f& GetObjectSpaceBoundingBox() const = 0;
            __host__ virtual const      BBox2f& GetWorldSpaceBoundingBox() const = 0;            
        };
        
        template<typename DeviceType = Device::SceneObject>
        class SceneObject : public DeviceType,
                            virtual public SceneObjectInterface,
                            public Cuda::Host::RenderObject
        {
        public:
            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override { return 0u;  }
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) override;
            __host__ virtual uint       OnSelect(const bool isSelected) override;

            __host__ virtual uint       GetDirtyFlags() const override final { return m_dirtyFlags; }
            __host__ virtual bool       IsFinalised() const override final { return m_isFinalised; }
            __host__ virtual bool       IsSelected() const override final { return m_attrFlags & kSceneObjectSelected; }
            __host__ virtual bool       IsConstructed() const override { return false; }
            __host__ virtual const Cuda::Host::RenderObject& GetRenderObject() const override final { return *this; }

            __host__ virtual const BBox2f& GetObjectSpaceBoundingBox() const override final { return Device::SceneObject::GetObjectSpaceBoundingBox(); }
            __host__ virtual const BBox2f& GetWorldSpaceBoundingBox() const override final { return Device::SceneObject::GetWorldSpaceBoundingBox(); }

            /*__host__ Device::SceneObject* GetDeviceInstance() const
            {
                AssertMsgFmt(cu_deviceSceneObjectInterface, "Device::SceneObject::cu_deviceSceneObjectInterface has not been initialised by its inheriting class '%s'", GetAssetID().c_str());
                return cu_deviceSceneObjectInterface;
            }*/

            __host__ virtual void SetAttributeFlags(const uint flags, bool isSet = true) override final
            {                
                //DeviceSuper::m_attrFlags = 1;
                /*if (SetGenericFlags(m_attrFlags, flags, isSet))
                {
                    SetDirtyFlags(kGI2DDirtyUI, true);
                }*/
            }

            //__host__ virtual uint GetStaticAttributes() const { return StaticAttributes; }

        protected:
            __host__ SceneObject(const std::string& id);

            template<typename SubType>
            __host__ void Synchronise(SubType* cu_object, const int type)
            {
                if (type == kSyncParams) { SynchroniseInheritedClass<SceneObjectParams>(cu_object, *this, kSyncParams); }
            }

            __host__ void SetDirtyFlags(const uint flags, const bool isSet = true) { SetGenericFlags(m_dirtyFlags, flags, isSet); }
            __host__ void ClearDirtyFlags() { m_dirtyFlags = 0; }

        protected:
            uint                        m_dirtyFlags;
            bool                        m_isFinalised;

            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;

            Device::SceneObject* cu_deviceSceneObjectInterface = nullptr;
        };   

        template<typename DeviceType>
        __host__ Host::SceneObject<DeviceType>::SceneObject(const std::string& id) :
            RenderObject(id),
            m_dirtyFlags(kGI2DDirtyAll),
            m_isFinalised(false)
        {
        }

        template<typename DeviceType>
        __host__ uint Host::SceneObject<DeviceType>::OnSelect(const bool isSelected)
        {
            if (SetGenericFlags(m_attrFlags, uint(kSceneObjectSelected), isSelected))
            {
                SetDirtyFlags(kGI2DDirtyUI, true);
            }
            return m_dirtyFlags;
        }

        template<typename DeviceType>
        __host__ uint Host::SceneObject<DeviceType>::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
        {
            if (stateID == "kMoveSceneObjectBegin")
            {
                m_onMove.dragAnchor = viewCtx.mousePos;
                m_onMove.isDragging = true;
                Log::Error("kMoveSceneObjectBegin");
            }
            else if (stateID == "kMoveSceneObjectDragging")
            {
                Assert(m_onMove.isDragging);

                const vec2 delta = viewCtx.mousePos - m_onMove.dragAnchor;
                m_onMove.dragAnchor = viewCtx.mousePos;
                m_transform.trans += delta;
                m_worldBBox += delta;

                // The geometry internal to this object hasn't changed, but it will affect the 
                Log::Warning("kMoveSceneObjectDragging");
                SetDirtyFlags(kGI2DDirtyBVH);
            }
            else if (stateID == "kMoveSceneObjectEnd")
            {
                m_onMove.isDragging = false;
                Log::Success("kMoveSceneObjectEnd");
            }

            return m_dirtyFlags;
        }
    }
}
