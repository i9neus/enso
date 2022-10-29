#pragma once

#include "Common.cuh"

#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "kernels/CudaRenderObject.cuh"

using namespace Cuda;

template<typename ObjectType, typename ParamsType>
__global__ static void KernelSynchroniseInheritedClass(ObjectType* cu_object, const size_t hostParamsSize, const ParamsType* cu_params)
{
    // Check that the size of the object in the device matches that of the host. Empty base optimisation can bite us here. 
    assert(cu_object);
    assert(cu_params);
    assert(sizeof(ParamsType) == hostParamsSize);

    ParamsType& cast = static_cast<ParamsType&>(*cu_object);
    cast = *cu_params;
}

template<typename ParamsType, typename ObjectType>
__host__ void SynchroniseInheritedClass(ObjectType* cu_object, const ParamsType& params)
{
    //AssertIsTransferrableType<ParamsType>();
    Assert(cu_object);

    ParamsType* cu_params;
    IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
    IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

    IsOk(cudaDeviceSynchronize());
    KernelSynchroniseInheritedClass << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
    IsOk(cudaDeviceSynchronize());

    IsOk(cudaFree(cu_params));
}

enum AssetSyncType : int { kSyncObjects = 1, kSyncParams = 2 };

namespace GI2D
{
    enum SceneObjectFlags : uint
    {
        kSceneObjectSelected = 1u
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
            __device__ virtual bool EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx, vec4& L) const { return false; }

            __host__ __device__ const BBox2f& GetObjectSpaceBoundingBox() const { return m_objectBBox; };
            __host__ __device__ const BBox2f& GetWorldSpaceBoundingBox() const { return m_worldBBox; };

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
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) = 0;
            __host__ virtual uint       OnSelect(const bool isSelected) = 0;
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
        
        template<typename DeviceSuper>
        class SceneObject : public DeviceSuper,
                            virtual public SceneObjectInterface,
                            public Cuda::Host::RenderObject
        {
        public:
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) override;
            __host__ virtual uint       OnSelect(const bool isSelected) override;

            __host__ virtual uint       GetDirtyFlags() const override final { return m_dirtyFlags; }
            __host__ virtual bool       IsFinalised() const override final { return m_isFinalised; }
            __host__ virtual bool       IsSelected() const override final { return m_attrFlags & kSceneObjectSelected; }
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
                DeviceSuper::m_attrFlags = 1;
                /*if (SetGenericFlags(m_attrFlags, flags, isSet))
                {
                    SetDirtyFlags(kGI2DDirtyUI, true);
                }*/
            }

        protected:
            __host__ SceneObject(const std::string& id);

            template<typename SubType>
            __host__ void Synchronise(SubType* cu_object, const int type)
            {
                if (type == kSyncParams) { SynchroniseInheritedClass<SceneObjectParams>(cu_object, *this); }
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
    }
}
