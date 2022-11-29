#pragma once

#include "Common.cuh"

#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "core/GenericObject.cuh"
#include "core/math/Math.cuh"

namespace Enso
{
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
        //static_assert(std::is_base_of<Device::SceneObject, ObjectType>::value, "cu_object not derived from Device::SceneObject");

        ParamsType* cu_params;
        IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
        IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

        IsOk(cudaDeviceSynchronize());
        KernelSynchroniseInheritedClass << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params, syncFlags);
        IsOk(cudaDeviceSynchronize());

        IsOk(cudaFree(cu_params));
    }

    enum AssetSyncType : int { kSyncObjects = 1, kSyncParams = 2 };

    enum SceneObjectFlags : uint
    {
        kSceneObjectSelected = 1u,
        kSceneObjectInteractiveElement = 2u
    };

    struct SceneObjectParams
    {
        __host__ __device__ SceneObjectParams() {}

        BBox2f                      m_objectBBox;
        BBox2f                      m_worldBBox;

        BidirectionalTransform2D    m_transform;
        uint                        m_attrFlags;
    };

    namespace Host { class SceneObject; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class SceneObject : public SceneObjectParams
        {
            friend Host::SceneObject;

        public:
            __device__ virtual vec4             EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx) const { return vec4(0.0f); }

            __host__ __device__ const BBox2f& GetObjectSpaceBoundingBox() const { return m_objectBBox; };
            __host__ __device__ const BBox2f& GetWorldSpaceBoundingBox() const { return m_worldBBox; };

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
        class SceneObject : public Host::GenericObject,
                            public SceneObjectParams
        {
        public:
            __host__ virtual bool       Finalise() = 0;

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) { return 0u; }
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx);
            __host__ virtual uint       OnSelect(const bool isSelected);

            __host__ virtual uint       GetDirtyFlags() const { return m_dirtyFlags; }
            __host__ virtual bool       IsFinalised() const { return m_isFinalised; }
            __host__ virtual bool       IsSelected() const { return m_attrFlags & kSceneObjectSelected; }
            __host__ virtual bool       IsConstructed() const { return false; }
            __host__ virtual const Host::SceneObject& GetSceneObject() const { return *this; }

            __host__ virtual const BBox2f& GetObjectSpaceBoundingBox() const { return m_objectBBox; }
            __host__ virtual const BBox2f& GetWorldSpaceBoundingBox() const { return m_worldBBox; }

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
            __host__ SceneObject(const std::string& id, Device::SceneObject& hostInstance);

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

            Device::SceneObject& m_hostInstance;
        };
    }
}
