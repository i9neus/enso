#pragma once

#include "Common.cuh"

#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "kernels/CudaRenderObject.cuh"

using namespace Cuda;

namespace GI2D
{
    enum SceneObjectFlags : uint
    {
        kSceneObjectSelected = 1u
    };

    struct SceneObjectParams
    {
        __host__ __device__ SceneObjectParams() {}

        BBox2f                      objectBBox;
        BBox2f                      worldBBox;

        BidirectionalTransform2D    transform;
        uint                        attrFlags;
    };

    // This class provides an interface for querying the tracable via geometric operations
    class SceneObjectInterface
    {
    public:
        __device__ virtual bool                             EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx, vec4& L) const { return false; }

        __host__ __device__ const BBox2f&                   GetObjectSpaceBoundingBox() const { return m_params.objectBBox; };
        __host__ __device__ const BBox2f&                   GetWorldSpaceBoundingBox() const { return m_params.worldBBox; };

        __device__ void                                     Synchronise(const SceneObjectParams& params);

    protected:
        __host__ __device__ SceneObjectInterface() {}

        __device__ bool                                     EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            return m_params.transform.RayToObjectSpace(world);
        }

    protected:
        SceneObjectParams m_params;

    private:
        BBox2f m_handleInnerBBox;
    };

    namespace Device
    {
        class SceneObject : virtual public SceneObjectInterface,
                            public Cuda::Device::RenderObject
        {
        public:
            __device__ SceneObject() {}
        };
    }

    namespace Host
    {
        class SceneObject : virtual public SceneObjectInterface,
                            public Cuda::Host::RenderObject
        {
        public:
            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) = 0;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) = 0;
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx);
            __host__ virtual uint       OnSelect(const bool isSelected);

            __host__ virtual bool       Finalise() = 0;

            __host__ virtual bool       IsConstructed() const = 0;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) = 0;

            __host__ uint               GetDirtyFlags() const { return m_dirtyFlags; }
            __host__ bool               IsFinalised() const { return m_isFinalised; }
            __host__ bool               IsSelected() const { return m_params.attrFlags & kSceneObjectSelected; }

            __host__ SceneObjectInterface* GetDeviceInstance() const
            {
                AssertMsgFmt(cu_deviceSceneObjectInterface, "SceneObjectInterface::cu_deviceSceneObjectInterface has not been initialised by its inheriting class '%s'", GetAssetID().c_str());
                return cu_deviceSceneObjectInterface;
            }

            __host__ virtual void SetAttributeFlags(const uint flags, bool isSet = true)
            {
                if (SetGenericFlags(m_params.attrFlags, flags, isSet))
                {
                    SetDirtyFlags(kGI2DDirtyUI, true);
                }
            }

        protected:
            __host__ SceneObject(const std::string& id);

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

            SceneObjectInterface* cu_deviceSceneObjectInterface;
        };
    }
}