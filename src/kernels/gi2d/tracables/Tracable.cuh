#pragma once

#include "../SceneObject.cuh"

using namespace Cuda;

namespace GI2D
{
    enum TracableFlags : uint
    {
        kTracableSelected = 1u
    };
    
    // This class provides an interface for querying the tracable via geometric operations
    class TracableInterface : virtual public SceneObjectInterface
    {
    public:
        __host__ __device__ virtual bool                    IntersectRay(Ray2D& ray, HitCtx2D& hit) const { return false; }
        //__host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
        __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const;

        //__host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }

        __device__ virtual vec4                             EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final { return EvaluatePrimitives(pWorld, viewCtx); }
        __host__ __device__ virtual const vec3              GetColour() const { return kOne; }

    protected:
        __host__ __device__ TracableInterface() {}

        __device__ virtual vec4                             EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const = 0;

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            return m_params.transform.RayToObjectSpace(world);
        }

    private:
        BBox2f m_handleInnerBBox;
    };
    
    namespace Device
    {
        class Tracable : virtual public TracableInterface,
                         public GI2D::Device::SceneObject
        {
        public:
            __device__ Tracable() {}            
        };
    }

    namespace Host
    {
        class Tracable : virtual public TracableInterface,
                         public GI2D::Host::SceneObject, 
                         public Cuda::AssetTags<Host::Tracable, TracableInterface>
        {
        public:
            __host__ virtual TracableInterface* GetDeviceInstance() const = 0;

        protected:
            __host__ Tracable(const std::string& id) : SceneObject(id) {}

        protected:
            bool                        m_isFinalised;

            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;
        };
    }
}