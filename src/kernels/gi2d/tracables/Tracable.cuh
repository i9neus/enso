#pragma once

#include "../SceneObject.cuh"

using namespace Cuda;

namespace GI2D
{
    enum TracableFlags : uint
    {
        kTracableSelected = 1u
    };

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Tracable : public Device::SceneObject
        {
        public:
            __host__ __device__ virtual bool                    IntersectRay(Ray2D& ray, HitCtx2D& hit) const { return false; }
            //__host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
            __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const;

            //__host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }

            __device__ virtual vec4                             EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final 
            { 
                return EvaluatePrimitives(pWorld, viewCtx); 
            }
            __host__ __device__ virtual const vec3              GetColour() const { return kOne; }

        protected:
            __host__ __device__ Tracable() {}

            __device__ virtual vec4                             EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const { return vec4(0.0f); }

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
        class TracableInterface : virtual public SceneObjectInterface
        {
        public:
            __host__ virtual Device::Tracable*  GetDeviceInstance() const = 0;
        };
        
        template<typename DeviceType>
        class Tracable : virtual public TracableInterface,
                         public Host::SceneObject<DeviceType>
        {
            using Super = Host::SceneObject<DeviceType>;
        public:
            __host__ virtual Device::Tracable* GetDeviceInstance() const override { return cu_deviceTracableInstance; }

        protected:
            __host__ Tracable(const std::string& id) : Super(id) {}
            
            template<typename SubType> __host__ void Synchronise(SubType* deviceData, const int syncType) { SceneObject<DeviceType>::Synchronise(deviceData, syncType); }

        protected:
            Device::Tracable*               cu_deviceTracableInstance = nullptr;

            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;
        };
    }
}