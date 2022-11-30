#pragma once

#include "../SceneObject.cuh"

namespace Enso
{
    enum TracableFlags : int
    {
        kTracableNotALight = -1
    };

    struct TracableParams
    {
        __host__ __device__ TracableParams() {}

        int m_lightIdx = kTracableNotALight;
    };
     
    namespace Host { class Tracable; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Tracable : public Device::SceneObject,
                         public TracableParams
        {
            friend class Host::Tracable;
        public:
            __host__ __device__ virtual bool                    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const { return false; }
            //__host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
            __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const;

            //__host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }
         
            __host__ __device__ virtual vec3                    GetColour() const { return kOne; }
            __host__ __device__ virtual int                     GetLightIdx() const { return m_lightIdx; }

        protected:
            __host__ __device__ Tracable() {}

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
        class Tracable : public Host::SceneObject,
                         public TracableParams

        {
        public:
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual void       SetLightIdx(const int idx) { m_lightIdx = idx; }
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:
            __host__ Tracable(const std::string& id, Device::Tracable& hostInstance) : 
                SceneObject(id, hostInstance),
                m_hostInstance(hostInstance)
            {
            }
            
            template<typename SubType> __host__ inline void Synchronise(SubType* deviceInstance, const int syncType) 
            { 
                SceneObject::Synchronise(deviceInstance, syncType); 

                if (syncType & kSyncParams) { SynchroniseInheritedClass<TracableParams>(deviceInstance, *this, kSyncParams); }
            }

        protected:
            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;

            Device::Tracable&               m_hostInstance;
        };
    }
}