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
        __device__ void Validate() const {}

        int lightIdx = kTracableNotALight;
    };
     
    namespace Host { class Tracable; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Tracable : public Device::SceneObject
        {
            friend class Host::Tracable;
        public:
            __host__ __device__ virtual bool        IntersectRay(const Ray2D& ray, HitCtx2D& hit) const { return false; }
            //__host__ __device__ virtual bool      InteresectPoint(const vec2& p, const float& thickness) const { return false; }
            __host__ __device__ virtual bool        IntersectBBox(const BBox2f& bBox) const;

            //__host__ __device__ virtual vec2      PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }
         
            __host__ __device__ virtual vec3        GetColour() const { return kOne; }
            __host__ __device__ virtual int         GetLightIdx() const { return m_params.lightIdx; }
            __device__ void                         Synchronise(const TracableParams& params) { m_params = params; }

        protected:
            __host__ __device__ Tracable() {}

            __host__ __device__ __forceinline__ RayBasic2D RayToObjectSpace(const Ray2D& world) const { return SceneObject::m_params.transform.RayToObjectSpace(world); }
            __host__ __device__ __forceinline__ vec2 NormalToWorldSpace(const vec2& object) const { return SceneObject::m_params.transform.NormalToWorldSpace(object); }
            __host__ __device__ __forceinline__ vec2 PointToWorldSpace(const vec2& world) const { return SceneObject::m_params.transform.PointToWorldSpace(world); }

            TracableParams m_params;

        private:
            BBox2f m_handleInnerBBox;
        };
    }

    namespace Host
    {        
        class Tracable : public Host::SceneObject
        {
        public:
            __host__ virtual void       SetLightIdx(const int idx) { m_hostInstance.m_params.lightIdx = idx; }
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override;

            __host__ virtual bool       HasOverlay() const override { return true; }


        protected:
            __host__ Tracable(const Asset::InitCtx& initCtx, Device::Tracable& hostInstance, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ void SetDeviceInstance(Device::Tracable* deviceInstance);
            
            __host__ virtual void Synchronise(const uint syncFlags) override;

        protected:
            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;

            Device::Tracable&               m_hostInstance;
            Device::Tracable*               cu_deviceInstance;
        };
    }
}