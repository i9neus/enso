#pragma once

#include "core/2d/Ctx.cuh"
#include "core/3d/Ctx.cuh"
#include "core/GenericObject.cuh"
#include "core/3d/Transform.cuh"

#include "core/DirtinessFlags.cuh"
#include "core/Image.cuh"
#include "core/GenericObject.cuh"
#include "core/HighResolutionTimer.h"

#include "io/Serialisable.cuh"

#include "../../FwdDecl.cuh"

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

        BidirectionalTransform transform;

        int lightIdx = kTracableNotALight;
    };
     
    namespace Host { class Tracable; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Tracable : public Device::GenericObject
        {
            friend class Host::Tracable;
        public:
            __host__ __device__ virtual bool        IntersectRay(const Ray& ray, HitCtx& hit) const { return false; }
            //__host__ __device__ virtual bool      InteresectPoint(const vec2& p, const float& thickness) const { return false; }
            //__host__ __device__ virtual bool        IntersectBBox(const BBox2f& bBox) const;

            //__host__ __device__ virtual vec2      PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }
         
            __host__ __device__ virtual vec3        GetColour() const { return kOne; }
            __host__ __device__ virtual int         GetLightIdx() const { return m_params.lightIdx; }
            __device__ void                         Synchronise(const TracableParams& params) { m_params = params; }

        protected:
            __host__ __device__ Tracable() {}

            __host__ __device__ __forceinline__ RayBasic RayToObjectSpace(const Ray& world) const { return m_params.transform.RayToObjectSpace(world.od); }
            __host__ __device__ __forceinline__ vec3 NormalToWorldSpace(const vec3& object) const { return m_params.transform.NormalToWorldSpace(object); }
            //__host__ __device__ __forceinline__ vec3 PointToWorldSpace(const vec3& world) const { return m_params.transform.PointToWorldSpace(world); }

            TracableParams m_params;

        private:
            BBox2f m_handleInnerBBox;
        };
    }

    namespace Host
    {        
        class Tracable : public Host::GenericObject
        {
        public:
            __host__ virtual void       SetLightIdx(const int idx) { m_params.lightIdx = idx; }
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;

            __host__ virtual void       Synchronise(const uint syncFlags) override final;

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:
            __host__ Tracable(const Asset::InitCtx& initCtx);

            __host__ void               SetDeviceInstance(Device::Tracable* deviceInstance) { cu_deviceInstance = deviceInstance; }
            
            __host__ virtual void       OnSynchroniseTracable(const uint syncFlags) = 0;
        protected:
            Device::Tracable*               cu_deviceInstance;

            TracableParams                  m_params;
        };
    }
}