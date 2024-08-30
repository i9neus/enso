#pragma once

#include "core/2d/Ctx.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Transform.cuh"

#include "core/assets/DirtinessFlags.cuh"
#include "core/containers/Image.cuh"
#include "core/assets/GenericObject.cuh"
#include "core/utils/HighResolutionTimer.h"

#include "io/Serialisable.cuh"

#include "../SceneObject.cuh"
#include "../../FwdDecl.cuh"
#include "../pointclouds/GaussianPointCloud.cuh"

namespace Enso
{
    namespace Host 
    { 
        class Tracable; 
        class LightSampler;
    }
    
    enum TracableFlags : int
    {
        kTracableNotALight = -1
    };

    struct TracableParams
    {
        __host__ __device__ TracableParams() :
            materialIdx(-1),
            radiance(0.)
        {}

        __device__ void Validate() const {}

        BidirectionalTransform      transform;
        int                         materialIdx;
        vec3                        radiance;
        bool                        isLight;
    };
     
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Tracable : public Device::SceneObject
        {
        public:
            __device__ virtual ~Tracable() noexcept {}

            __device__ virtual bool                 IntersectRay(Ray& ray, HitCtx& hit) const = 0;
            //__host__ __device__ virtual bool      InteresectPoint(const vec2& p, const float& thickness) const { return false; }
            //__host__ __device__ virtual bool      IntersectBBox(const BBox2f& bBox) const;

            //__host__ __device__ virtual vec2      PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }
         
            __device__ __forceinline__ bool         IsLight() const { return m_params.isLight; }
            __device__ __forceinline__ int          GetMaterialIdx() const { return m_params.materialIdx; }
            __device__ __forceinline__ const vec3&  GetRadiance() const { return m_params.radiance; }

            __device__ void                         Synchronise(const TracableParams& params) { m_params = params; }

        protected:
            __device__ Tracable() {}

            __device__ __forceinline__ RayBasic     RayToObjectSpace(const Ray& world) const { return m_params.transform.RayToObjectSpace(world.od); }
            __device__ __forceinline__ vec3         NormalToWorldSpace(const vec3& object) const { return m_params.transform.NormalToWorldSpace(object); }
            //__host__ __device__ __forceinline__ vec3 PointToWorldSpace(const vec3& world) const { return m_params.transform.PointToWorldSpace(world); }

            TracableParams m_params;

        private:
            BBox2f m_handleInnerBBox;
        };
    }

    namespace Host
    {        
        class Tracable : public Host::SceneObject
        {
            friend class LightSampler;

        public:
            __host__ virtual ~Tracable() {}
           
            __host__ void       SetMaterialIdx(const int idx) { m_params.materialIdx = idx; }

            __host__ Device::Tracable* GetDeviceInstance() { return cu_deviceInstance; }

            __host__ virtual void Synchronise(const uint syncFlags) override final
            {
                if (syncFlags & kSyncParams)
                {
                    SynchroniseObjects<Device::Tracable>(cu_deviceInstance, m_params);
                }              
                OnSynchroniseTracable(syncFlags);
            }

            __host__ virtual std::vector<GaussianPoint> GenerateGaussianPointCloud(const int numPoints, const float areaGain, MersenneTwister& rng) = 0;
            __host__ virtual float CalculateSurfaceArea() const  = 0;
            __host__ inline const BidirectionalTransform& GetTransform() const { return m_params.transform; }

        protected:
            __host__ Tracable(const InitCtx& initCtx, const BidirectionalTransform& transform, const int materialIdx);

            __host__ void               MakeLight(const vec3& radiance);
            __host__ void               SetDeviceInstance(Device::Tracable* deviceInstance) { cu_deviceInstance = deviceInstance; }            
            __host__ virtual void       OnSynchroniseTracable(const uint syncFlags) = 0;
        protected:
            Device::Tracable*               cu_deviceInstance;

            TracableParams                  m_params;
        };
    }
}