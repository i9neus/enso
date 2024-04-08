#pragma once

#include "../FwdDecl.cuh"
#include "core/AssetAllocator.cuh"
#include "PathTracer2D.cuh"
#include "AccumulationBuffer.cuh"

namespace Enso
{
    class Ray2D;
    class HitCtx2D;
    class RenderCtx;
    class UIViewCtx;

    struct CameraObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(accumBuffer);
            CudaAssert(scene);
        }

        Device::AccumulationBuffer* accumBuffer = nullptr;
        const Device::SceneDescription* scene = nullptr;
    };
    
    namespace Device
    {   
        class Camera
        {
        public:
            __device__ virtual void Prepare(const uint dirtyFlags);
            __device__ virtual void Integrate();

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const = 0;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
            __device__ void Synchronise(const CameraObjects& objects);
            __device__ void Synchronise(const AccumulationBufferParams& params) { m_params = params; }

        protected:
            __device__ Camera();

            CameraObjects                           m_objects;
            AccumulationBufferParams                m_params;

        private:
            Device::PathTracer2D                    m_voxelTracer;
            int                                     m_frameIdx;
        };
    }

    namespace Host
    {
        class Camera
        {
        public:
            __host__ virtual void Integrate();
            //__host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual Device::Camera* GetDeviceInstance() const = 0;
            __host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx);

        protected:
            __host__ Camera(const std::string& id, const AssetHandle<const Host::SceneDescription>& scene, const AssetAllocator& allocator);
            __host__ virtual ~Camera();

            __host__ void Initialise(const int numProbes, const int numHarmonics, const size_t accumBufferSize, Device::Camera* deviceInstance);
            __host__ void OnDestroyAsset();
            __host__ void Synchronise(const int syncFlags);

        protected:            
            AssetHandle<Host::AccumulationBuffer>                   m_accumBuffer;
            AssetHandle<const Host::SceneDescription>               m_scene;
            AccumulationBufferParams                                m_params;

            CameraObjects                                           m_deviceObjects;
            Device::Camera*                                         cu_deviceInstance;
        
        private:
            int                                                     m_dirtyFlags;
            const AssetAllocator&                                   m_parentAllocator;
        };
    };
}      