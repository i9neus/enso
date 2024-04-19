#pragma once

#include "../FwdDecl.cuh"
#include "core/AssetAllocator.cuh"
#include "PathTracer2D.cuh"
#include "AccumulationBuffer.cuh"
#include "../SceneObject.cuh"

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
        const Device::SceneContainer* scene = nullptr;
    };

    struct CameraParams
    {
        __device__ void Validate() const
        {
            accum.Validate();
            CudaAssert(maxSamples >= 0);            
        }
        
        int maxSamples = 100;

        AccumulationBufferParams accum;
    };
    
    namespace Device
    {   
        class Camera : public Device::SceneObject
        {
        public:
            __device__ virtual void Prepare(const uint dirtyFlags);
            __device__ virtual void Integrate();

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const = 0;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
            __device__ void Synchronise(const CameraObjects& objects);
            __device__ void Synchronise(const CameraParams& params) { m_params = params; }

        protected:
            __device__ Camera();

            CameraObjects                           m_objects;
            CameraParams                            m_params;

        private:
            Device::PathTracer2D                    m_voxelTracer;
            int                                     m_frameIdx;
        };
    }

    namespace Host
    {
        class Camera : public Host::SceneObject
        {
        public:
            __host__ virtual void Integrate();
            //__host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual Device::Camera* GetDeviceInstance() const override { return cu_deviceInstance; }
            __host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx);

        protected:
            __host__ Camera(const Asset::InitCtx& initCtx, Device::Camera& hostInstance, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ virtual ~Camera() noexcept;
            __host__ void SetDeviceInstance(Device::Camera* deviceInstance);

            __host__ void Initialise(const int numProbes, const int numHarmonics, const size_t accumBufferSize);
            __host__ void Synchronise(const int syncFlags);

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:            
            AssetHandle<Host::AccumulationBuffer>                   m_accumBuffer;
            AssetHandle<const Host::SceneContainer>               m_scene;
            CameraParams                                            m_params;

            CameraObjects                                           m_deviceObjects;
        
        private:
            int                                                     m_dirtyFlags;
            
            Device::Camera*                                         cu_deviceInstance = nullptr;
        };
    };
}      