#pragma once

#include "PathTracer2D.cuh"
#include "../SceneObject.cuh"
#include "core/math/Math.cuh"

namespace Enso
{     
    struct VoxelProxyGridParams
    {
        __host__ __device__ VoxelProxyGridParams();

        BidirectionalTransform2D        m_cameraTransform;

        struct
        {
            ivec2                       size;
            uint                        numProbes;

            uint						subprobesPerProbe;	//			<-- A sub-probe is a set of SH coefficients + data. Multiple sub-probes are accumulated to make a full probe. 
            uint						bucketsPerProbe; //				<-- The total number of accumulation units (coefficients + data) per probe
            uint						totalBuckets; //				<-- The total number of accumulation units in the grid
            uint						totalSubprobes; //				<-- The total number of subprobes in the grid
        }
        m_grid;
    };

    struct VoxelProxyGridObjects
    {
        const Device::SceneDescription*         m_scenePtr = nullptr;
        Device::Vector<vec3>*                   m_accumBuffer = nullptr;
    };

    namespace Host { class VoxelProxyGrid; }

    namespace Device
    {
        class VoxelProxyGrid : public Device::SceneObject,
                               public ICamera2D
        {
        public:
            __device__ VoxelProxyGrid();

            __device__ vec3 Evaluate(const vec2& posWorld) const;

            __device__ void Prepare(const uint dirtyFlags);
            __device__ void Render(void* debugData = nullptr);

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __host__ __device__ void OnSynchronise(const int);

        private:
            PathTracer2D                            m_voxelTracer;
            int                                     m_frameIdx;
            
            Device::SceneDescription                m_scene;
        };
    }

    namespace Host
    {
        class VoxelProxyGrid : public Host::SceneObject
        {
        public:
            __host__ VoxelProxyGrid(const std::string& id, AssetHandle<Host::SceneDescription>& scene, const uint width, const uint height);
            __host__ virtual ~VoxelProxyGrid() {}
            
            __host__ void OnDestroyAsset();

            //__host__ void OnPreRender();
            __host__ void Render();
            __host__ void Synchronise(const int syncType);

            __host__ virtual bool       Finalise() { return true; }
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx);

            __host__ Device::VoxelProxyGrid* GetDeviceInstance() { return cu_deviceInstance; }

        private:
            Device::VoxelProxyGrid*                cu_deviceInstance = nullptr;
            VoxelProxyGridObjects                  m_deviceObjects;

            AssetHandle<Host::SceneDescription>    m_scene;

            AssetHandle<Host::Vector<vec3>>        m_hostAccumBuffer;

            Device::VoxelProxyGrid                 m_hostInstance;
        };
    }
}