#pragma once

#include "PathTracer2D.cuh"
#include "../SceneObject.cuh"

using namespace Cuda;

namespace GI2D
{     
    struct VoxelProxyGridParams
    {
        __host__ __device__ VoxelProxyGridParams();

        ViewTransform2D             m_cameraTransform;

        uint						subprobesPerProbe;	//			<-- A sub-probe is a set of SH coefficients + data. Multiple sub-probes are accumulated to make a full probe. 
        uint						bucketsPerProbe; //				<-- The total number of accumulation units (coefficients + data) per probe
        uint						totalBuckets; //				<-- The total number of accumulation units in the grid
        uint						totalSubprobes; //				<-- The total number of subprobes in the grid
    };

    struct VoxelProxyGridObjects
    {
        Device::SceneDescription                        m_scene;
        Core::Device::Vector<vec3>*                     m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class VoxelProxyGrid : public Cuda::Device::RenderObject,
                               public VoxelProxyGridParams,
                               public VoxelProxyGridObjects,
                               public Camera2D
        {
        public:
            __device__ VoxelProxyGrid();

            __device__ void Evaluate(const vec3& posWorld) const;

            __device__ virtual bool CreateRay(Ray2D& ray, RenderCtx& renderCtx) const override final;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __device__ void OnSynchronise(const int);

        private:
            PathTracer2D                            m_voxelTracer;
        };
    }

    namespace Host
    {
        class VoxelProxyGrid : public Cuda::Host::RenderObject,
                               public VoxelProxyGridParams
        {
        public:
            __host__ VoxelProxyGrid(const std::string& id, AssetHandle<Host::SceneDescription>& scene,  const uint width, const uint height);
            __host__ virtual ~VoxelProxyGrid() {}
            
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags);
            __host__ void Synchronise(const int syncType);

            __host__ Device::VoxelProxyGrid* GetDeviceInstance() { return cu_deviceInstance; }

        private:
            GI2D::Device::VoxelProxyGrid*          cu_deviceInstance = nullptr;
            VoxelProxyGridObjects                  m_deviceObjects;

            AssetHandle<Host::SceneDescription>    m_scene;

            AssetHandle<Core::Host::Vector<vec3>>  m_hostAccumBuffer;
        };
    }
}