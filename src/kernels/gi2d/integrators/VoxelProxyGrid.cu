#include "VoxelProxyGrid.cuh"

namespace GI2D
{    
    __host__ __device__ VoxelProxyGridParams::VoxelProxyGridParams()
    {
    }

    __device__ Device::VoxelProxyGrid::VoxelProxyGrid() : 
        m_voxelTracer(m_scene)
    {

    }
    
    __device__ void Device::VoxelProxyGrid::OnSynchronise(const int)
    {
    }

    __device__ bool Device::VoxelProxyGrid::CreateRay(Ray2D& ray, RenderCtx& renderCtx) const
    {

    }

    __device__ void Device::VoxelProxyGrid::Accumulate(const vec4& L, const RenderCtx& ctx)
    {

    }

    __device__ void Device::VoxelProxyGrid::Evaluate(const vec3& posWorld) const
    {

    }
    
    __host__ Host::VoxelProxyGrid::VoxelProxyGrid(const std::string& id, AssetHandle<Host::SceneDescription>& scene,
                                                  const uint width, const uint height) : 
        Cuda::Host::RenderObject(id),
        m_scene(scene)
    {
        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 100;
        constexpr uint kAccumBufferSize = kGridWidth * kGridHeight;

        m_hostAccumBuffer = CreateChildAsset<Core::Host::Vector<vec3>>("accumBuffer", kAccumBufferSize, Core::kVectorHostAlloc);

        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.m_scene = m_scene->GetDeviceObjects();
    }

    __host__ void Host::VoxelProxyGrid::OnDestroyAsset()
    {
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::VoxelProxyGrid::Synchronise(const int syncType)
    {
        //Host::SceneObject<>::Synchronise(cu_deviceInstance, syncType);

        if (syncType == kSyncParams) { SynchroniseInheritedClass<VoxelProxyGridParams>(cu_deviceInstance, *this, kSyncParams);  }
        if (syncType == kSyncObjects) { SynchroniseInheritedClass<VoxelProxyGridObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects); }
    }

    __host__ void Host::VoxelProxyGrid::Rebuild(const uint dirtyFlags)
    {

    }
}