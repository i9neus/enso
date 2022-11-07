#include "VoxelProxyGrid.cuh"

namespace GI2D
{    
    __host__ __device__ VoxelProxyGridParams::VoxelProxyGridParams()
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
    
    __host__ Host::VoxelProxyGrid::VoxelProxyGrid(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables,
                                                  const uint width, const uint height) : 
        Host::SceneObject<>::SceneObject(id)
    {
        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 100;
        constexpr uint kAccumBufferSize = kGridWidth * kGridHeight;

        m_hostAccumBuffer = CreateChildAsset<Core::Host::Vector<vec3>>("accumBuffer", kAccumBufferSize, Core::kVectorHostAlloc);
    }

    __host__ void Host::VoxelProxyGrid::OnDestroyAsset()
    {
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::VoxelProxyGrid::Synchronise(const int syncType)
    {
        Host::SceneObject<>::Synchronise(cu_deviceInstance, syncType);

        if (syncType == kSyncParams) { SynchroniseInheritedClass<VoxelProxyGridParams>(cu_deviceInstance, *this, kSyncParams);  }
        if (syncType == kSyncObjects) { SynchroniseInheritedClass<VoxelProxyGridObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects); }
    }
}