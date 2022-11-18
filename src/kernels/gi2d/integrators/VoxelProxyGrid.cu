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
    
    __device__ void Device::VoxelProxyGrid::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncObjects)
        {
            assert(m_scenePtr);
            m_scene = *m_scenePtr;
        }
    }

    __device__ bool Device::VoxelProxyGrid::CreateRay(Ray2D& ray, RenderCtx& renderCtx) const
    {
        const vec2 probePosNorm = vec2(float(0.5f + renderCtx.sampleIdx % m_grid.size.x), float(0.5f + renderCtx.sampleIdx / m_grid.size.x));
        
        // Transform from screen space to view space
        ray.o = m_cameraTransform.PointToWorldSpace(probePosNorm);

        // Randomly scatter
        const float theta = renderCtx.rng.Rand<0>() * kTwoPi;
        ray.d = vec2(cosf(theta), sinf(theta));

        return true;
    }

    __device__ void Device::VoxelProxyGrid::Accumulate(const vec4& L, const RenderCtx& renderCtx)
    {
        (*m_accumBuffer)[kKernelIdx] += L.xyz;
    }

    __device__ vec3 Device::VoxelProxyGrid::Evaluate(const vec2& posWorld) const
    {
        const ivec2 probeIdx = ivec2(m_cameraTransform.PointToObjectSpace(posWorld));  

        if (probeIdx.x < 0 || probeIdx.x >= m_grid.size.x || probeIdx.y < 0 || probeIdx.y >= m_grid.size.y) { return kOne * 0.2; }

        return (*m_accumBuffer)[probeIdx.y * m_grid.size.x + probeIdx.x] / float(max(1, m_frameIdx));
    }

    __device__ void Device::VoxelProxyGrid::Render()
    {
        if (kKernelIdx >= m_grid.numProbes) { return; }
      
        RenderCtx renderCtx(kKernelIdx, uint(m_frameIdx), 0, *this);
        m_voxelTracer.Integrate(renderCtx);
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::VoxelProxyGrid::Prepare(const uint dirtyFlags)
    {
        if (dirtyFlags)
        {
            // Save ourselves a deference here by caching the scene pointers
            assert(m_scenePtr);
            m_scene = *m_scenePtr;
            m_frameIdx = 0;
        }
        else
        {
            m_frameIdx++;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);
    
    __host__ Host::VoxelProxyGrid::VoxelProxyGrid(const std::string& id, AssetHandle<Host::SceneDescription>& scene,
                                                  const uint width, const uint height) : 
        Host::SceneObject(id, m_hostInstance),
        m_scene(scene)
    {
        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 100;

        m_grid.size = ivec2(kGridWidth, kGridHeight);
        m_grid.numProbes = Area(m_grid.size);  

        m_cameraTransform.Construct(vec2(-0.5f), 0.0f, float(kGridWidth));

        m_hostAccumBuffer = CreateChildAsset<Core::Host::Vector<vec3>>("accumBuffer", m_grid.numProbes, Core::kVectorHostAlloc);

        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.m_scenePtr = m_scene->GetDeviceInstance();

        cu_deviceInstance = InstantiateOnDevice<Device::VoxelProxyGrid>();

        Synchronise(kSyncParams | kSyncObjects);             
    }

    __host__ void Host::VoxelProxyGrid::OnDestroyAsset()
    {
        m_hostAccumBuffer.DestroyAsset();

        DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::VoxelProxyGrid::Synchronise(const int syncType)
    {
        if (syncType & kSyncObjects) { SynchroniseInheritedClass<VoxelProxyGridObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects); }
        if (syncType & kSyncParams) { SynchroniseInheritedClass<VoxelProxyGridParams>(cu_deviceInstance, *this, kSyncParams); }
    }

    __host__ bool Host::VoxelProxyGrid::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
    {
        m_dirtyFlags = parentFlags;
        Synchronise(kSyncParams);

        return true;
    }

    __host__ void Host::VoxelProxyGrid::Render()
    {
        KernelPrepare<< <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);
        
        if (m_dirtyFlags)
        {
            m_hostAccumBuffer->Wipe();
            m_dirtyFlags = 0;
        }               

        const int blockSize = 16;
        const int gridSize = (m_grid.numProbes + blockSize - 1) / blockSize;

        KernelRender << <gridSize, blockSize >> > (cu_deviceInstance);
    }
}