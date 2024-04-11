#include "Camera.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"
#include "core/Vector.cuh"
#include "../Dirtiness.cuh"
#include "AccumulationBuffer.cuh"

namespace Enso
{
    __device__ Device::Camera::Camera() : m_frameIdx(1) {}

    __device__ void Device::Camera::Prepare(const uint dirtyFlags)
    {
        m_frameIdx = (dirtyFlags & kDirtyIntegrators) ? 0 : (m_frameIdx + 1);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);

    __device__ void Device::Camera::Synchronise(const CameraObjects& objects) 
    { 
        m_objects = objects; 
        m_voxelTracer.Synchronise(*m_objects.scene);
    }

    __device__ void Device::Camera::Integrate()
    {        
        if (kKernelIdx >= m_params.accum.totalSubprobes) { return; }
        CudaAssertDebug(m_objects.accumBuffer)

        //const int probeIdx = kKernelIdx / m_params.accum.unitsPerProbe;
        //const int subprobeIdx = ((kKernelIdx / m_params.accum.numHarmonics) % m_params.accum.subprobesPerProbe);

        /*const uchar ctxFlags = (probeIdx / m_params.accum.gridSize.x == m_params.accum.gridSize.y / 2 &&
            probeIdx % m_params.accum.gridSize.x == m_params.accum.gridSize.x / 2 &&
            subprobeIdx == 0) ? kRenderCtxDebug : 0;*/
        
        RenderCtx renderCtx(kKernelIdx, uint(m_frameIdx), 0, *this, 0);

        /*if (ctxFlags & kRenderCtxDebug)
        {
            renderCtx.debugData = &m_kifsDebug;
        }*/

        m_voxelTracer.Integrate(renderCtx);
    }
    DEFINE_KERNEL_PASSTHROUGH(Integrate);

    __host__ Host::Camera::Camera(const std::string& id, const AssetHandle<const Host::SceneDescription>& scene, const AssetAllocator& allocator) :
        m_scene(scene),
        m_parentAllocator(allocator),
        m_dirtyFlags(0), 
        cu_deviceInstance(nullptr)
    {

    }

    __host__ Host::Camera::~Camera()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Camera::OnDestroyAsset()
    {
        m_accumBuffer.DestroyAsset();
    }

    __host__ void Host::Camera::Initialise(const int numProbes, const int numHarmonics, const size_t accumBufferSize, Device::Camera* deviceInstance)
    {
        AssertMsg(!cu_deviceInstance, "Already initialised");        
        cu_deviceInstance = deviceInstance;

        m_accumBuffer = m_parentAllocator.CreateChildAsset<Host::AccumulationBuffer>("accumBuffer", numProbes, numHarmonics, accumBufferSize);
        m_params.accum = m_accumBuffer->GetParams();

        m_deviceObjects.accumBuffer = m_accumBuffer->GetDeviceInstance();
        m_deviceObjects.scene = m_scene->GetDeviceInstance();

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ bool Host::Camera::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx)
    {
        m_dirtyFlags = dirtyFlags;        
        return true;
    }

    __host__ void Host::Camera::Integrate()
    {
        if (m_dirtyFlags & kDirtyIntegrators)
        {
            m_accumBuffer->Clear();
        }
        m_dirtyFlags = 0;

        // If we've already accumuated the specified number of samples, we're done
        if (m_params.maxSamples > 0 && m_accumBuffer->GetTotalAccumulatedSamples() >= m_params.maxSamples) { return; }

        KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        ScopedDeviceStackResize(1024 * 10, [this]() -> void
            {
                KernelIntegrate << <m_params.accum.kernel.grids.accumSize, m_params.accum.kernel.blockSize >> > (cu_deviceInstance);
            });

        m_accumBuffer->Reduce();

        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::Camera::Synchronise(const int syncFlags)
    {
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::Camera>(cu_deviceInstance, m_deviceObjects); }
        if (syncFlags & kSyncParams) { SynchroniseObjects<Device::Camera>(cu_deviceInstance, m_params); }
    }

    __host__ bool Host::Camera::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node cameraNode = node.AddChildObject("camera");
        cameraNode.AddValue("maxSamples", m_params.maxSamples);
        return true;
    }

    __host__ uint Host::Camera::Deserialise(const Json::Node& node, const int flags)
    {
        uint dirtyFlags = 0u;

        const Json::Node cameraNode = node.GetChildObject("camera", flags);
        if (cameraNode)
        {
            if (cameraNode.GetValue("maxSamples", m_params.maxSamples, flags)) { dirtyFlags |= kDirtyIntegrators; }
        }

        m_params.maxSamples = max(0, m_params.maxSamples);

        return dirtyFlags;
    }
}