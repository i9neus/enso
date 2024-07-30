#include "Camera.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"
#include "core/Vector.cuh"
#include "core/DirtinessFlags.cuh"
#include "core/AccumulationBuffer.cuh"

namespace Enso
{
    __device__ Device::Camera::Camera() : m_frameIdx(1) {}

    __device__ void Device::Camera::Prepare(const bool resetIntegrator)
    {
        m_frameIdx = resetIntegrator ? 0 : (m_frameIdx + 1);
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

    __host__ Host::Camera::Camera(const Asset::InitCtx& initCtx, Device::Camera* hostInstance, const AssetHandle<const Host::SceneContainer>& scene) :
        SceneObject(initCtx, hostInstance, scene),
        m_scene(scene),
        m_dirtyFlags(0)
    {
        
        Listen({ kDirtyIntegrators });
    }

    __host__ void Host::Camera::OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller)
    {
        //SceneObject::OnDirty(flag, caller);

        SetDirty(kDirtyIntegrators);
    }

    __host__ void Host::Camera::SetDeviceInstance(Device::Camera* deviceInstance)
    {
        SceneObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::SceneObject>(deviceInstance));
        cu_deviceInstance = deviceInstance;
    }

    __host__ Host::Camera::~Camera() noexcept
    {
        m_accumBuffer.DestroyAsset();
    }

    __host__ void Host::Camera::Initialise(const int numProbes, const int numHarmonics, const size_t accumBufferSize)
    {
        Assert(cu_deviceInstance);        

        m_accumBuffer = AssetAllocator::CreateChildAsset<Host::AccumulationBuffer>(*this, "accumBuffer", numProbes, numHarmonics, accumBufferSize);
        m_params.accum = m_accumBuffer->GetParams();

        m_deviceObjects.accumBuffer = m_accumBuffer->GetDeviceInstance();
        m_deviceObjects.scene = m_scene->GetDeviceInstance();
    }

    __host__ void Host::Camera::Integrate()
    {
        if (IsDirty(kDirtyIntegrators))
        {
            // Something has signaled to clear the integrators
            m_accumBuffer->Clear();
        }
        else if (m_params.maxSamples > 0 && m_accumBuffer->GetTotalAccumulatedSamples() >= m_params.maxSamples)
        {
            // If we've already accumuated the specified number of samples, we're done
            return;
        }

        KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);
        
        ScopedDeviceStackResize(1024 * 10, [this]() -> void
            {
                KernelIntegrate << <m_params.accum.kernel.grids.accumSize, m_params.accum.kernel.blockSize >> > (cu_deviceInstance);
            });

        m_accumBuffer->Reduce();

        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::Camera::OnSynchroniseSceneObject(const uint syncFlags)
    {
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::Camera>(cu_deviceInstance, m_deviceObjects); }
        if (syncFlags & kSyncParams) { SynchroniseObjects<Device::Camera>(cu_deviceInstance, m_params); }

        OnSynchroniseCamera(syncFlags);
    }

    __host__ bool Host::Camera::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node cameraNode = node.AddChildObject("camera");
        cameraNode.AddValue("maxSamples", m_params.maxSamples);
        return true;
    }

    __host__ bool Host::Camera::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = false;
        const Json::Node cameraNode = node.GetChildObject("camera", flags);
        if (cameraNode)
        {
            isDirty |= cameraNode.GetValue("maxSamples", m_params.maxSamples, flags);
        }

        m_params.maxSamples = max(0, m_params.maxSamples);

        return isDirty;
    }
}