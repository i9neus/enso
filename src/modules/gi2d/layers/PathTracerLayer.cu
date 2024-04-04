#include "PathTracerLayer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"

#include "../RenderCtx.cuh"
#include "../SceneDescription.cuh"
#include "../integrators/VoxelProxyGrid.cuh"

namespace Enso
{        
    __host__ __device__ PathTracerLayerParams::PathTracerLayerParams()
    {
        accum.downsample = 1;
    }

    __device__ void Device::PathTracerLayer::OnSynchronise(const int syncFlags)
    {
        m_frameIdx = 0;      
    }

    __device__ void Device::PathTracerLayer::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        m_objects.accumBuffer->At(kKernelPos<ivec2>()) += L;
    }

    __device__ bool Device::PathTracerLayer::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        // Transform from screen space to view space
        ray.o = UILayer::m_params.viewCtx.transform.matrix * vec2(kKernelPos<ivec2>() * m_params.accum.downsample);
        if (!UILayer::m_params.viewCtx.sceneBounds.Contains(ray.o)) { return false; }

        // Randomly scatter
        const float theta = renderCtx.rng.Rand<0>() * kTwoPi;
        ray.d = vec2(cosf(theta), sinf(theta));

        return true;
    }

    __device__ void Device::PathTracerLayer::Render()
    {        
        return;
        
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_objects.accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_objects.accumBuffer->Height()) { return; }

        RenderCtx renderCtx(kKernelY * kKernelWidth + kKernelX, uint(m_frameIdx), 0, *this);

        m_overlayTracer.Integrate(renderCtx);       

    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::PathTracerLayer::Prepare(const uint dirtyFlags)
    {
        m_frameIdx++;

        // Save ourselves a deference here by caching the scene pointers
        assert(m_objects.scenePtr);
        m_scene = *m_objects.scenePtr;
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);

    __device__ void Device::PathTracerLayer::Composite(Device::ImageRGBA* deviceOutputImage)
    {        
        assert(deviceOutputImage);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= deviceOutputImage->Width() || xyScreen.y < 0 || xyScreen.y >= deviceOutputImage->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = UILayer::m_params.viewCtx.transform.matrix * vec2(xyScreen);

        if (!UILayer::m_params.viewCtx.sceneBounds.Contains(xyView))
        { 
            deviceOutputImage->At(xyScreen) = vec4(0.1f, 0.1f, 0.1f, 1.0f);
            return; 
        }

        vec4 L(0.0f);

        const vec2 uv = vec2(xyScreen) * vec2(m_objects.accumBuffer->Dimensions()) / vec2(deviceOutputImage->Dimensions());
        L = m_objects.accumBuffer->Lerp(uv);
        L.xyz /= fmaxf(L.w, 1.0f);

        //L.xyz += m_scene.voxelProxy->Evaluate(xyView);

        deviceOutputImage->At(xyScreen) = vec4(L.xyz, 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::PathTracerLayer::PathTracerLayer(const std::string& id, const AssetHandle<Host::SceneDescription>& scene, const uint width, const uint height, const uint downsample, cudaStream_t renderStream) :
        UILayer(id, scene)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Host::ImageRGBW>("id_2dgiAccumBuffer", width / downsample, height / downsample, renderStream);

        m_deviceObjects.scenePtr = scene->GetDeviceInstance(); 
        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        m_params.accum.downsample = downsample;

        cu_deviceData = InstantiateOnDevice<Device::PathTracerLayer>();

        Synchronise(kSyncObjects);
    }

    Host::PathTracerLayer::~PathTracerLayer()
    {
        OnDestroyAsset();
    }

    __host__ void Host::PathTracerLayer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        m_dirtyFlags = dirtyFlags;
        
        UILayer::Rebuild(dirtyFlags, viewCtx, selectionCtx);

        Synchronise(kSyncParams);
    }

    __host__ void Host::PathTracerLayer::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceData, syncType);

        if (syncType & kSyncObjects) { SynchroniseObjects<Device::PathTracerLayer>(cu_deviceData, m_deviceObjects); }
        if (syncType & kSyncParams) { SynchroniseObjects<Device::PathTracerLayer>(cu_deviceData, m_params); }
    }

    __host__ void Host::PathTracerLayer::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::PathTracerLayer::Render()
    {
        // Advance the frame counter
        KernelPrepare << <1, 1 >> > (cu_deviceData, m_dirtyFlags);
        
        if (m_dirtyFlags & (kDirtyMaterials | kDirtyObjectBounds | kDirtyObjectBVH))
        {
            m_hostAccumBuffer->Clear(vec4(0.0f));
        }        
        m_dirtyFlags = 0;

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        // Render the frame
        KernelRender << < gridSize, blockSize, 0 >> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::PathTracerLayer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}