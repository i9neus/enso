#include "PathTracerLayer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

#include "../RenderCtx.cuh"

using namespace Cuda;

namespace GI2D
{        
    __host__ __device__ PathTracerLayerParams::PathTracerLayerParams()
    {
        m_accum.downsample = 1;
        m_frameIdx = 0;
    }

    __device__ Device::PathTracerLayer::PathTracerLayer() : 
        m_overlayTracer(m_scene)
    {  

    }

    __device__ void Device::PathTracerLayer::OnSynchronise(const int syncFlags)
    {
    }

    __device__ void Device::PathTracerLayer::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        m_accumBuffer->At(kKernelPos<ivec2>()) += L;
    }

    __device__ bool Device::PathTracerLayer::CreateRay(Ray2D& ray, RenderCtx& renderCtx) const
    {
        // Transform from screen space to view space
        ray.o = m_viewCtx.transform.matrix * vec2(kKernelPos<ivec2>() * m_accum.downsample);
        if (!m_viewCtx.sceneBounds.Contains(ray.o)) { return false; }

        // Randomly scatter
        const float theta = renderCtx.rng.Rand<0>() * kTwoPi;
        ray.d = vec2(cosf(theta), sinf(theta));

        return true;
    }

    __device__ void Device::PathTracerLayer::Render()
    {        
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_accumBuffer->Height()) { return; }

        RenderCtx renderCtx(kKernelY * kKernelWidth + kKernelX, uint(m_frameIdx), 0, *this);

        m_overlayTracer.Integrate(renderCtx);       

    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::PathTracerLayer::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {        
        assert(deviceOutputImage);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= deviceOutputImage->Width() || xyScreen.y < 0 || xyScreen.y >= deviceOutputImage->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_viewCtx.transform.matrix * vec2(xyScreen);

        if (!m_viewCtx.sceneBounds.Contains(xyView)) 
        { 
            deviceOutputImage->At(xyScreen) = vec4(0.1f, 0.1f, 0.1f, 1.0f);
            return; 
        }

        const vec2 uv = vec2(xyScreen) * vec2(m_accumBuffer->Dimensions()) / vec2(deviceOutputImage->Dimensions());
        vec4 L = m_accumBuffer->Lerp(uv);

        deviceOutputImage->At(xyScreen) = vec4(L.xyz / fmaxf(L.w, 1.0f), 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::PathTracerLayer::PathTracerLayer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables,
                                           const uint width, const uint height, const uint downsample, cudaStream_t renderStream) :
        UILayer(id, bih, tracables)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Cuda::Host::ImageRGBW>("id_2dgiAccumBuffer", width / downsample, height / downsample, renderStream);

        m_deviceObjects.m_scene.bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.m_scene.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        m_accum.downsample = downsample;

        cu_deviceData = InstantiateOnDevice<Device::PathTracerLayer>();

        Synchronise(kSyncObjects);
    }

    Host::PathTracerLayer::~PathTracerLayer()
    {
        OnDestroyAsset();
    }


    __host__ void Host::PathTracerLayer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        UILayer::Rebuild(dirtyFlags, viewCtx, selectionCtx);

        Synchronise(kSyncParams);
    }

    __host__ void Host::PathTracerLayer::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceData, syncType);

        if (syncType & kSyncObjects) { SynchroniseInheritedClass<PathTracerLayerObjects>(cu_deviceData, m_deviceObjects, kSyncObjects); }
        if (syncType & kSyncParams) { SynchroniseInheritedClass<PathTracerLayerParams>(cu_deviceData, *this, kSyncParams); }
    }

    __host__ void Host::PathTracerLayer::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::PathTracerLayer::Render()
    {
        if (m_dirtyFlags)
        {
            m_hostAccumBuffer->Clear(vec4(0.0f));
            m_dirtyFlags = 0;
        }

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0 >> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::PathTracerLayer::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}