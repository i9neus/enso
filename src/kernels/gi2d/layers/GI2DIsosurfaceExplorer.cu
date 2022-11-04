#include "GI2DIsosurfaceExplorer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "kernels/math/Complex.cuh"
#include "generic/Hash.h"

#include "../RenderCtx.cuh"

using namespace Cuda;

namespace GI2D
{
    __host__ __device__ IsosurfaceExplorerParams::IsosurfaceExplorerParams()
    {
        m_accum.width = 0;
        m_accum.height = 0;
        m_accum.downsample = 1;
        m_frameIdx = 0;
    }

    __host__ __device__ void IsosurfaceExplorerInterface::EvaluateField(const vec2& xyView, float& F, vec2& gradF)
    {
        if (!m_inspectors || m_inspectors->IsEmpty()) 
        { 
            F = 0.0f;
            gradF = 0.0f;
            return; 
        }

        constexpr float c = 0.05f;
        constexpr float kMetaThreshold = 8.0f;
        
        float denom = 0.0f;
        for (int isoIdx = 0; isoIdx < m_inspectors->Size(); ++isoIdx)
        {          
            const vec2 pt = (*m_inspectors)[isoIdx]->GetWorldSpaceBoundingBox().Centroid();
            
            F += c / length2(xyView - pt);

            float n = sqr(xyView.x - pt.x) + sqr(xyView.y - pt.y);
            gradF += -c * 2.0f * (xyView - pt) / sqr(n);

            denom += c / n;
        }

        F = 1 / sqrtf(F) - 1 / sqrtf(kMetaThreshold);

        // Vector derivative of F
        gradF = -0.5f * gradF / powf(denom, 1.5);

        // Vector derivative of |F|
        //gradF = (0.5f * gradF) * (-1.0 / sqrt(kMetaThreshold) + 1.0 / sqrt(denom)) / (pow(denom, 1.5) * (abs(-1.0 / sqrt(kMetaThreshold)) + 1.0 / sqrt(denom)));
    }

    __device__ Device::IsosurfaceExplorer::IsosurfaceExplorer() { }

    __device__ vec2 Mul(const vec2& a, const vec2& b)
    {
        return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    __device__ void Device::IsosurfaceExplorer::Render()
    {
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_viewCtx.transform.matrix * vec2(xyScreen * m_accum.downsample);

        if (!m_viewCtx.sceneBounds.Contains(xyView)) { m_accumBuffer->At(xyScreen) = vec4(0.0f, 0.0f, 0.0f, 1.0f); return; }

        float F = 0.0f;
        vec2 gradF(0.0f);
        EvaluateField(xyView, F, gradF);

        vec3 rgb;
        rgb = (F > 0.0f && F < 0.01f) ? kOne : vec3(normalize(gradF) * 0.5f + 0.5f, 0.0f);
        
        m_accumBuffer->At(xyScreen) = vec4(rgb, 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::IsosurfaceExplorer::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {
        assert(deviceOutputImage);

        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x >= 0 && xyScreen.x < m_accumBuffer->Width() && xyScreen.y >= 0 && xyScreen.y < m_accumBuffer->Height())
        {
            deviceOutputImage->Blend(xyScreen, m_accumBuffer->At(xyScreen));
            //vec4& target = deviceOutputImage->At(xyScreen);
            //target = Blend(target, m_accumBuffer->At(xyScreen));
            //target.xyz += m_accumBuffer->At(xyScreen).xyz;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::IsosurfaceExplorer::IsosurfaceExplorer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables, 
                                                 AssetHandle<InspectorContainer>& inspectors, const uint width, const uint height, const uint downsample, cudaStream_t renderStream) :
        UILayer(id, bih, tracables),
        m_hostInspectors(inspectors)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Cuda::Host::ImageRGBW>("accumBuffer", width / downsample, height / downsample, renderStream);

        m_deviceObjects.m_inspectors = m_hostInspectors->GetDeviceInstance();
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        m_accum.width = width;
        m_accum.height = height;
        m_accum.downsample = downsample;

        cu_deviceData = InstantiateOnDevice<Device::IsosurfaceExplorer>();

        Synchronise(kSyncObjects);
    }

    Host::IsosurfaceExplorer::~IsosurfaceExplorer()
    {
        OnDestroyAsset();
    }


    __host__ void Host::IsosurfaceExplorer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        UILayer::Rebuild(dirtyFlags, viewCtx, selectionCtx);

        Synchronise(kSyncParams);
    }

    __host__ void Host::IsosurfaceExplorer::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceData, syncType);

        if (syncType & kSyncObjects) { SynchroniseInheritedClass<IsosurfaceExplorerObjects>(cu_deviceData, m_deviceObjects, kSyncObjects); }
        if (syncType & kSyncParams) { SynchroniseInheritedClass<IsosurfaceExplorerParams>(cu_deviceData, *this, kSyncParams); }
    }

    __host__ void Host::IsosurfaceExplorer::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::IsosurfaceExplorer::Render()
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

    __host__ void Host::IsosurfaceExplorer::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}