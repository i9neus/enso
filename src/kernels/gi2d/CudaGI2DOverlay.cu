#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "CudaGI2DOverlay.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"
#include "Tracable.cuh"
#include "kernels/CudaAssetContainer.cuh"

using namespace Cuda;

namespace GI2D
{    
    __host__ __device__ OverlayParams::OverlayParams()
    {
        gridCtx.show = true;
        gridCtx.lineAlpha = 0.0;
        gridCtx.majorLineSpacing = 1.0f;
        gridCtx.majorLineSpacing = 1.0f;

        selectionCtx.isLassoing = false;
    }

    __device__ Device::Overlay::Overlay(const OverlayParams& params, const Objects& objects) :
        m_params(params),
        m_objects(objects)
    {
    }

    __device__ void Device::Overlay::Synchronise(const OverlayParams& params)
    {
        m_params = params;
    }

    __device__ void Device::Overlay::Synchronise(const Objects& objects)
    {
        m_objects = objects;

        assert(objects.bih->GetNumPrimitives() == objects.tracables->Size());
    }

    __device__ void Device::Overlay::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {
        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x >= 0 && xyScreen.x < m_objects.accumBuffer->Width() && 
            xyScreen.y >= 0 && xyScreen.y < m_objects.accumBuffer->Height())
        {            
            deviceOutputImage->Blend(xyScreen, m_objects.accumBuffer->At(xyScreen));
            //vec4& target = deviceOutputImage->At(xyScreen);
            //target = Blend(target, m_objects.accumBuffer->At(xyScreen));
            //target.xyz += m_objects.accumBuffer->At(xyScreen).xyz;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    __device__ void Device::Overlay::Render()
    {
        assert(m_objects.accumBuffer);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_objects.accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_objects.accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_params.viewCtx.transform.matrix * vec2(xyScreen);

        //m_objects.accumBuffer->At(xyScreen) = vec4(xyView, 0.0f, 1.0f);
        //return;

        vec4 L(0.0f, 0.0f, 0.0f, 0.0f);

        if (!m_params.viewCtx.sceneBounds.Contains(xyView)) 
        { 
            L = vec4(0.0f);
        }
        else if (m_params.gridCtx.show)
        {
            // Draw the grid
            vec2 xyGrid = fract(xyView / vec2(m_params.gridCtx.majorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_params.viewCtx.dPdXY / m_params.gridCtx.majorLineSpacing * mix(1.0f, 3.0f, m_params.gridCtx.lineAlpha)) 
            { 
                L = Blend(L, kOne, 0.5 * (1 - m_params.gridCtx.lineAlpha));
            }
            xyGrid = fract(xyView / vec2(m_params.gridCtx.minorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_params.viewCtx.dPdXY / m_params.gridCtx.minorLineSpacing * 1.5f)
            { 
                L = Blend(L, kOne, 0.5 * m_params.gridCtx.lineAlpha);
            }
        }  

        if (m_objects.bih && m_objects.tracables)
        {            
            const Cuda::Device::Vector<TracableInterface*>& tracables = *(m_objects.tracables);
      
            auto onPointIntersectLeaf = [&, this](const uint* idxRange) -> void
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    assert(idx < tracables.Size());
                    assert(tracables[idx]);

                    const auto& tracable = *tracables[idx];
                    L = Blend(L, tracable.EvaluateOverlay(xyView, m_params.viewCtx));

                    if (tracable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, m_params.viewCtx.dPdXY)) L = vec4(kRed, 1.0f);
                }
            };          
            m_objects.bih->TestPoint(xyView, onPointIntersectLeaf);
        }

        // Draw the lasso 
        if (m_params.selectionCtx.isLassoing && m_params.selectionCtx.lassoBBox.PointOnPerimiter(xyView, m_params.viewCtx.dPdXY * 2.f)) { L = vec4(kRed, 1.0f); }

        // Draw the selected object's bounding box
        if (m_params.selectionCtx.numSelected > 0. && m_params.selectionCtx.selectedBBox.PointOnPerimiter(xyView, m_params.viewCtx.dPdXY * 2.f)) { L = vec4(kGreen, 1.0f); }

        m_objects.accumBuffer->At(xyScreen) = L;
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    Host::Overlay::Overlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Cuda::Host::AssetVector<Host::Tracable>>& tracables,
                                   const uint width, const uint height, cudaStream_t renderStream) :
        UILayer(id, bih, tracables)
    {        
        // Create some Cuda objects
        m_hostAccumBuffer = CreateAsset<Cuda::Host::ImageRGBW>("id_2dgiOverlayBuffer", width, height, renderStream);

        m_objects.bih = m_hostBIH->GetDeviceInstance();
        m_objects.tracables = m_hostTracables->GetDeviceInstance();
        m_objects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        cu_deviceData = InstantiateOnDevice<Device::Overlay>(m_params, m_objects); 
    }

    Host::Overlay::~Overlay()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Overlay::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::Overlay::Render()
    {
        if (!m_dirtyFlags) { return; }
        
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());

        m_dirtyFlags = 0;
    }

    __host__ void Host::Overlay::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const
    {        
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::Overlay::Synchronise()
    {
        if (!m_dirtyFlags) { return; }

        m_params.viewCtx = m_viewCtx;
        m_params.selectionCtx = m_selectionCtx;
         
        // Calculate some values for the guide grid
        const float logScale = std::log10(m_params.viewCtx.transform.scale);
        constexpr float kGridScale = 0.05f;
        m_params.gridCtx.majorLineSpacing = kGridScale * std::pow(10.0f, std::ceil(logScale));
        m_params.gridCtx.minorLineSpacing = kGridScale * std::pow(10.0f, std::floor(logScale));
        m_params.gridCtx.lineAlpha = 1 - (logScale - std::floor(logScale));
        m_params.gridCtx.show = true;
        m_params.selectionCtx.lassoBBox.Rectify();
        m_params.selectionCtx.selectedBBox.Rectify();

        // Upload to the device
        SynchroniseObjects(cu_deviceData, m_params);
    }
}