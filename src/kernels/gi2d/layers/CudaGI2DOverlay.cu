#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "CudaGI2DOverlay.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"
#include "kernels/CudaAssetContainer.cuh"
#include "../tracables/primitives/LineSegment.cuh"

using namespace Cuda;

namespace GI2D
{    
    __host__ __device__ OverlayParams::OverlayParams()
    {
        m_gridCtx.show = true;
        m_gridCtx.lineAlpha = 0.0;
        m_gridCtx.majorLineSpacing = 1.0f;
        m_gridCtx.majorLineSpacing = 1.0f;
    }

    __device__ Device::Overlay::Overlay()
    {
    }

    __device__ void Device::Overlay::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {
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

    __device__ void Device::Overlay::Render()
    {
        assert(m_accumBuffer);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_viewCtx.transform.matrix * vec2(xyScreen);

        //m_accumBuffer->At(xyScreen) = vec4(xyView, 0.0f, 1.0f);
        //return;

        vec4 L(0.0f, 0.0f, 0.0f, 0.0f);

        if (!m_viewCtx.sceneBounds.Contains(xyView)) 
        { 
            L = vec4(0.0f);
        }
        else if (m_gridCtx.show)
        {
            // Draw the grid
            vec2 xyGrid = fract(xyView / vec2(m_gridCtx.majorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_viewCtx.dPdXY / m_gridCtx.majorLineSpacing * mix(1.0f, 3.0f, m_gridCtx.lineAlpha)) 
            { 
                L = Blend(L, kOne, 0.5 * (1 - m_gridCtx.lineAlpha));
            }
            xyGrid = fract(xyView / vec2(m_gridCtx.minorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_viewCtx.dPdXY / m_gridCtx.minorLineSpacing * 1.5f)
            { 
                L = Blend(L, kOne, 0.5 * m_gridCtx.lineAlpha);
            }
        }  

        // Draw the tracables
        if (m_bih && m_tracables)
        {            
            const VectorInterface<TracableInterface*>& tracables = *(m_tracables);
      
            auto onPointIntersectLeaf = [&, this](const uint* idxRange) -> void
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    assert(idx < tracables.Size());
                    assert(tracables[idx]);

                    const auto& tracable = *tracables[idx];
                    vec4 LTracable;
                    if (tracable.EvaluateOverlay(xyView, m_viewCtx, LTracable))
                    {
                        L = Blend(L, LTracable);
                    }

                    if (tracable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, m_viewCtx.dPdXY)) L = vec4(kRed, 1.0f);
                }
            };          
            m_bih->TestPoint(xyView, onPointIntersectLeaf);
        }

        // Draw the widgets
        if (m_widgets)
        {
            for (int idx = 0; idx < m_widgets->Size(); ++idx)
            {
                vec4 LWidget;
                if ((*m_widgets)[idx]->EvaluateOverlay(xyView, m_viewCtx, LWidget))
                {
                    L = Blend(L, LWidget);
                }
            }
        }

        // Draw the lasso 
        if (m_selectionCtx.isLassoing && m_selectionCtx.lassoBBox.PointOnPerimiter(xyView, m_viewCtx.dPdXY * 2.f)) { L = vec4(kRed, 1.0f); }

        // Draw the selected object's bounding box
        if (m_selectionCtx.numSelected > 0. && m_selectionCtx.selectedBBox.PointOnPerimiter(xyView, m_viewCtx.dPdXY * 2.f)) { L = vec4(kGreen, 1.0f); }

        m_accumBuffer->At(xyScreen) = L;
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    Host::Overlay::Overlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables, AssetHandle<WidgetContainer>& widgets,
                                   const uint width, const uint height, cudaStream_t renderStream) :
        UILayer(id, bih, tracables),
        m_hostWidgets(widgets)
    {        
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Cuda::Host::ImageRGBW>("accumBuffer", width, height, renderStream);

        m_deviceObjects.m_bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.m_tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.m_widgets = m_hostWidgets->GetDeviceInstance();

        cu_deviceData = InstantiateOnDevice<Device::Overlay>(); 

        Synchronise(kSyncObjects);
    }

    Host::Overlay::~Overlay()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Overlay::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceData, syncType);

        if (syncType & kSyncObjects) { SynchroniseObjects2<OverlayObjects>(cu_deviceData, m_deviceObjects); }
        if (syncType & kSyncParams)  { SynchroniseObjects2<OverlayParams>(cu_deviceData, *this); }
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

    /*__host__ void Host::Overlay::TraceRay()
    {
        const auto& tracables = *m_hostTracables;
        Ray2D ray(vec2(0.0f), normalize(m_viewCtx.mousePos));
        HitCtx2D hit;
        
        auto onIntersect = [&](const uint* primRange, RayRange2D& range)
        {
            for (uint idx = primRange[0]; idx < primRange[1]; ++idx)
            {
                if (tracables[idx]->IntersectRay(ray, hit))
                {
                    if (hit.tFar < range.tFar)
                    {
                        range.tFar = hit.tFar;
                    }
                }
            }
        };
        m_hostBIH->TestRay(ray, kFltMax, onIntersect);        
    }*/

    __host__ void Host::Overlay::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        UILayer::Rebuild(dirtyFlags, viewCtx, selectionCtx);
        
        if (!m_dirtyFlags) { return; }
         
        // Calculate some values for the guide grid
        const float logScale = std::log10(m_viewCtx.transform.scale);
        constexpr float kGridScale = 0.05f;
        m_gridCtx.majorLineSpacing = kGridScale * std::pow(10.0f, std::ceil(logScale));
        m_gridCtx.minorLineSpacing = kGridScale * std::pow(10.0f, std::floor(logScale));
        m_gridCtx.lineAlpha = 1 - (logScale - std::floor(logScale));
        m_gridCtx.show = true;
        m_selectionCtx.lassoBBox.Rectify();
        m_selectionCtx.selectedBBox.Rectify();

        // Upload to the device
        Synchronise(kSyncParams);
    }
}