#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "OverlayLayer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"
#include "core/AssetContainer.cuh"
#include "../tracables/primitives/LineSegment.cuh"
#include "../SceneDescription.cuh"

namespace Enso
{    
    __host__ __device__ OverlayLayerParams::OverlayLayerParams()
    {
        m_gridCtx.show = true;
        m_gridCtx.lineAlpha = 0.0;
        m_gridCtx.majorLineSpacing = 1.0f;
        m_gridCtx.majorLineSpacing = 1.0f;
    }

    __device__ void Device::OverlayLayer::OnSynchronise(const int syncFlags)
    {

    }

    __device__ void Device::OverlayLayer::Composite(Device::ImageRGBA* deviceOutputImage)
    {
        assert(deviceOutputImage);
        
        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x >= 0 && xyScreen.x < m_accumBuffer->Width() && xyScreen.y >= 0 && xyScreen.y < m_accumBuffer->Height())
        {            
            deviceOutputImage->BlendPixel(xyScreen, m_accumBuffer->At(xyScreen));
            //vec4& target = deviceOutputImage->At(xyScreen);
            //target = Blend(target, m_accumBuffer->At(xyScreen));
            //target.xyz += m_accumBuffer->At(xyScreen).xyz;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    __device__ void Device::OverlayLayer::Render()
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
        if (m_scene.tracableBIH && m_scene.tracables)
        {            
            const Vector<Device::Tracable*>& tracableList = *(m_scene.tracables);
      
            auto onPointIntersectLeaf = [&, this](const uint* idxRange) -> bool
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    assert(idx < tracableList.Size());
                    assert(tracableList[idx]);

                    const auto& drawable = *tracableList[idx];
                    vec4 LPrim = drawable.EvaluateOverlay(xyView, m_viewCtx);
                    if(LPrim.w > 0.0f)
                    {                        
                        L = Blend(L, LPrim); 
                    }

                    if (drawable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, m_viewCtx.dPdXY)) L = vec4(kRed, 1.0f);                  
                }
                return false;
            };          
            m_scene.tracableBIH->TestPoint(xyView, onPointIntersectLeaf);
        }

        // Draw the lasso 
        if (m_selectionCtx.isLassoing && m_selectionCtx.lassoBBox.PointOnPerimiter(xyView, m_viewCtx.dPdXY * 2.f)) { L = vec4(kRed, 1.0f); }

        // Draw the selected object's bounding box
        if (m_selectionCtx.numSelected > 0 && m_selectionCtx.selectedBBox.PointOnPerimiter(xyView, m_viewCtx.dPdXY * 2.f)) { L = vec4(kGreen, 1.0f); }

        m_accumBuffer->At(xyScreen) = L;
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::OverlayLayer::Prepare(const uint dirtyFlags)
    {
        // Save ourselves a deference here by caching the scene pointers
        assert(m_scenePtr);
        m_scene = *m_scenePtr;
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);

    Host::OverlayLayer::OverlayLayer(const std::string& id, const AssetHandle<Host::SceneDescription>& scene, const uint width, const uint height, cudaStream_t renderStream) :
        UILayer(id, scene)
    {        
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Host::ImageRGBW>("accumBuffer", width, height, renderStream);

        m_deviceObjects.m_scenePtr = m_scene->GetDeviceInstance();
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        cu_deviceInstance = InstantiateOnDevice<Device::OverlayLayer>();

        Synchronise(kSyncObjects);
    }

    Host::OverlayLayer::~OverlayLayer()
    {
        OnDestroyAsset();
    }

    __host__ void Host::OverlayLayer::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceInstance, syncType);

        if (syncType & kSyncObjects) { SynchroniseInheritedClass<OverlayLayerObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects); }
        if (syncType & kSyncParams) { SynchroniseInheritedClass<OverlayLayerParams>(cu_deviceInstance, *this, kSyncParams); }
    }

    __host__ void Host::OverlayLayer::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::OverlayLayer::Render()
    {
        if (!m_dirtyFlags) { return; }

        KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        IsOk(cudaDeviceSynchronize());

        m_dirtyFlags = 0;
    }

    __host__ void Host::OverlayLayer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    /*__host__ void Host::OverlayLayer::TraceRay()
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

    __host__ void Host::OverlayLayer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
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