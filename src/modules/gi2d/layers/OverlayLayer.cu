#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "OverlayLayer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/AssetContainer.cuh"
#include "core/2d/primitives/LineSegment.cuh"
#include "../scene/SceneContainer.cuh"
#include "core/math/ColourUtils.cuh"

#include "io/json/JsonUtils.h"

namespace Enso
{    
    __host__ __device__ OverlayLayerParams::OverlayLayerParams()
    {
        gridCtx.show = true;
        gridCtx.lineAlpha = 0.0;
        gridCtx.majorLineSpacing = 1.0f;
        gridCtx.majorLineSpacing = 1.0f;
    }

    __device__ void Device::OverlayLayer::Composite(Device::ImageRGBA* deviceOutputImage)
    {
        CudaAssertDebug(deviceOutputImage);
        
        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x >= 0 && xyScreen.x < m_objects.accumBuffer->Width() && xyScreen.y >= 0 && xyScreen.y < m_objects.accumBuffer->Height())
        {            
            CompositeBlend(deviceOutputImage->At(xyScreen), m_objects.accumBuffer->At(xyScreen));
            
            //vec4& target = deviceOutputImage->At(xyScreen);
            //target = Blend(target, m_objects.accumBuffer->At(xyScreen));
            //target.xyz += m_objects.accumBuffer->At(xyScreen).xyz;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    template<typename ContainerType>
    __device__ void DrawOverlayElements(const vec2& xyView, const UIViewCtx& viewCtx, const BIH2D<BIH2DFullNode>* bih, const ContainerType* elementListPtr, vec4& L)
    {
        if (!bih || !elementListPtr) { return; }

        const ContainerType& elementList = *elementListPtr;
        auto onPointIntersectLeaf = [&](const uint* idxRange, const uint* primIdxs) -> bool
        {
            for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
            {
                CudaAssertDebug(primIdxs[idx] < elementList.Size());
                CudaAssertDebug(elementList[primIdxs[idx]]);

                const auto& drawable = *elementList[primIdxs[idx]];
                vec4 LPrim = drawable.EvaluateOverlay(xyView, viewCtx, false);
                if (LPrim.w > 0.0f)
                {
                    L = Blend(L, LPrim);
                }

                if (drawable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, viewCtx.dPdXY))
                    L = vec4(kRed, 1.0f);
            }
            return false;
        };
        bih->TestPoint(xyView, onPointIntersectLeaf);
        
        /*for (int idx = 0; idx < elementList.Size(); ++idx)
        {
            CudaAssertDebug(elementList[idx]);

            const auto& drawable = *elementList[idx];
            if (drawable.GetWorldSpaceBoundingBox().Contains(xyView))
            {
                vec4 LPrim = drawable.EvaluateOverlay(xyView, viewCtx, false);
                if (LPrim.w > 0.0f)
                {
                    L = Blend(L, LPrim);
                }
            }

            if (drawable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, viewCtx.dPdXY)) L = vec4(kRed, 1.0f);
        }*/
    }

    __device__ void Device::OverlayLayer::Render()
    {
        CudaAssertDebug(m_objects.accumBuffer);

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
                L = Blend(L, kOne, 0.1 * (1 - m_params.gridCtx.lineAlpha));
            }
            xyGrid = fract(xyView / vec2(m_params.gridCtx.minorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_params.viewCtx.dPdXY / m_params.gridCtx.minorLineSpacing * 1.5f)
            { 
                L = Blend(L, kOne, 0.1 * m_params.gridCtx.lineAlpha);
            }
        }  

        // Draw the tracables and widgets    
        DrawOverlayElements(xyView, m_params.viewCtx, m_objects.scene->sceneBIH, m_objects.scene->sceneObjects, L);

        // Draw the lasso 
        if (m_params.selectionCtx.isLassoing && m_params.selectionCtx.lassoBBox.PointOnPerimiter(xyView, m_params.viewCtx.dPdXY * 2.f)) { L = vec4(kRed, 1.0f); }

        // Draw the selected object's bounding box
        if (m_params.selectionCtx.numSelected > 0 && m_params.selectionCtx.selectedBBox.PointOnPerimiter(xyView, m_params.viewCtx.dPdXY * 2.f)) { L = vec4(kGreen, 1.0f); }

        m_objects.accumBuffer->At(xyScreen) = L;
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    /*__device__ void Device::OverlayLayer::Prepare(const uint dirtyFlags)
    {
        // Save ourselves a deference here by caching the scene pointers
        assert(m_objects.scenePtr);
        m_scene = *m_objects.scenePtr;
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);*/

    Host::OverlayLayer::OverlayLayer(const Asset::InitCtx& initCtx, const AssetHandle<const Host::SceneContainer>& scene, const uint width, const uint height, cudaStream_t renderStream):
        GenericObject(initCtx),
        m_scene(scene)
    {                
        // Create some Cuda objects
        m_hostAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "accumBuffer", width, height, renderStream);

        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.scene = m_scene->GetDeviceInstance();

        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::OverlayLayer>(*this);

        Synchronise(kSyncObjects);
    }

    Host::OverlayLayer::~OverlayLayer() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::OverlayLayer::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::OverlayLayer>(cu_deviceInstance, m_deviceObjects); }
        if (syncFlags & kSyncParams) { SynchroniseObjects<Device::OverlayLayer>(cu_deviceInstance, m_params); }
    }

    __host__ void Host::OverlayLayer::Render()
    {
        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        IsOk(cudaDeviceSynchronize());
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
        Ray2D ray(vec2(0.0f), normalize(UILayer::m_params.viewCtx.mousePos));
        HitCtx2D hit;
        
        auto onIntersect = [&](const uint* primRange, const uint* primIdxs, RayRange2D& range)
        {
            for (uint idx = primRange[0]; idx < primRange[1]; ++idx)
            {
                if (tracables[primIdxs[idx]]->IntersectRay(ray, hit))
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

    __host__ bool Host::OverlayLayer::Prepare(const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {        
        // Copy the view context
        m_params.viewCtx = viewCtx;
        
        // Copy some data to the device selection context
        m_params.selectionCtx.isLassoing = selectionCtx.isLassoing;
        m_params.selectionCtx.lassoBBox = selectionCtx.lassoBBox;
        m_params.selectionCtx.numSelected = selectionCtx.selectedObjects.size();
        m_params.selectionCtx.selectedBBox = selectionCtx.selectedBBox;
         
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
        Synchronise(kSyncParams);
        return true;
    }
}