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
        grid.show = true;
        grid.lineAlpha = 0.0;
        grid.majorLineSpacing = 1.0f;
        grid.majorLineSpacing = 1.0f;

        selection.isLassoing = false;
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
    }

    __host__ __device__ __forceinline__ bool PointOnPerimiter(const BBox2f& bBox, const vec2& p, float thickness)
    {
        thickness *= 0.5f;
        return (p.x >= bBox.lower.x - thickness && p.y >= bBox.lower.y - thickness && p.x <= bBox.upper.x + thickness && p.y <= bBox.upper.y + thickness) &&
            (p.x <= bBox.lower.x + thickness || p.y <= bBox.lower.y + thickness || p.x >= bBox.upper.x - thickness || p.y >= bBox.upper.y - thickness);
    }

    __device__ __forceinline__ vec4 Blend(vec4 lowerRgba, const vec3 upperRgba, const float& upperAlpha)
    {
        lowerRgba.xyz = lowerRgba.xyz * (1.0f - upperAlpha) + upperRgba * upperAlpha;
        lowerRgba.w = mix(lowerRgba.w, 1.0, upperAlpha);
        return lowerRgba;
    }

    __device__ void Device::Overlay::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {
        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x >= 0 && xyScreen.x < m_objects.accumBuffer->Width() && xyScreen.y >= 0 && xyScreen.y < m_objects.accumBuffer->Height())
        {
            vec4& target = deviceOutputImage->At(xyScreen);
            const vec4& source = m_objects.accumBuffer->At(xyScreen);
            target = Blend(target, source.xyz, source.w);
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    __device__ void Device::Overlay::Render()
    {
        assert(m_objects.accumBuffer);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_objects.accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_objects.accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_params.view.matrix * vec2(xyScreen);

        //m_objects.accumBuffer->At(xyScreen) = vec4(xyView, 0.0f, 1.0f);
        //return;

        vec4 L(0.0f);

        // Draw the grid
        if (!m_params.view.sceneBounds.Contains(xyView)) 
        { 
            L = vec4(0.0f);
        }
        else if (m_params.grid.show)
        {
            vec2 xyGrid = fract(xyView / vec2(m_params.grid.majorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_params.view.dPdXY / m_params.grid.majorLineSpacing * mix(1.0f, 3.0f, m_params.grid.lineAlpha)) 
            { 
                L = Blend(L, kOne, 0.5 * (1 - m_params.grid.lineAlpha));
            }
            xyGrid = fract(xyView / vec2(m_params.grid.minorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_params.view.dPdXY / m_params.grid.minorLineSpacing * 1.5f)
            { 
                L = Blend(L, kOne, 0.5 * m_params.grid.lineAlpha);
            }
        }      

        if (m_objects.bih && m_objects.tracables)
        {
            const Cuda::Device::Vector<Tracable*>& tracables = *(m_objects.tracables);
      
            /*auto onPointIntersectLeaf = [&, this](const uint* idxRange) -> void
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    const float line = tracables[idx]->Evaluate(xyView, 0.001f, m_params.view.dPdXY);
                    if (line > 0.f)
                    {
                        L = Blend(L, segments[idx].IsSelected() ? vec3(1.0f, 0.1f, 0.0f) : kOne, line);
                        //L += kRed;
                    }
                }
            };*/
            /*auto onPointIntersectInner = [&, this](BBox2f bBox, const uchar& depth) -> void
            {
                //bBox.Grow(m_params.dPdXY * depth * -5.0f);
                //if (bBox.Contains(xyView)) { L = mix(L, Hue(depth / 5.0f), 0.3f); }
                //if (PointOnPerimiter(bBox, xyView, m_params.dPdXY * 2.0)) { L = kOne; }
            };*/
            //m_objects.bih->TestPoint(xyView, onPointIntersectLeaf/*, onPointIntersectInner*/);

            /*for (int idx = 0; idx < segments.Size(); ++idx)
            {
                const float line = segments[idx].Evaluate(xyView, 0.001f, m_params.dPdXY);
                if (line > 0.f)
                {
                    //L = mix(L, (idx == m_params.selectedSegmentIdx || idx == hitSegment) ? vec3(1.0f, 0.1f, 0.0f) : kOne, line);
                    L += kBlue;
                }
            }*/
        }

        // Draw the lasso 
        if (m_params.selection.isLassoing && PointOnPerimiter(m_params.selection.lassoBBox, xyView, m_params.view.dPdXY * 2.)) { L = vec4(kRed, 1.0f); }

        if (m_params.selection.numSelected > 0. && PointOnPerimiter(m_params.selection.selectedBBox, xyView, m_params.view.dPdXY * 2.)) { L = vec4(kGreen, 1.0f); }

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

        cu_deviceData = InstantiateOnDevice<Device::Overlay>(GetAssetID(), m_params, m_objects); 
    }

    Host::Overlay::~Overlay()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Overlay::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::Overlay::Render()
    {
        if (!m_isDirty) { return; }
        
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());

        m_isDirty = false;
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
        if (!m_isDirty) { return; }        

        m_params.view = m_viewCtx.transform;
         
        // Calculate some values for the guide grid
        const float logScale = std::log10(m_params.view.scale);
        constexpr float kGridScale = 0.05f;
        m_params.grid.majorLineSpacing = kGridScale * std::pow(10.0f, std::ceil(logScale));
        m_params.grid.minorLineSpacing = kGridScale * std::pow(10.0f, std::floor(logScale));
        m_params.grid.lineAlpha = 1 - (logScale - std::floor(logScale));
        m_params.grid.show = true;
        m_params.selection.lassoBBox.Rectify();
        m_params.selection.selectedBBox.Rectify();

        // Upload to the device
        SynchroniseObjects(cu_deviceData, m_params);
    }
}