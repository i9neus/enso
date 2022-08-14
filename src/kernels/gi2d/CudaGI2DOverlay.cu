#include "CudaGI2DOverlay.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

namespace Cuda
{    
    __host__ __device__ GI2DOverlayParams::GI2DOverlayParams()
    {
        grid.show = true;
        grid.lineAlpha = 0.0;
        grid.majorLineSpacing = 1.0f;
        grid.majorLineSpacing = 1.0f;

        selection.isLassoing = false;

        mousePosView = vec2(0.f);
        rayOriginView = vec2(0.f);
        viewScale = 1.0f;
        sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

        selectedSegmentIdx = kInvalidSegment;
    }

    __device__ Device::GI2DOverlay::GI2DOverlay(const GI2DOverlayParams& params, const Objects& objects) :
        m_params(params),
        m_objects(objects)
    {
    }

    __device__ void Device::GI2DOverlay::Synchronise(const GI2DOverlayParams& params)
    {
        m_params = params;
    }

    __device__ void Device::GI2DOverlay::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __host__ __device__ __forceinline__ bool PointOnPerimiter(const BBox2f& bBox, const vec2& p, float thickness)
    {
        thickness *= 0.5f;
        return (p.x >= bBox.lower.x - thickness && p.y >= bBox.lower.y - thickness && p.x <= bBox.upper.x + thickness && p.y <= bBox.upper.y + thickness) &&
            (p.x <= bBox.lower.x + thickness || p.y <= bBox.lower.y + thickness || p.x >= bBox.upper.x - thickness || p.y >= bBox.upper.y - thickness);
    }

    __device__ void Device::GI2DOverlay::Render(Device::ImageRGBA* deviceOutputImage)
    {
        assert(deviceOutputImage);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= deviceOutputImage->Width() || xyScreen.y < 0 || xyScreen.y >= deviceOutputImage->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_params.viewMatrix * vec2(xyScreen);

        vec3 L = vec3(0.1f);
        if (!m_params.sceneBounds.Contains(xyView))
        {
            deviceOutputImage->At(xyScreen) = vec4(vec3(L), 1.0f);
            return;
        }

        // Draw the grid
        if (m_params.grid.show)
        {
            vec2 xyGrid = fract(xyView / vec2(m_params.grid.majorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < 0.02f * mix(1.0, 0.1, m_params.grid.lineAlpha)) { L = 0.3f; }
            xyGrid = fract(xyView / vec2(m_params.grid.minorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < 0.02f) { L = max(L, kOne * 0.3f * m_params.grid.lineAlpha); }
        }

        if (m_objects.bih && m_objects.lineSegments)
        {
            const Vector<LineSegment>& segments = *(m_objects.lineSegments);

            /*LineSegment ray(m_params.rayOriginView, normalize(m_params.mousePosView));
            const float line = ray.Evaluate(xyView, 0.001f, m_params.dPdXY);
            if (line > 0.f)
            {
                L = mix(L, vec3(1.0f, 0.8f, 0.05f), line);
            }

            const Vector<LineSegment>& segments = *(m_objects.lineSegments);
            int hitSegment = -1;
            auto onRayIntersectLeaf = [&, this](const uint& idx, float& tNearest) -> void
            {
                //if (segments[idx].GetBoundingBox().Contains(xyView)) { L = mix(L, kRed, 0.2f); }
                if (PointOnPerimiter(segments[idx].GetBoundingBox(), xyView, m_params.dPdXY * 2.)) { L = kOne; }
                float tPrim = segments[idx].TestRay(ray.v, ray.dv);
                if (tPrim != kFltMax && tPrim < tNearest)
                {
                    tNearest = tPrim;
                    hitSegment = idx;
                }
            };
            auto onRayIntersectInner = [&, this](const BBox2f& bBox, const vec2& t, const bool isLeaf) -> void
            {
                if (PointOnPerimiter(bBox, xyView, m_params.dPdXY * 2.)) { L = kRed; }
                else if (isLeaf && bBox.Contains(xyView)) { L = mix(L, kRed, 0.2f); }

                if (length2(ray.v + ray.dv * t[0] - xyView) < sqr(m_params.dPdXY * 5.0)) L = kYellow;
                if (length2(ray.v + ray.dv * t[1] - xyView) < sqr(m_params.dPdXY * 5.0)) L = kBlue;
            };
            m_objects.bih->TestRay(ray.v, ray.dv, onRayIntersectLeaf, onRayIntersectInner);*/

            auto onPointIntersectLeaf = [&, this](const uint& idxStart, const uint& idxEnd) -> void
            {
                for (int idx = idxStart; idx < idxEnd; ++idx)
                {
                    const float line = segments[idx].Evaluate(xyView, 0.001f, m_params.dPdXY);
                    if (line > 0.f)
                    {
                        L = mix(L, segments[idx].IsSelected() ? vec3(1.0f, 0.1f, 0.0f) : kOne, line);
                        //L += kRed;
                    }
                }
            };
            /*auto onPointIntersectInner = [&, this](BBox2f bBox, const uchar& depth) -> void
            {
                //bBox.Grow(m_params.dPdXY * depth * -5.0f);
                //if (bBox.Contains(xyView)) { L = mix(L, Hue(depth / 5.0f), 0.3f); }
                //if (PointOnPerimiter(bBox, xyView, m_params.dPdXY * 2.0)) { L = kOne; }
            };*/
            m_objects.bih->TestPoint(xyView, onPointIntersectLeaf/*, onPointIntersectInner*/);

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

        // Draw the selection box
        if (m_params.selection.isLassoing)
        {
            if (PointOnPerimiter(m_params.selection.bBox, xyView, m_params.dPdXY * 1.)) { L = kOne; }
        }

        deviceOutputImage->At(xyScreen) = vec4(vec3(L), 1.0f);
    }

    DEFINE_KERNEL_PASSTHROUGH_ARGS(Render);

    Host::GI2DOverlay::GI2DOverlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Host::Vector<LineSegment>>& lineSegments) :
        Asset(id),
        m_hostBIH2D(bih),
        m_hostLineSegments(lineSegments)
    {
        m_objects.bih = m_hostBIH2D->GetDeviceInstance();
        m_objects.lineSegments = m_hostLineSegments->GetDeviceInstance();

        cu_deviceData = InstantiateOnDevice<Device::GI2DOverlay>(GetAssetID(), m_params, m_objects); 
    }

    Host::GI2DOverlay::~GI2DOverlay()
    {
        OnDestroyAsset();
    }

    __host__ void Host::GI2DOverlay::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ void Host::GI2DOverlay::Render(AssetHandle<Host::ImageRGBA>& hostOutputImage)
    {
        /*auto onPointIntersectLeaf = [&, this](const uint& idx) -> void {};
        auto onPointIntersectInner = [&, this](BBox2f bBox, const uchar& depth) -> void { };
        m_hostBIH2D->TestPrimitive(m_params.mousePosView, onPointIntersectLeaf, onPointIntersectInner);

        const vec2 o = m_params.rayOriginView, d = normalize(m_params.mousePosView);
        auto onRayIntersectLeaf = [&, this](const uint& idx, float& tNearest) -> void 
        {
            float tPrim = (*m_hostLineSegments)[idx].TestRay(o, d);
            if (tPrim < tNearest)
            {
                tNearest = tPrim;
            }
        };
        auto onRayIntersectInner = [&, this](const BBox2f& bBox, const vec2& t, const bool&) -> void {};        
        m_hostBIH2D->TestRay(o, d, onRayIntersectLeaf, onRayIntersectInner);*/
        
        const auto& meta = hostOutputImage->GetMetadata();
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);    

        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::GI2DOverlay::SetParams(const GI2DOverlayParams& newParams)
    {
        m_params = newParams;

        //Log::Warning("%s", vec2(m_params.viewMatrix.i02, m_params.viewMatrix.i12).format());

        const float logScale = std::log10(m_params.viewScale);
        constexpr float kGridScale = 0.05f;

        m_params.grid.majorLineSpacing = kGridScale * std::pow(10.0f, std::ceil(logScale));
        m_params.grid.minorLineSpacing = kGridScale * std::pow(10.0f, std::floor(logScale));
        m_params.grid.lineAlpha = 1 - (logScale - std::floor(logScale));
        m_params.dPdXY = length(vec2(m_params.viewMatrix.i00, m_params.viewMatrix.i10));
        m_params.selection.bBox.Rectify();

        if (m_hostBIH2D->IsConstructed())
        {
            const Vector<LineSegment>& segments = *m_hostLineSegments;
            auto onIntersectLeaf = [&, this](const uint idx)
            {
                if (segments[idx].TestPoint(m_params.mousePosView, m_params.dPdXY * 5.0f))
                {
                    m_params.selectedSegmentIdx = idx;
                }
            };

            //m_hostBIH2D->TestPrimitive(m_params.mousePosView, onIntersectLeaf);
        }

        SynchroniseObjects(cu_deviceData, m_params);
    }
}