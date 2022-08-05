#include "CudaGI2DOverlay.cuh"
#include "kernels/math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__ float LineSegment::Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const
    {
        const vec2 perp = v + saturate((dot(p, dv) - dot(v, dv)) / dot(dv, dv)) * dv;
        return saturate(1.0f - (length(p - perp) - thickness) / dPdXY);
    }
    
    __host__ __device__ GI2DOverlayParams::GI2DOverlayParams()
    {
        grid.show = true;
        grid.lineAlpha = 0.0;
        grid.majorLineSpacing = 1.0f;
        grid.majorLineSpacing = 1.0f;

        mousePosView = vec2(-kFltMax);
        viewScale = 1.0f;
        sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

        selectedSegmentIdx = kInvalidSegment;
    }
    
    __device__ Device::GI2DOverlay::GI2DOverlay(const GI2DOverlayParams& params) :
        m_params(params)
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

        LineSegment segment(vec2(-1.0f), vec2(1.0f, 1.5f));
        const float dPdXY = length(vec2(m_params.viewMatrix.i00, m_params.viewMatrix.i10));
        float line = segment.Evaluate(xyView, 0.001f, dPdXY);

        L = (segment.Evaluate(m_params.mousePosView, 0.001f + 2.f * dPdXY, dPdXY) > 0.f) ?
            mix(L, kOne, line) : mix(L, vec3(1.f, 0.f, 0.f), line);

        deviceOutputImage->At(xyScreen) = vec4(vec3(L), 1.0f);
    }

    DEFINE_KERNEL_PASSTHROUGH_ARGS(Render);

    Host::GI2DOverlay::GI2DOverlay(const std::string& id) :
        Asset(id)
    {
        //m_hostBIH2D = CreateChildAsset<Host::BIH2D>("id_bih2D", this);
        m_hostLineSegments = CreateChildAsset<Host::Array<LineSegment>>("id_lineSegments", this, 0, m_hostStream);
        
        cu_deviceData = InstantiateOnDevice<Device::GI2DOverlay>(GetAssetID(), m_params);
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

        SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ void Host::GI2DOverlay::UpdateLineSegments(const std::list<LineSegment>& segments)
    {

    }
}