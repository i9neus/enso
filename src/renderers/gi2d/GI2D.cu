#include "GI2D.cuh"
#include "generic/Math.h"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/CudaGI2DOverlay.cuh"
#include "kernels/gi2d/CudaPrimitive2D.cuh"

using namespace Cuda;

struct CudaObjects
{
    Cuda::AssetHandle<Cuda::Host::GI2DOverlay>                  overlayRenderer;
    Cuda::AssetHandle<Cuda::Host::Vector<Cuda::LineSegment>>    hostLineSegments;

    Cuda::GI2DOverlayParams                                     overlayParams;
    Cuda::AssetHandle<Cuda::Host::BIH2DAsset>                   primitiveBIH;
};

GI2D::GI2D() :
    m_objects(std::make_unique<CudaObjects>())
{
}

GI2D::~GI2D()
{
    Destroy();
}

std::shared_ptr<RendererInterface> GI2D::Instantiate()
{
    return std::make_shared<GI2D>();
}

void GI2D::OnInitialise()
{
    m_view.trans = vec2(0.5f);
    m_view.scale = 1.0f;
    m_view.rotate = 0.0;
    m_objects->overlayParams.viewMatrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
    m_view.zoomSpeed = 10.0f;   

    //m_primitiveContainer.Create(m_renderStream);

    m_objects->hostLineSegments = CreateAsset<Host::Vector<LineSegment>>("id_lineSegments", kVectorHostAlloc, m_renderStream);
    
    m_objects->primitiveBIH = CreateAsset<Host::BIH2DAsset>("id_gi2DBIH");
    m_objects->overlayRenderer = CreateAsset<Host::GI2DOverlay>("id_gi2DOverlay", m_objects->primitiveBIH, m_objects->hostLineSegments);
    
    auto& primIdxs = m_objects->primitiveBIH->GetPrimitiveIndices();
    
    constexpr int kCircleSegs = 10;
    Host::Vector<LineSegment>& segments = *m_objects->hostLineSegments;
    primIdxs.resize(kCircleSegs);
    segments.Resize(kCircleSegs);
    for (uint idx = 0; idx < kCircleSegs; ++idx)
    {
        const float theta0 = kTwoPi * float(idx) / float(kCircleSegs);
        const float theta1 = kTwoPi * float(idx + 1) / float(kCircleSegs);
        segments[idx] = LineSegment(vec2(std::cos(theta0), std::sin(theta0)) * 0.25f, vec2(std::cos(theta1), std::sin(theta1)) * 0.25f);
        primIdxs[idx] = idx;
    }
    segments.Synchronise(kVectorSyncUpload);
    
    // Construct the BVH
    std::function<BBox2f(uint)> getPrimitiveBBox = [&segments](const uint& idx) -> BBox2f
    {
        return segments[idx].GetBoundingBox();
    };
    m_objects->primitiveBIH->Build(getPrimitiveBBox);

    SetDirtyFlags(kGI2DDirtyLineSegments | kGI2DDirtyParams);
}

void GI2D::OnDestroy()
{
    m_objects->overlayRenderer.DestroyAsset();
    m_objects->primitiveBIH.DestroyAsset();
    m_objects->hostLineSegments.DestroyAsset();
}

void GI2D::OnRender()
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //Log::Write("Tick");

    if (m_dirtyFlags & kGI2DDirtyParams)
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_objects->overlayRenderer->SetParams(m_objects->overlayParams);

        ClearDirtyFlags(kGI2DDirtyParams);
    }
    if (m_dirtyFlags & kGI2DDirtyLineSegments)
    {

        ClearDirtyFlags(kGI2DDirtyLineSegments);
    }

    m_objects->overlayRenderer->Render(m_compositeImage);
}

void GI2D::OnKey(const uint code, const bool isSysKey, const bool isDown)
{

}

void GI2D::OnMouseButton(const uint code, const bool isDown)
{
    // Is the view being changed?
    if (IsKeyDown(VK_CONTROL))
    {
        // Dragging?
        if (code & (kMouseLButton | kMouseRButton | kMouseMButton))
        {
            m_view.dragAnchor = vec2(m_mouse.pos);
            m_view.rotAxis = normalize(m_view.dragAnchor - vec2(m_clientWidth, m_clientHeight) * 0.5f);
            m_view.transAnchor = m_view.trans;
            m_view.scaleAnchor = m_view.scale;
            m_view.rotAnchor = m_view.rotate;
        }
        if (code & kMouseLButton && !isDown)
        {

        }
        else if (code & kMouseRButton && !isDown)
        {

        }
    }
    else if(code == kMouseLButton && isDown)
    {
       
    }
}

mat3 GI2D::ConstructViewMatrix(const vec2& trans, const float rotate, const float scale) const
{
    const float sinTheta = std::sin(rotate);
    const float cosTheta = std::cos(rotate);
    mat3 m = mat3::Indentity();
    m.i00 = scale * cosTheta; m.i01 = scale * sinTheta;
    m.i10 = scale * sinTheta; m.i11 = scale * -cosTheta;
    m.i02 = trans.x;
    m.i12 = trans.y;
    return m;
}

void GI2D::OnMouseMove()
{
    // Is the view being changed?
    if (IsKeyDown(VK_CONTROL))
    {
        // Dragging?
        if (IsMouseButtonDown(kMouseLButton | kMouseRButton | kMouseMButton))
        {
            OnViewChange();
        }
    }
    
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_objects->overlayParams.mousePosView = m_objects->overlayParams.viewMatrix * vec2(m_mouse.pos);
    }

    // Mark the scene as dirty
    SetDirtyFlags(kGI2DDirtyParams);
}

void GI2D::OnViewChange()
{
    if (IsMouseButtonDown(kMouseLButton))
    {
        // Update the transformation
        m_objects->overlayParams.viewMatrix = ConstructViewMatrix(m_view.transAnchor, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
        const vec2 dragDelta = (m_objects->overlayParams.viewMatrix * vec2(m_view.dragAnchor)) - (m_objects->overlayParams.viewMatrix * vec2(m_mouse.pos));
        m_view.trans = m_view.transAnchor + dragDelta;

        //Log::Write("Trans: %s", m_view.trans.format());
    }
    // Zooming?
    else if (IsMouseButtonDown(kMouseRButton))
    {
        float logScaleAnchor = std::log2(::math::max(1e-10f, m_view.scaleAnchor));
        logScaleAnchor += m_view.zoomSpeed * float(m_mouse.pos.y - m_view.dragAnchor.y) / m_clientHeight;
        m_view.scale = std::pow(2.0, logScaleAnchor);

        //Log::Write("Scale: %f", m_view.scale);
    }
    // Rotating?
    else if (IsMouseButtonDown(kMouseMButton))
    {
        const vec2 delta = normalize(vec2(m_mouse.pos) - vec2(m_clientWidth, m_clientHeight) * 0.5f);
        const float theta = std::acos(dot(delta, m_view.rotAxis)) * (float(dot(delta, vec2(m_view.rotAxis.y, -m_view.rotAxis.x)) < 0.0f) * 2.0 - 1.0f);
        m_view.rotate = m_view.rotAnchor + theta;

        if (std::abs(std::fmod(m_view.rotate, kHalfPi)) < 0.05f) { m_view.rotate = std::round(m_view.rotate / kHalfPi) * kHalfPi; }

        //Log::Write("Theta: %f", m_view.rotate);
    }

    // Update the parameters in the overlay renderer
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_objects->overlayParams.viewMatrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
        m_objects->overlayParams.viewScale = m_view.scale;
    }
}

void GI2D::OnMouseWheel()
{

}

void GI2D::OnResizeClient()
{
}