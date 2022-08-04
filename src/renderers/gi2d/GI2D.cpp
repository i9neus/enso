#include "GI2D.h"
#include "kernels/gi2d/CudaGI2DOverlay.cuh"
#include "generic/Math.h"

using namespace Cuda;

GI2D::GI2D() :
    m_overlayParams(std::make_unique<GI2DOverlayParams>())
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
    m_overlayParams->viewMatrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
    m_view.zoomSpeed = 10.0f;

    m_overlayRenderer = CreateAsset<Host::GI2DOverlay>("id_gi2DOverlay");
    m_overlayRenderer->SetParams(*m_overlayParams);
}

void GI2D::OnDestroy()
{
    m_overlayRenderer.DestroyAsset();
}

void GI2D::OnPreRender()
{

}

void GI2D::OnRender()
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //Log::Write("Tick");

    if (m_dirtyFlags == kGI2DDirty)
    {
        m_overlayRenderer->SetParams(*m_overlayParams);
        m_dirtyFlags = kGI2DClean;
    }

    m_overlayRenderer->Render(m_compositeImage);
}

void GI2D::OnPostRender()
{

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
            if (IsMouseButtonDown(kMouseLButton))
            {
                // Update the transformation
                m_overlayParams->viewMatrix = ConstructViewMatrix(m_view.transAnchor, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
                const vec2 dragDelta = TransformSceenToView2D(m_overlayParams->viewMatrix, vec2(m_view.dragAnchor)) - TransformSceenToView2D(m_overlayParams->viewMatrix, vec2(m_mouse.pos));
                m_view.trans = m_view.transAnchor + dragDelta;

                //Log::Write("Trans: %s", m_view.trans.format());
            }
            // Zooming?
            else if (IsMouseButtonDown(kMouseRButton))
            {
                float logScaleAnchor = std::log2(::math::max(1e-10f, m_view.scaleAnchor));
                float inc = m_view.zoomSpeed * float(m_mouse.pos.y - m_view.dragAnchor.y) / m_clientHeight;
                logScaleAnchor += inc;
                m_view.scale = std::pow(2.0, logScaleAnchor);                
                
                Log::Write("Scale: %f", m_view.scale);
            }
            // Rotating?
            else if (IsMouseButtonDown(kMouseMButton))
            {
                const vec2 delta = normalize(vec2(m_mouse.pos) - vec2(m_clientWidth, m_clientHeight) * 0.5f);
                const float theta = std::acos(dot(delta, m_view.rotAxis)) * (float(dot(delta, vec2(m_view.rotAxis.y, -m_view.rotAxis.x)) < 0.0f) * 2.0 - 1.0f);

                m_view.rotate = m_view.rotAnchor + theta;
                Log::Write("Theta: %f", m_view.rotate);
            }

            // Update the parameters in the overlay renderer
            m_overlayParams->viewMatrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
            m_overlayParams->viewScale = m_view.scale;

            // Mark the scene as dirty
            m_dirtyFlags = kGI2DDirty;
        }
    }
}

void GI2D::OnMouseWheel()
{

}

void GI2D::OnResizeClient()
{
}