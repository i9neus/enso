#include "GI2D.h"

GI2D::GI2D()
{
}

GI2D::~GI2D()
{

}

std::shared_ptr<RendererInterface> GI2D::Instantiate()
{
    return std::make_shared<GI2D>();
}

void GI2D::Initialise()
{

}

void GI2D::Destroy()
{

}

void GI2D::OnResizeClient()
{

}

void GI2D::PreRender()
{

}
    
void GI2D::Render()
{

}

void GI2D::PostRender()
{

}