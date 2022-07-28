#include "GI2D.h"
#include "kernels/2dgi/CudaTestCard.cuh"

using namespace Cuda;

GI2D::GI2D()
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

void GI2D::Initialise()
{
    m_hostTestCard = CreateAsset<TestCard>("id_testCard");
}

void GI2D::OnDestroy()
{
    m_hostTestCard.DestroyAsset();
}

void GI2D::OnResizeClient()
{

}

void GI2D::OnPreRender()
{

}
    
void GI2D::OnRender()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //Log::Write("Tick");

    m_hostTestCard->Composite(m_compositeImage);
}

void GI2D::OnPostRender()
{

}