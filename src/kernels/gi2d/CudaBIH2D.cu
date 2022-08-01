#include "CudaBIH2D.cuh"

namespace Cuda
{
    __host__ Host::BIH2D::BIH2D(const std::string& id) : Asset(id)
    {
        cu_deviceData = InstantiateOnDevice<Device::BIH2D>(GetAssetID());

        m_hostNodes = CreateChildAsset<Host::Array<BIH2DNode>>(tfm::format("%s_nodes", id), this, m_hostStream);
    }

    Host::BIH2D::~BIH2D()
    {
        OnDestroyAsset();
    }

    void Host::BIH2D::OnDestroyAsset()
    {
        m_hostNodes.DestroyAsset();
        
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    void Host::BIH2D::Resize(const size_t numPrimitives)
    {        
        m_hostNodes->Resize(numPrimitives);
    }    
}