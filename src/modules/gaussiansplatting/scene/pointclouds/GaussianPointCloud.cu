#include "GaussianPointCloud.cuh"

namespace Enso
{
    __host__ Host::GaussianPointCloud::GaussianPointCloud(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::GaussianPointCloud>(*this))
    {
        m_hostSplatList = AssetAllocator::CreateChildAsset<Host::Vector<GaussianPoint>>(*this, "pointlist");
        m_deviceObjects.splats = m_hostSplatList->GetDeviceInstance();
        
        Synchronise(kSyncObjects);
    }

    __host__ Host::GaussianPointCloud::~GaussianPointCloud() noexcept
    {
        m_hostSplatList.DestroyAsset();
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::GaussianPointCloud::Synchronise(const uint syncFlags) 
    {
        if (syncFlags & kSyncObjects)
        {
            SynchroniseObjects<Device::GaussianPointCloud>(cu_deviceInstance, m_deviceObjects);
        }
    }

    __host__ void Host::GaussianPointCloud::AppendSplats(const std::vector<GaussianPoint>& points)
    {
        m_hostSplatList->insert(points);
    }

    __host__ void Host::GaussianPointCloud::Finalise()
    {
        m_hostSplatList->Synchronise(kVectorSyncUpload);
    }
}