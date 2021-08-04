#include "CudaCamera.cuh"
#include "generic/JsonUtils.h"

#include "../CudaSampler.cuh"
#include "../CudaHash.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRay.cuh"

#include "CudaPerspectiveCamera.cuh"
#include "generic/JsonUtils.h"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

namespace Cuda
{
    __host__ Host::Camera::Camera(const ::Json::Node& node, const std::string& id) : 
        m_isLiveCamera(false)
    {
        // Create the packed ray buffer
        m_hostCompressedRayBuffer = AssetHandle<Host::CompressedRayBuffer>("id_hostCompressedRayBuffer", 512 * 512, m_hostStream);
        m_hostCompressedRayBuffer->Clear(CompressedRay());

        // Create the occupancy buffer and render stats
        m_hostBlockRayOccupancy = AssetHandle<Host::Array<uint>>("id_hostBlockRayOccupancy", 512 * 512 / 32, m_hostStream);
        m_hostRenderStats = AssetHandle < Host::ManagedObject<Device::RenderState::Stats>>(tfm::format("%s_renderStats", id));
    }

    __host__ void Host::Camera::OnDestroyAsset()
    {
        m_hostCompressedRayBuffer.DestroyAsset();
        m_hostBlockRayOccupancy.DestroyAsset();
        m_hostRenderStats.DestroyAsset();
    }
    
    __host__ void Host::Camera::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::RenderObject::FromJson(node, flags);

        node.GetValue("live", m_isLiveCamera, flags);
    }
}