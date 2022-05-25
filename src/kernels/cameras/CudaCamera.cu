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
    __host__ __device__ CameraParams::CameraParams() 
    {
        isLive = false;
        isActive = true;
        overrides.maxDepth = -1;
        seed = 0;
        randomiseSeed = false;

        minMaxSamples = ivec2(-1, -1);
        errorThreshold = 0.0f;
        samplingMode = kCameraSamplingFixed;
        adaptiveSamplingGamma = 1.0f;
        useFilteredError = false;
    }

    __host__ void CameraParams::ToJson(::Json::Node& node) const
    {        
        node.AddValue("live", isLive);
        node.AddValue("active", isActive);
        node.AddValue("splatClamp", splatClamp);
        node.AddValue("seed", seed);
        node.AddValue("randomiseSeed", randomiseSeed);
        node.AddEnumeratedParameter("samplingMode", std::vector<std::string>({ "fixed", "adaptiverelative", "adaptiveabsolute" }), samplingMode);
        node.AddVector("minMaxSamples", minMaxSamples);
        node.AddValue("errorThreshold", errorThreshold);
        node.AddValue("adaptiveSamplingGamma", adaptiveSamplingGamma);
        node.AddValue("useFilteredError", useFilteredError);
        
        auto childNode = node.AddChildObject("overrides");
        childNode.AddValue("maxDepth", overrides.maxDepth);
        childNode.AddValue("minDepth", overrides.minDepth);
    }

    __host__ uint CameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("live", isLive, flags);
        node.GetValue("active", isActive, flags);
        node.GetValue("splatClamp", splatClamp, flags);
        node.GetValue("seed", seed, flags);
        node.GetValue("randomiseSeed", randomiseSeed, flags);
        node.GetEnumeratedParameter("samplingMode", std::vector<std::string>({ "fixed", "adaptiverelative", "adaptiveabsolute" }), samplingMode, flags);
        node.GetVector("minMaxSamples", minMaxSamples, flags);
        node.GetValue("errorThreshold", errorThreshold, flags);
        node.GetValue("adaptiveSamplingGamma", adaptiveSamplingGamma, flags);
        node.GetValue("useFilteredError", useFilteredError, flags);
        
        auto childNode = node.GetChildObject("overrides", flags);
        if (childNode)
        {
            childNode.GetValue("maxDepth", overrides.maxDepth, flags);
            childNode.GetValue("minDepth", overrides.minDepth, flags);
        }

        return kRenderObjectClean;
    }
    
    __host__ Host::Camera::Camera(const ::Json::Node& node, const std::string& id, const int rayBufferSize) :
        Host::RenderObject(id),
        m_frameIdx(0)
    {
        // Create the packed ray buffer
        m_hostCompressedRayBuffer = CreateChildAsset<Host::CompressedRayBuffer>(tfm::format("%s_compressedRayBuffer", id), this, rayBufferSize, m_hostStream);
        m_hostCompressedRayBuffer->Clear(CompressedRay());

        // Create the occupancy buffer and render stats
        m_hostBlockRayOccupancy = CreateChildAsset<Host::Array<uint>>(tfm::format("%s_blockRayOccupancy", id), this, rayBufferSize / 32, m_hostStream);
        m_hostRenderStats = CreateChildAsset< Host::ManagedObject<Device::RenderState::Stats>>(tfm::format("%s_renderStats", id), this);

        // Set the DAG path       
        Host::RenderObject::UpdateDAGPath(node);
    }

    __host__ void Host::Camera::OnDestroyAsset()
    {
        m_hostCompressedRayBuffer.DestroyAsset();
        m_hostBlockRayOccupancy.DestroyAsset();
        m_hostRenderStats.DestroyAsset();
    }
}