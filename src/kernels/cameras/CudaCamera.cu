﻿#include "CudaCamera.cuh"
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

        maxSamples = -1;
        errorThreshold = 0.0f;
        samplingMode = kCameraSamplingFixed;
        adaptiveSamplingGamma = 1.0f;
    }

    __host__ void CameraParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("live", isLive);
        node.AddValue("active", isActive);
        node.AddValue("splatClamp", splatClamp);
        node.AddValue("seed", seed);
        node.AddValue("randomiseSeed", randomiseSeed);
        node.AddEnumeratedParameter("samplingMode", std::vector<std::string>({ "fixed", "adaptiverelative", "adaptiveabsolute" }), samplingMode);
        node.AddValue("maxSamples", maxSamples);
        node.AddValue("errorThreshold", errorThreshold);
        node.AddValue("adaptiveSamplingGamma", adaptiveSamplingGamma);
        
        auto childNode = node.AddChildObject("overrides");
        childNode.AddValue("maxDepth", overrides.maxDepth);
    }

    __host__ void CameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("live", isLive, flags);
        node.GetValue("active", isActive, flags);
        node.GetValue("splatClamp", splatClamp, flags);
        node.GetValue("seed", seed, flags);
        node.GetValue("randomiseSeed", randomiseSeed, flags);
        node.GetEnumeratedParameter("samplingMode", std::vector<std::string>({ "fixed", "adaptiverelative", "adaptiveabsolute" }), samplingMode, flags);
        node.GetValue("maxSamples", maxSamples, flags);
        node.GetValue("errorThreshold", errorThreshold, flags);
        node.GetValue("adaptiveSamplingGamma", adaptiveSamplingGamma, flags);
        
        auto childNode = node.GetChildObject("overrides", flags);
        if (childNode)
        {
            childNode.GetValue("maxDepth", overrides.maxDepth, flags);
        }
    }
    
    __host__ Host::Camera::Camera(const ::Json::Node& node, const std::string& id, const int rayBufferSize) 
    {
        // Create the packed ray buffer
        m_hostCompressedRayBuffer = AssetHandle<Host::CompressedRayBuffer>(tfm::format("%s_compressedRayBuffer", id), rayBufferSize, m_hostStream);
        m_hostCompressedRayBuffer->Clear(CompressedRay());

        // Create the occupancy buffer and render stats
        m_hostBlockRayOccupancy = AssetHandle<Host::Array<uint>>(tfm::format("%s_blockRayOccupancy", id), rayBufferSize / 32, m_hostStream);
        m_hostRenderStats = AssetHandle < Host::ManagedObject<Device::RenderState::Stats>>(tfm::format("%s_renderStats", id));

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