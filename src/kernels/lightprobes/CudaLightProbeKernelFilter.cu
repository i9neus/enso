#include "CudaLightProbeKernelFilter.cuh"
#include "../CudaManagedArray.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void LightProbeKernelFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddEnumeratedParameter("filterType", std::vector<std::string>({ "gaussian" }), filterType);
        node.AddValue("radius", radius);
    }

    __host__ void LightProbeKernelFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetEnumeratedParameter("filterType", std::vector<std::string>({ "gaussian" }), filterType, flags);
        node.GetValue("radius", radius, flags);
    }
    
    __host__ Host::LightProbeKernelFilter::LightProbeKernelFilter(const ::Json::Node& jsonNode)
    {
        FromJson(jsonNode, Json::kRequiredWarn);
    }
    
    __host__ AssetHandle<Host::RenderObject> Host::LightProbeKernelFilter::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeKernelFilter(json), id);
    }

    __host__ void Host::LightProbeKernelFilter::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_params.FromJson(node, flags);
    }

    __host__ void Host::LightProbeKernelFilter::OnDestroyAsset()
    {
        m_swapBuffer.DestroyAsset();
    }

    __host__ void Host::LightProbeKernelFilter::Bind(RenderObjectContainer& sceneObjects)
    {
        m_grid = sceneObjects.FindFirstOfType<Host::LightProbeGrid>();

        if (!m_grid)
        {
            Log::Error("Error: LightProbeKernelFilter: no light probe grids were found in this scene.\n");
            return;
        }

        m_gridSHData = m_grid->GetSHDataAsset();
    }

    __host__ void Host::LightProbeKernelFilter::OnPostRenderPass()
    {
        /*if (m_gridSHData->Size() == 0) { return; }

        if (m_swapBuffer->Size() != m_gridSHData->Size())
        {
            m_swapBuffer->Resize(m_gridSHData->Size());
        }*/
    }
}