#include "CudaLightProbeIO.cuh"
#include "../CudaManagedArray.cuh"

#include "generic/JsonUtils.h"
#include "io/USDIO.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ LightProbeIOParams::LightProbeIOParams() : 
        doExportUSD(false)
    {
    }

    __host__ void LightProbeIOParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("doExportUSD", doExportUSD);
    }

    __host__ void LightProbeIOParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("doExportUSD", doExportUSD, Json::kSilent);
    }

    __host__ Host::LightProbeIO::LightProbeIO(const ::Json::Node& node, const std::string& id)
    {
        FromJson(node, Json::kRequiredWarn);        

        AssertMsgFmt(!GlobalAssetRegistry::Get().Exists(m_inputGridID), "Error: an asset with ID '%s' already exists'.", m_inputGridID.c_str());

        // Create some objects
        m_hostInputGrid = AssetHandle<Host::LightProbeGrid>(m_inputGridID, m_inputGridID);

        ImportProbeGrid();
    }

    __host__ AssetHandle<Host::RenderObject> Host::LightProbeIO::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeIO(json, id), id);
    }

    __host__ void Host::LightProbeIO::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_params.FromJson(node, flags);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);
        node.GetValue("usdImportPath", m_usdImportPath, flags);
        node.GetValue("usdExportPath", m_usdExportPath, flags);
    }

    __host__ void Host::LightProbeIO::OnDestroyAsset()
    {
        m_hostOutputGrid.DestroyAsset();
    }

    __host__ void Host::LightProbeIO::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostOutputGrid = sceneObjects.FindByID(m_outputGridID).DynamicCast<Host::LightProbeGrid>();
        if (!m_hostOutputGrid)
        {
            Log::Error("Error: LightProbeIO::Bind(): the specified output light probe grid '%s' is invalid.\n", m_outputGridID);
            return;
        }        
    }

    __host__ void Host::LightProbeIO::ImportProbeGrid()
    {
        if (m_usdImportPath.empty()) { return; }
        
        std::vector<vec3> rawData;
        LightProbeGridParams gridParams;
        
        try
        {
            USDIO::ReadGridDataUSD(rawData, gridParams, m_usdImportPath, USDIO::SHPackingFormat::kUnity);
        }
        catch (const std::runtime_error& err)
        {
            Log::Error("Failed to import probe grid from '%s'. Assertion failed: %s", m_usdImportPath, err.what());
            return;
        }

        m_hostInputGrid->Prepare(gridParams);
        m_hostInputGrid->SetRawData(rawData);
        m_hostInputGrid->UpdateAggregateStatistics(1);

        // Output summary data about imported probe grid
        {
            const auto& params = m_hostInputGrid->GetParams();
            Log::Indent indent(tfm::format("Successfully imported probe grid from '%s'.", m_usdImportPath));
            Log::Write("Grid dimensions: %s", params.gridDensity.format());
            Log::Write("Aspect ratio: %s", params.aspectRatio.format());
            Log::Write("SH order: %i", params.shOrder);
        }
    }

    __host__ void Host::LightProbeIO::OnPostRenderPass()
    {
       
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::LightProbeIO::GetChildObjectHandles()
    {
        std::vector<AssetHandle<Host::RenderObject>> objects;
        objects.emplace_back(m_hostInputGrid);
        return objects;
    }

    __host__ void Host::LightProbeIO::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects)
    {
       
    }
}