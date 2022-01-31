#include "CudaLightProbeIO.cuh"
#include "../CudaManagedArray.cuh"

#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"
#include "io/USDIO.h"

#include <filesystem>

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ LightProbeIOParams::LightProbeIOParams() : 
        doCommand(kNull),
        exportUSD(true)
    {
    }

    __host__ void LightProbeIOParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("doCommand", doCommand);
        node.AddValue("exportUSD", exportUSD);
    }

    __host__ void LightProbeIOParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("doCommand", doCommand, Json::kSilent);
        node.GetValue("exportUSD", exportUSD, flags);
    }

    __host__ Host::LightProbeIO::LightProbeIO(const std::string& id, const ::Json::Node& node) :
        RenderObject(id),
        m_isBatchActive(false),
        m_currentIOPaths(m_usdIOList.end())
    {
        FromJson(node, Json::kRequiredWarn);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredWarn);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredWarn);
        node.GetValue("usdImportPath", m_usdImportPath, Json::kRequiredWarn);
        node.GetValue("usdExportPath", m_usdExportPath, Json::kRequiredWarn);
        node.GetValue("usdImportDirectory", m_usdImportDirectory, Json::kRequiredWarn);
        node.GetValue("usdExportDirectory", m_usdExportDirectory, Json::kRequiredWarn);

        AssertMsg(!m_inputGridID.empty() && !m_outputGridID.empty(), "Must specify an input and output light probe grid ID.");

        AssertMsgFmt(!GlobalResourceRegistry::Get().Exists(m_inputGridID), "Error: an asset with ID '%s' already exists'.", m_inputGridID.c_str());

        // Create some objects
        m_hostInputGrid = CreateAsset<Host::LightProbeGrid>(m_inputGridID);

        EnumerateProbeGrids();
        ImportProbeGrid(m_usdImportPath);
    }

    __host__ AssetHandle<Host::RenderObject> Host::LightProbeIO::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::LightProbeIO>(id, json);
    }

    __host__ void Host::LightProbeIO::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_params.FromJson(node, flags);

        switch (m_params.doCommand)
        {
        case LightProbeIOParams::kDoBatch:
            BeginBatchFilter();
            break;
        case LightProbeIOParams::kDoSave:
            ExportProbeGrid(m_usdExportPath);
            break;
        case LightProbeIOParams::kDoNext:
            if (!m_isBatchActive)
            {
                AdvanceNextUSD(true, 1);
            }
            break;
        case LightProbeIOParams::kDoPrevious:
            if (!m_isBatchActive)
            {
                AdvanceNextUSD(true, -1);
            }
            break;
        }
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

    __host__ bool Host::LightProbeIO::AdvanceNextUSD(bool advance, const int direction)
    {
        Assert(direction != 0);
        if ((m_currentIOPaths == m_usdIOList.end() && direction > 0) || 
            (m_currentIOPaths == m_usdIOList.begin() && direction < 0)) { return false; }

        while (true)
        {
            if (advance) { std::advance(m_currentIOPaths, direction); }

            if (m_currentIOPaths == m_usdIOList.end()) { return false; }

            if (ImportProbeGrid(m_currentIOPaths->first)) { break; }

            advance = true;
        }

        //m_hostInputGrid->SetSemaphore("tag_do_filter", true);
        return true;
    }

    __host__ void Host::LightProbeIO::EnumerateProbeGrids()
    {
        Log::Debug("Looking in %s...", m_usdImportDirectory);

        std::vector<std::string> inputFiles;
        EnumerateDirectoryFiles(m_usdImportDirectory, ".usd", inputFiles);

        m_usdExportDirectory = DeslashifyPath(m_usdExportDirectory);

        Log::Debug("Found %i USD files", inputFiles.size());
        for (const auto& inputPath : inputFiles)
        {
            const auto slashIdx = inputPath.rfind('/');
            if (slashIdx == std::string::npos) { continue; }

            std::string outputPath = m_usdExportDirectory + inputPath.substr(slashIdx, inputPath.size() - slashIdx);

            m_usdIOList.emplace_back(inputPath, outputPath);
            //Log::Debug("%s -> %s", inputPath, outputPath);
        }

        m_currentIOPaths = m_usdIOList.begin();
    }

    __host__ void Host::LightProbeIO::BeginBatchFilter()
    {
        if (m_usdIOList.empty())
        {
            Log::Error("Error: no USD files were found in '%s'", m_usdImportDirectory);
            return;
        }

        m_currentIOPaths = m_usdIOList.begin();

        if (!AdvanceNextUSD(false, 1))
        {
            Log::Error("Error: could not load a valid USD file");
            return;
        }
        
        m_isBatchActive = true;
    }

    __host__ void Host::LightProbeIO::ExportProbeGrid(const std::string& filePath) const
    {
        Assert(!m_usdImportPath.empty());

        const std::string extension = GetExtension(filePath);

        // Pull the raw data from the light probe grid object
        std::vector<vec3> rawData;
        const LightProbeGridParams gridParams = m_hostOutputGrid->GetParams();
        m_hostOutputGrid->GetRawData(rawData);

        try
        {
            USDIO::WriteGridDataUSD(rawData, gridParams, filePath, USDIO::SHPackingFormat::kUnity);
        }
        catch (const std::runtime_error& err)
        {
            Log::Error("Failed to export probe grid from '%s'. Assertion failed: %s", filePath, err.what());
        }
    }

    __host__ bool Host::LightProbeIO::ImportProbeGrid(const std::string& filePath)
    {
        Assert(!m_usdImportPath.empty());

        std::vector<vec3> rawData;
        LightProbeGridParams gridParams;

        try
        {
            USDIO::ReadGridDataUSD(rawData, gridParams, filePath, USDIO::SHPackingFormat::kUnity);
        }
        catch (const std::runtime_error& err)
        {
            Log::Error("Failed to import probe grid from '%s'. Assertion failed: %s", filePath, err.what());
            return false;
        }

        gridParams.clipRegion[0] = ivec3(0);
        gridParams.clipRegion[1] = gridParams.gridDensity;
        
        m_hostInputGrid->Prepare(gridParams);
        m_hostInputGrid->SetRawData(rawData);
        m_hostInputGrid->UpdateAggregateStatistics(1);
        m_hostInputGrid->SetSemaphore("tag_do_filter", true);

        // Output summary data about imported probe grid
        {
            const auto& params = m_hostInputGrid->GetParams();
            Log::Indent indent(tfm::format("Successfully imported probe grid from '%s'.", filePath));
            Log::Write("Grid dimensions: %s", params.gridDensity.format());
            Log::Write("Aspect ratio: %s", params.aspectRatio.format());
            Log::Write("SH order: %i", params.shOrder);
        }

        return true;
    }

    __host__ void Host::LightProbeIO::OnPostRenderPass()
    {
        if (!m_isBatchActive || m_hostOutputGrid->GetSemaphore("tag_is_filtered") != true) { return; }

        // Export the output grid to USD
        if (m_params.exportUSD)
        {
            Assert(m_currentIOPaths != m_usdIOList.end());
            ExportProbeGrid(m_currentIOPaths->second);
        }

        // Advance to the next USD file in the list
        if (!AdvanceNextUSD(true, 1))
        {
            m_isBatchActive = false;
        }
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