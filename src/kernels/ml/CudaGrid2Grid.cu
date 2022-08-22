#include "CudaGrid2Grid.cuh"
#include "../cameras/CudaLightProbeCamera.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ Grid2GridParams::Grid2GridParams()
    {

    }

    __host__ void Grid2GridParams::ToJson(::Json::Node& node) const
    {
      
    }

    __host__ uint Grid2GridParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        return kRenderObjectDirtyRender;
    }

    __host__ Host::Grid2Grid::Grid2Grid(const std::string& id, const ::Json::Node& node) :
        RenderObject(id),
        m_gridSize(1), m_blockSize(1)
    {
        FromJson(node, Json::kSilent);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);
        node.GetValue("modelPath", m_modelPath, Json::kRequiredAssert | Json::kNotBlank);

        AssertMsgFmt(!GlobalResourceRegistry::Get().Exists(m_outputGridID), "Error: an asset with ID '%s' already exists'.", m_outputGridID.c_str());

        // Create some objects
        m_hostOutputGrid = CreateChildAsset<Host::LightProbeGrid>(m_outputGridID);

        // Initialise the grid2grid model
        m_onnxEvaluator.Initialise(m_modelPath);

        m_rawData.reserve(8 * 8 * 8 * 5);
    }

    __host__ AssetHandle<Host::RenderObject> Host::Grid2Grid::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::Grid2Grid>(id, json);
    }

    __host__ uint Host::Grid2Grid::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_objects->params.FromJson(node, flags);

        Prepare();

        return kRenderObjectDirtyRender;
    }

    __host__ void Host::Grid2Grid::OnDestroyAsset()
    {
        m_hostOutputGrid.DestroyAsset();
    }

    __host__ void Host::Grid2Grid::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostInputGrid = sceneObjects.FindByID(m_inputGridID).DynamicCast<Host::LightProbeGrid>();
        if (!m_hostInputGrid)
        {
            Log::Error("Error: Grid2Grid::Bind(): the specified input light probe grid '%s' is invalid.\n", m_inputGridID);
            return;
        }

        // Get the light probe camera object and listen out for rebuilds
        auto& probeCamera = sceneObjects.FindFirstOfType<Host::LightProbeCamera>();
        if (probeCamera)
        {
            probeCamera->Listen(*this, "OnBuildGrids", &Host::Grid2Grid::OnBuildInputGrids);
        }

        Prepare();
    }

    __host__ void Host::Grid2Grid::Prepare()
    {
        m_isActive = true;
        m_isValidInput = true;

        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        const auto& gridParams = m_hostInputGrid->GetParams();

        if (gridParams.gridDensity != ivec3(8, 8, 8) || gridParams.shOrder != 1)
        {
            Log::Warning("Warning: Grid2Grid requires 8x8x8 grid of order L1. Input is %s of order L%i.", gridParams.gridDensity.format(), gridParams.shOrder);
            m_isValidInput = false;
            return;
        }

        // Initialise the output grids so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(gridParams);

        m_objects.Upload();
    }

    __host__ void Host::Grid2Grid::OnBuildInputGrids(const RenderObject& originObject, const std::string& eventID)
    {
        // Run the filter every time the input grids are updated
        Execute();
    }

    __host__ void Host::Grid2Grid::Execute()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        const LightProbeGridParams& gridParams = m_hostInputGrid->GetParams();

        // Pass-through filter just copies the data
        //if (m_objects->params.filterType == kKernelFilterNull/* || !m_hostInputGrid->IsConverged()*/)
        if (!m_isValidInput)
        {
            m_hostOutputGrid->Replace(*m_hostInputGrid);           
            m_isActive = true;
            return;
        }

        // Invoke the grid2grid model
        m_hostInputGrid->GetRawData(m_rawData);
        m_onnxEvaluator.Evaluate(m_rawData, m_rawData); 
        m_hostOutputGrid->SetRawData(m_rawData);
        
        m_isActive = false;
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::Grid2Grid::GetChildObjectHandles()
    {    
        return std::vector<AssetHandle<Host::RenderObject>>({ m_hostOutputGrid });
    }

    __host__ void Host::Grid2Grid::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects, const uint dirtyFlags)
    {
        if (m_hostInputGrid && m_hostOutputGrid &&
            m_hostInputGrid->GetParams() != m_hostOutputGrid->GetParams())
        {
            Prepare();
        }
    }
}