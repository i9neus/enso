#include "CudaFCNNProbeDenoiser.cuh"
#include "../cameras/CudaLightProbeCamera.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ FCNNProbeDenoiserParams::FCNNProbeDenoiserParams()
    {

    }

    __host__ void FCNNProbeDenoiserParams::ToJson(::Json::Node& node) const
    {

    }

    __host__ void FCNNProbeDenoiserParams::FromJson(const ::Json::Node& node, const uint flags)
    {

    }

    __host__ Host::FCNNProbeDenoiser::FCNNProbeDenoiser(const std::string& id, const ::Json::Node& node) :
        RenderObject(id),
        m_gridSize(1), m_blockSize(1)
    {
        FromJson(node, Json::kRequiredWarn);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);
        node.GetValue("modelRootPath", m_onnxParams.modelRootPath, Json::kRequiredAssert | Json::kNotBlank);
        node.GetValue("modelProprocessPath", m_onnxParams.modelPreprocessPath, Json::kRequiredAssert | Json::kNotBlank);
        node.GetValue("modelPostprocessPath", m_onnxParams.modelPostprocessPath, Json::kRequiredAssert | Json::kNotBlank);
        node.GetValue("modelDenoiserPath", m_onnxParams.modelDenoiserPath, Json::kRequiredAssert | Json::kNotBlank);

        AssertMsgFmt(!GlobalResourceRegistry::Get().Exists(m_outputGridID), "Error: an asset with ID '%s' already exists'.", m_outputGridID.c_str());

        // Create some objects
        m_hostOutputGrid = CreateChildAsset<Host::LightProbeGrid>(m_outputGridID, this);

        // Initialise the grid2grid model
        m_onnxEvaluator.Initialise(m_onnxParams);
    }
    __host__ AssetHandle<Host::RenderObject> Host::FCNNProbeDenoiser::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::FCNNProbeDenoiser>(id, json);
    }

    __host__ void Host::FCNNProbeDenoiser::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_params.FromJson(node, flags);

        Prepare();
    }

    __host__ void Host::FCNNProbeDenoiser::OnDestroyAsset()
    {
        m_hostOutputGrid.DestroyAsset();
    }

    __host__ void Host::FCNNProbeDenoiser::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostInputGrid = sceneObjects.FindByID(m_inputGridID).DynamicCast<Host::LightProbeGrid>();
        if (!m_hostInputGrid)
        {
            Log::Error("Error: FCNNProbeDenoiser::Bind(): the specified input light probe grid '%s' is invalid.\n", m_inputGridID);
            return;
        }

        // Get the light probe camera object and listen out for rebuilds
        auto& probeCamera = sceneObjects.FindFirstOfType<Host::LightProbeCamera>();
        if (probeCamera)
        {
            probeCamera->Listen(*this, "OnBuildGrids", &Host::FCNNProbeDenoiser::OnBuildInputGrids);
        }

        Prepare();
    }

    __host__ void Host::FCNNProbeDenoiser::Prepare()
    {
        m_isActive = true;
        m_isValidInput = true;

        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        const auto& gridParams = m_hostInputGrid->GetParams();

        if (cwiseMin(gridParams.gridDensity) < 16 || gridParams.shOrder != 1)
        {
            Log::Warning("Warning: FCNNProbeDenoiser requires grid of at least 16x16x16 or order L1. Input is %s of order L%i.", gridParams.gridDensity.format(), gridParams.shOrder);
            m_isValidInput = false;
            return;
        }
        
        m_rawData.reserve(Volume(gridParams.gridDensity) * 5);

        // Initialise the output grids so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(gridParams);
    }

    __host__ void Host::FCNNProbeDenoiser::OnBuildInputGrids(const RenderObject& originObject, const std::string& eventID)
    {
        // Run the filter every time the input grids are updated
        Execute();
    }

    __host__ void Host::FCNNProbeDenoiser::Execute()
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

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::FCNNProbeDenoiser::GetChildObjectHandles()
    {
        return std::vector<AssetHandle<Host::RenderObject>>({ m_hostOutputGrid });
    }

    __host__ void Host::FCNNProbeDenoiser::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects)
    {
        if (m_hostInputGrid && m_hostOutputGrid &&
            m_hostInputGrid->GetParams() != m_hostOutputGrid->GetParams())
        {
            Prepare();
        }
    }
}