#include "USDIO.h"

#include "generic/D3DIncludes.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"
#include "generic/Log.h"

#include "kernels/CudaLightProbeGrid.cuh"

#ifndef _DEBUG 

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/base/plug/registry.h>

#endif

namespace USDIO
{
#ifndef _DEBUG   
    
    void InitialiseUSD(const Json::Node& usdJson)
    {
        std::string usdPluginDirectory;
        usdJson.GetValue("pluginDirectory", usdPluginDirectory, Json::kSilent);

        // Not specified? Look in the module directory.
        if (usdPluginDirectory.empty()) { usdPluginDirectory = GetModuleDirectory(); }
        
        auto& registry = pxr::PlugRegistry::GetInstance();
        auto plugs = registry.RegisterPlugins(usdPluginDirectory);

        auto extensions = pxr::SdfFileFormat::FindAllFileFormatExtensions();
        AssertMsg(!extensions.empty(), "Unable to initialise USD: could not find any plugins");

        Log::Debug("Found %i USD extensions:\n", extensions.size());
        for (auto& ex : extensions)
        {
            Log::Debug(" - .%s\n", ex);
        }
    }

    template<typename Type>
    void SetUSDAttribute(pxr::UsdPrim& prim, const std::string& id, const Type& data)
    {
        pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken(id));
        if (!attr) { Log::Error("Internal error: USD attribute '%s' not found.\n'", id); }

        attr.Set(data);
    }

    void WriteGridDataUSD(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, std::string usdExportPath)
    {
        // Load the root config
        Json::Document configJson;
        configJson.Deserialise("config.json");

        Json::Node usdJson = configJson.GetChildObject("usd", Json::kRequiredWarn);
        if (!usdJson) { Log::Error("Error\n");  return; }

        std::string usdTemplatePath;
        std::string usdDescription;
        if (!usdJson.GetValue("templatePath", usdTemplatePath, Json::kRequiredWarn)) { return; }

        if(usdExportPath.empty())
        {
            if (!usdJson.GetValue("exportPath", usdExportPath, Json::kRequiredWarn)) { return; }
        }

        usdJson.GetValue("description", usdDescription, Json::kSilent);

        const std::string layerTemplateStr = ReadTextFile(usdTemplatePath);
        Assert(!layerTemplateStr.empty());

        //const std::string& usdExportPath = grid->GetUSDExportPath();
        
        InitialiseUSD(usdJson);       
        
        pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();

        Assert(stage);
        auto root = stage->GetRootLayer();
        Assert(root);
        root->ImportFromString(layerTemplateStr);

        const pxr::SdfPath path("/ProbeVolume");
        pxr::UsdPrim prim = stage->GetPrimAtPath(path);
        Assert(prim);

        const int numSamples = int(gridData[gridParams.coefficientsPerProbe - 1].y + 0.5f);
        
        SetUSDAttribute(prim, "description", usdDescription);
        SetUSDAttribute(prim, "sampleNum", numSamples);
        SetUSDAttribute(prim, "size", pxr::GfVec3f(gridParams.transform.scale().x, gridParams.transform.scale().y, gridParams.transform.scale().z));
        SetUSDAttribute(prim, "resolution", pxr::GfVec3f(gridParams.gridDensity.x, gridParams.gridDensity.y, gridParams.gridDensity.z));

        pxr::VtFloatArray coeffs(gridParams.numProbes * (gridParams.coefficientsPerProbe - 1) * 3);
        pxr::VtFloatArray dataValidity(gridParams.numProbes);

        for (int probeIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx)
        {            
            // Set the SH coefficients
            for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe - 1; ++coeffIdx)
            {
                const int destIdx = probeIdx * (gridParams.coefficientsPerProbe - 1) + coeffIdx;
                const int sourceIdx = probeIdx * gridParams.coefficientsPerProbe + coeffIdx;
                coeffs[destIdx * 3] = gridData[sourceIdx].x;
                coeffs[destIdx * 3 + 1] = gridData[sourceIdx].y;
                coeffs[destIdx * 3 + 2] = gridData[sourceIdx].z;
            }

            // Set the validity coefficients
            const int sourceIdx = probeIdx * gridParams.coefficientsPerProbe + (gridParams.coefficientsPerProbe - 1);
            dataValidity[probeIdx] = gridData[sourceIdx].x;
        }

        SetUSDAttribute(prim, "coefficients", coeffs);
        SetUSDAttribute(prim, "dataValidity", dataValidity);

        stage->SetDefaultPrim(prim);
        stage->Export(usdExportPath);

        Log::Write("Exported USD file to '%s'\n", usdExportPath);
    }

#else 

    #define USD_DISABLED_FUNCTION(func) func { Log::Debug("***** Warning: USD exporting is disabled in debug mode. ****\n"); } 

    USD_DISABLED_FUNCTION(void WriteGridDataUSD(const std::vector<Cuda::vec3>&, const Cuda::LightProbeGridParams&, std::string usdExportPath))
    USD_DISABLED_FUNCTION(void TestUSD())   

#endif

    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& gridAsset, const std::string& usdExportPath)
    {
        Assert(gridAsset);         
        
        std::vector<Cuda::vec3> rawData;
        gridAsset->GetRawData(rawData);
        const Cuda::LightProbeGridParams& gridParams = gridAsset->GetParams();

        int dataSize = gridParams.numProbes * gridParams.coefficientsPerProbe;
        Assert(dataSize > 0);
        std::vector<Cuda::vec3> swizzledData;
        swizzledData.resize(dataSize);

        auto SwizzleIndex = [&](const Cuda::ivec3& idx) -> Cuda::ivec3
        {
            // Swizzle the axes
            switch (gridParams.axisSwizzle)
            {
            case Cuda::kXZY: return idx.xzy; 
            case Cuda::kYXZ: return idx.yxz; 
            case Cuda::kYZX: return idx.yzx; 
            case Cuda::kZXY: return idx.zxy; 
            case Cuda::kZYX: return idx.zyx; 
            }
            return idx;
        };        

        const Cuda::ivec3 swizzledGridDensity = SwizzleIndex(gridParams.gridDensity);

        // Step through each probe in XYZ space and remap it to the swizzled space specified by the grid parameters
        for (int probeIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx)
        {
            Cuda::ivec3 gridIdx = Cuda::GridIdxFromProbeIdx(probeIdx, gridParams.gridDensity);            

            // Invert the axes where appropriate
            if (gridParams.invertX) { gridIdx.x = gridParams.gridDensity.x - gridIdx.x - 1; }
            if (gridParams.invertY) { gridIdx.y = gridParams.gridDensity.y - gridIdx.y - 1; }
            if (gridParams.invertZ) { gridIdx.z = gridParams.gridDensity.z - gridIdx.z - 1; }
            
            // Swizzle the grid index
            Cuda::ivec3 swizzledGridIdx = SwizzleIndex(gridIdx);

            // Map back onto the data array
            const uint swizzledProbeIdx = ProbeIdxFromGridIdx(swizzledGridIdx, swizzledGridDensity);
            Assert(swizzledProbeIdx < gridParams.numProbes);

            // Copy the coefficient data 
            for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe; ++coeffIdx)
            {
                swizzledData[swizzledProbeIdx * gridParams.coefficientsPerProbe + coeffIdx] = rawData[probeIdx * gridParams.coefficientsPerProbe + coeffIdx];
            }
        }

        WriteGridDataUSD(swizzledData, gridParams, usdExportPath);

        /*Log::Debug("%i elements\n", rawData.size());
        for (int i = 0; i < gridParams.numProbes; i++)
        {
            Log::Debug("  - %i\n", i);
            for (int j = 0; j < gridParams.coefficientsPerProbe; j++)
            {
                const auto& raw = rawData[i * gridParams.coefficientsPerProbe + j];
                const auto& swizzled = swizzledData[i * gridParams.coefficientsPerProbe + j];
                Log::Debug("    - %i: %s -> %i\n", j, raw.format(), swizzled.format());
            }
        }*/
    }
}
