#include "USDIO.h"

#include "generic/D3DIncludes.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"
#include "generic/Log.h"

#include "kernels/CudaLightProbeGrid.cuh"

#ifndef _DEBUG 

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
        
        auto& registry = PlugRegistry::GetInstance();
        auto plugs = registry.RegisterPlugins(usdPluginDirectory);

        auto extensions = SdfFileFormat::FindAllFileFormatExtensions();
        AssertMsg(!extensions.empty(), "Unable to initialise USD: could not find any plugins");

        Log::Debug("Found %i USD extensions:\n", extensions.size());
        for (auto& ex : extensions)
        {
            Log::Debug(" - .%s\n", ex);
        }
    }   

    void WriteGridDataUSD(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid)
    {
        // Load the root config
        Json::Document configJson;
        configJson.Load("config.json");

        Json::Node usdJson = configJson.GetChildObject("usd", Json::kRequiredAssert);

        std::string usdTemplatePath;
        std::string usdExportDirectory;
        usdJson.GetValue("templatePath", usdTemplatePath, Json::kRequiredAssert);
        usdJson.GetValue("exportDirectory", usdExportDirectory, Json::kRequiredAssert);

        const std::string layerTemplateStr = LoadTextFile(usdTemplatePath);
        Assert(!layerTemplateStr.empty());

        const std::string usdExportPath = SlashifyPath(usdExportDirectory) + "test.usda";
        
        InitialiseUSD(usdJson);       
        
        UsdStageRefPtr stage = UsdStage::CreateInMemory();

        Assert(stage);
        auto root = stage->GetRootLayer();
        Assert(root);
        root->ImportFromString(layerTemplateStr);

        const SdfPath path("/ProbeVolume");
        UsdPrim prim = stage->GetPrimAtPath(path);  
        Assert(prim);
        
        UsdAttribute attr = prim.GetAttribute(TfToken("description"));
        attr.Set("hello");

        stage->SetDefaultPrim(prim);
        stage->Export(usdExportPath);
    }

    void TestUSD()
    {
        ExportLightProbeGrid(Cuda::AssetHandle<Cuda::Host::LightProbeGrid>());
    }

#else 

    #define USD_DISABLED_FUNCTION(func) func { Log::Debug("***** Warning: USD exporting is disabled in debug mode. ****\n"); } 

    USD_DISABLED_FUNCTION(void WriteGridDataUSD(const std::vector<Cuda::vec3>& rawData))
    USD_DISABLED_FUNCTION(void TestUSD())   

#endif

    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& gridAsset)
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
