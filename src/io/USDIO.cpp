#include "USDIO.h"

#include "generic/D3DIncludes.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"
#include "generic/Log.h"

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
    
    using namespace pxr;

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

    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid)
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

    USD_DISABLED_FUNCTION(void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid))
    USD_DISABLED_FUNCTION(void TestUSD())
   

#endif
}
