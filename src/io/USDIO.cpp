#include "USDIO.h"

#include <filesystem>

#include "generic/D3DIncludes.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"
#include "generic/Log.h"
#include "generic/GlobalStateAuthority.h"

#include "kernels/lightprobes/CudaLightProbeGrid.cuh"
#include "kernels/math/CudaSphericalHarmonics.cuh"

#ifndef _DEBUG 

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/base/plug/registry.h>

#endif

namespace USDIO
{
    template<typename ContainerType>
    void PackCoefficients(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, const SHPackingFormat shFormat, ContainerType& coeffs, ContainerType& dataValidity)
    {
        // Precompute some coefficients
        Cuda::vec3 C[5], D[5];

        for (int probeIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx)
        {
            int destIdx = probeIdx * (gridParams.coefficientsPerProbe - 1);
            int sourceIdx = probeIdx * gridParams.coefficientsPerProbe;

            if (shFormat == SHPackingFormat::kUnity)
            {
                for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe; ++coeffIdx) { C[coeffIdx] = gridData[sourceIdx + coeffIdx]; }

                // Pre-multiply the coefficients
                C[0] *= Cuda::SH::Legendre(0);
                C[1] *= Cuda::SH::Legendre(1);
                C[2] *= Cuda::SH::Legendre(1);
                C[3] *= Cuda::SH::Legendre(1);

                // Pack the coefficients using Unity's perferred format
                D[0] = C[0];
                D[1] = Cuda::vec3(C[1].x, C[2].x, C[3].x);
                D[2] = Cuda::vec3(C[1].y, C[2].y, C[3].y);
                D[3] = Cuda::vec3(C[1].z, C[2].z, C[3].z);
                D[4] = C[4];

                // Set the SH coefficients
                for (int coeffIdx = 0; coeffIdx < 4; ++coeffIdx)
                {
                    coeffs[(destIdx + coeffIdx) * 3] = D[coeffIdx].x;
                    coeffs[(destIdx + coeffIdx) * 3 + 1] = D[coeffIdx].y;
                    coeffs[(destIdx + coeffIdx) * 3 + 2] = D[coeffIdx].z;
                }

                // Data validity is inverted in Unity's format
                dataValidity[probeIdx] = 1.0f - gridData[sourceIdx + gridParams.coefficientsPerProbe - 1].x;
            }
            else
            {
                // Set the SH coefficients
                for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe - 1; ++coeffIdx)
                {
                    coeffs[(destIdx + coeffIdx) * 3] = gridData[sourceIdx + coeffIdx].x;
                    coeffs[(destIdx + coeffIdx) * 3 + 1] = gridData[sourceIdx + coeffIdx].y;
                    coeffs[(destIdx + coeffIdx) * 3 + 2] = gridData[sourceIdx + coeffIdx].z;
                }

                dataValidity[probeIdx] = gridData[sourceIdx + gridParams.coefficientsPerProbe - 1].x;
            }
        }
    }

    template<typename Type>
    void AddXYZVector(Json::Node& rootNode, const std::string& name, const Cuda::Tvec3<Type>& value)
    {
        Json::Node sizeNode = rootNode.AddChildObject(name);
        sizeNode.AddValue("x", value.x);
        sizeNode.AddValue("y", value.y);
        sizeNode.AddValue("z", value.z);
    }

    void WriteGridDataJSON(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, std::string jsonExportPath, const SHPackingFormat shFormat)
    {
        Json::Document rootNode;

        // Emplace the headers
        rootNode.AddValue("description", "");
        rootNode.AddValue("sampleNum", gridParams.minMaxSamplesPerProbe.y);
        AddXYZVector(rootNode, "size", gridParams.transform.scale());
        AddXYZVector(rootNode, "resolution", gridParams.gridDensity);

        // Pack the grid data into the containers and add to the document        
        std::vector<float> coeffs(gridParams.numProbes * (gridParams.coefficientsPerProbe - 1) * 3);
        std::vector<float> dataValidity(gridParams.numProbes);
        PackCoefficients(gridData, gridParams, shFormat, coeffs, dataValidity);
        rootNode.AddArray("coefficients", coeffs);
        rootNode.AddArray("dataValidity", dataValidity);

        // Serialise to file
        rootNode.Serialise(jsonExportPath);

        Log::Write("Exported light probe grid to JSON '%s'\n", jsonExportPath);
    }

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

        /*Log::Debug("Found %i USD extensions:\n", extensions.size());
        for (auto& ex : extensions)
        {
            Log::Debug(" - .%s\n", ex);
        }*/
    }

    template<typename Type>
    bool SetUSDAttribute(pxr::UsdPrim& prim, const std::string& id, const Type& data)
    {
        pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken(id));
        if (!attr) 
        { 
            Log::Error("Internal error: USD attribute '%s' not found.\n'", id); 
            return false;
        }

        attr.Set(data);
        return true;
    }

    template<typename Type>
    bool GetUSDAttribute(const pxr::UsdPrim& prim, const std::string& id, Type& data)
    {
        pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken(id));
        if (attr)
        {
            attr.Get(&data);
            return true;
        }
        return false;
    }

    void ReadGridDataUSD(std::vector<Cuda::vec3>& gridData, Cuda::LightProbeGridParams& gridParams, const std::string usdImportPath, const SHPackingFormat shFormat)
    {
        // Load the root config
        const Json::Document& configJson = GSA().GetConfigJson();

        const Json::Node usdJson = configJson.GetChildObject("usd", Json::kRequiredWarn);
        if (!usdJson) { Log::Error("Error\n");  return; }
        
        InitialiseUSD(usdJson);

        pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(usdImportPath);
        Assert(stage);

        pxr::UsdPrim prim = stage->GetDefaultPrim();
        Assert(prim);

        pxr::GfVec3f gridSize;
        pxr::GfVec3f gridResolution;
        std::string usdDescription;
        pxr::VtFloatArray coeffs;
        pxr::VtFloatArray dataValidity;
        pxr::VtFloatArray dataMeanDistance;

        Assert(GetUSDAttribute(prim, "description", usdDescription));
        Assert(GetUSDAttribute(prim, "sampleNum", gridParams.minMaxSamplesPerProbe.y));
        Assert(GetUSDAttribute(prim, "size", gridSize));
        Assert(GetUSDAttribute(prim, "resolution", gridResolution));
        Assert(GetUSDAttribute(prim, "coefficients", coeffs));
        const bool hasMeanDistance = GetUSDAttribute(prim, "dataMeanDistance", dataMeanDistance);
        const bool hasValidity = GetUSDAttribute(prim, "dataValidity", dataValidity);

        gridParams.transform.scale = Cuda::vec3(gridSize[0], gridSize[1], gridSize[2]);
        gridParams.gridDensity = Cuda::ivec3(gridResolution[0], gridResolution[1], gridResolution[2]);
        gridParams.numProbes = Cuda::Volume(gridParams.gridDensity);
        gridParams.coefficientsPerProbe = (coeffs.size() / (gridParams.numProbes * 3)) + 1;

        AssertMsg(shFormat != SHPackingFormat::kUnity || gridParams.coefficientsPerProbe == 5, "Unity SH packing format expects L1 probes.");
        
        Assert(coeffs.size() % (gridParams.coefficientsPerProbe - 1) == 0);
        Assert(!hasValidity || dataValidity.size() == gridParams.numProbes);
        Assert(!hasMeanDistance || dataMeanDistance.size() == gridParams.numProbes);

        /*Log::Debug("Loading light probe grid from USD...");
        Log::Debug("  - Description: %s", usdDescription);
        Log::Debug("  - Resolution: %s", gridParams.gridDensity.format());
        Log::Debug("  - Size: %s", gridParams.transform.scale().format());
        Log::Debug("  - Coefficients per probe: %i", gridParams.coefficientsPerProbe);*/

        gridData.resize(gridParams.numProbes * gridParams.coefficientsPerProbe);

        // Precompute some coefficients
        const float kRootPi = std::sqrt(kPi);
        const float fC0 = 1.0f / (2.0f * kRootPi);
        const float fC1 = std::sqrt(3.0f) / (3.0f * kRootPi);
        Cuda::vec3 C[5], D[5];

        for (int probeIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx)
        {
            // Set the SH coefficients
            int destIdx = probeIdx * gridParams.coefficientsPerProbe;
            int sourceIdx = 3 * probeIdx * (gridParams.coefficientsPerProbe - 1);

            if (shFormat == SHPackingFormat::kUnity)
            {
                for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe - 1; ++coeffIdx, sourceIdx += 3)
                {
                    D[coeffIdx] = Cuda::vec3(coeffs[sourceIdx], coeffs[sourceIdx + 1], coeffs[sourceIdx + 2]);
                }

                // Unpack the coefficients
                C[0] = D[0];
                C[1] = Cuda::vec3(D[1].x, D[2].x, D[3].x);
                C[2] = Cuda::vec3(D[1].y, D[2].y, D[3].y);
                C[3] = Cuda::vec3(D[1].z, D[2].z, D[3].z);
                C[4] = D[4];

                // Remove the premultiplication
                C[0] /= fC0;
                C[1] /= -fC1;
                C[2] /= fC1;
                C[3] /= -fC1;

                for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe - 1; ++coeffIdx, ++destIdx)
                {
                    gridData[destIdx] = C[coeffIdx];
                }

                gridData[destIdx].x = hasValidity ? (1.0f - dataValidity[probeIdx]) : 1.0f;
            }
            else
            {
                for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe - 1; ++coeffIdx, ++destIdx, sourceIdx += 3)
                {
                    gridData[destIdx] = Cuda::vec3(coeffs[sourceIdx], coeffs[sourceIdx + 1], coeffs[sourceIdx + 2]);
                }

                gridData[destIdx].x = hasValidity ? dataValidity[probeIdx] : 1.0f;
            }

            // Set the validity and mean distance coefficients
            gridData[destIdx].y = hasMeanDistance ? dataMeanDistance[probeIdx] : 0.0f;
            gridData[destIdx].z = 1.0f;
            //Log::Debug("[%i, %i]: %s", probeIdx, gridParams.coefficientsPerProbe - 1, gridData[destIdx].format());
        }

        Log::Write("Imported USD file from '%s'\n", usdImportPath);
    }    

    void WriteGridDataUSD(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, std::string usdExportPath, const SHPackingFormat shFormat)
    {
        Assert(!usdExportPath.empty());  
        AssertMsg(shFormat != SHPackingFormat::kUnity || gridParams.coefficientsPerProbe == 5, "Unity packed SH format must be L1 probes.");
        
        // Load the root config
        Json::Document configJson;
        configJson.Deserialise("config.json");

        Json::Node usdJson = configJson.GetChildObject("usd", Json::kRequiredWarn);
        if (!usdJson) { Log::Error("Error\n");  return; }

        std::string usdTemplatePath;
        std::string usdDescription;
        if (!usdJson.GetValue("templatePath", usdTemplatePath, Json::kRequiredWarn)) { return; }
        
        //if(usdExportPath.empty())
        //{
        //    if (!usdJson.GetValue("exportPath", usdExportPath, Json::kRequiredWarn)) { return; }
        //}

        usdJson.GetValue("description", usdDescription, Json::kSilent);

        const std::string layerTemplateStr = ReadTextFile(usdTemplatePath);
        Assert(!layerTemplateStr.empty());
        
        InitialiseUSD(usdJson);       
        
        pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();

        Assert(stage);
        auto root = stage->GetRootLayer();
        Assert(root);
        root->ImportFromString(layerTemplateStr);

        const pxr::SdfPath path("/ProbeVolume");
        pxr::UsdPrim prim = stage->GetPrimAtPath(path);
        Assert(prim);
        
        SetUSDAttribute(prim, "description", usdDescription);
        SetUSDAttribute(prim, "sampleNum", gridParams.minMaxSamplesPerProbe.y);
        SetUSDAttribute(prim, "size", pxr::GfVec3f(gridParams.transform.scale().x, gridParams.transform.scale().y, gridParams.transform.scale().z));
        SetUSDAttribute(prim, "resolution", pxr::GfVec3f(gridParams.gridDensity.x, gridParams.gridDensity.y, gridParams.gridDensity.z));

        pxr::VtFloatArray coeffs(gridParams.numProbes * (gridParams.coefficientsPerProbe - 1) * 3);
        pxr::VtFloatArray dataValidity(gridParams.numProbes);
        pxr::VtFloatArray dataMeanDistance(gridParams.numProbes);

        // Pack the grid data into the USD containers
        PackCoefficients(gridData, gridParams, shFormat, coeffs, dataValidity);       

        SetUSDAttribute(prim, "coefficients", coeffs);
        SetUSDAttribute(prim, "dataMeanDistance", dataMeanDistance);
        SetUSDAttribute(prim, "dataValidity", dataValidity);

        stage->SetDefaultPrim(prim);
        stage->Export(usdExportPath);        

        Log::Write("Exported light probe grid to USD '%s'\n", usdExportPath);
    }

#else 

    #define USD_DISABLED_FUNCTION(func) func { Log::Error("***** Warning: USD input/output is disabled in debug mode. ****\n"); } 

    USD_DISABLED_FUNCTION(void WriteGridDataUSD(const std::vector<Cuda::vec3>&, const Cuda::LightProbeGridParams&, std::string usdExportPath, const SHPackingFormat shFormat))
    USD_DISABLED_FUNCTION(void TestUSD())   
    USD_DISABLED_FUNCTION(void ReadGridDataUSD(std::vector<Cuda::vec3>& gridData, Cuda::LightProbeGridParams& gridParams, const std::string usdImportPath, const SHPackingFormat shFormat))

#endif  

    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& gridAsset, Cuda::LightProbeGridParams& exportParams, const std::string& exportPath, const SHPackingFormat shFormat)
    {
        Assert(gridAsset);         
        
        const auto& dataTransform = exportParams.dataTransform;
        std::vector<Cuda::vec3> rawData;
        gridAsset->GetRawData(rawData);
        const Cuda::LightProbeGridParams& gridParams = gridAsset->GetParams();

        int dataSize = gridParams.numProbes * gridParams.coefficientsPerProbe;
        Assert(dataSize > 0);
        std::vector<Cuda::vec3> swizzledData;
        swizzledData.resize(dataSize);

        auto SwizzleIndices = [&](const int type) -> Cuda::ivec3
        {
            // Swizzle the axes
            switch (type)
            {
            case Cuda::kXZY: return Cuda::ivec3(0, 2, 1);
            case Cuda::kYXZ: return Cuda::ivec3(1, 0, 2);
            case Cuda::kYZX: return Cuda::ivec3(1, 2, 0);
            case Cuda::kZXY: return Cuda::ivec3(2, 0, 1);
            case Cuda::kZYX: return Cuda::ivec3(2, 1, 0);
            }
            return Cuda::ivec3(0, 1, 2);
        };

        auto SwizzleVector = [](const Cuda::ivec3& v, const Cuda::ivec3& i) { return Cuda::ivec3(v[i[0]], v[i[1]], v[i[2]]);  };       

        // Generate swizzle indices for probe positions
        const Cuda::ivec3 posSwiz = SwizzleIndices(dataTransform.posSwizzle);        

        // Swizzle the grid density
        const Cuda::ivec3 swizzledGridDensity = SwizzleVector(gridParams.gridDensity, posSwiz);

        // Factors for inverting L1 SH coefficients
        std::vector<float> shDirs(gridParams.coefficientsPerProbe, 1.0f);        
        if (dataTransform.shInvertX) { shDirs[1] = -1.0f; }
        if (dataTransform.shInvertY) { shDirs[2] = -1.0f; }
        if (dataTransform.shInvertZ) { shDirs[3] = -1.0f; }

        // Generate swizzle indices for the coefficients
        std::vector<int> shSwiz(gridParams.coefficientsPerProbe);
        for (int i = 0; i < shSwiz.size(); ++i) { shSwiz[i] = i; }
        shSwiz[1] = SwizzleIndices(dataTransform.shSwizzle)[0] + 1;
        shSwiz[2] = SwizzleIndices(dataTransform.shSwizzle)[1] + 1;
        shSwiz[3] = SwizzleIndices(dataTransform.shSwizzle)[2] + 1;

        // Swizzle the axes
        Cuda::vec2 coeffMinMax = Cuda::kMinMaxReset;
        for (int probeIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx)
        {
            Cuda::ivec3 gridPos = Cuda::GridPosFromProbeIdx(probeIdx, gridParams.gridDensity);

            // Invert the axes where appropriate
            if (dataTransform.posInvertX) { gridPos.x = gridParams.gridDensity.x - gridPos.x - 1; }
            if (dataTransform.posInvertY) { gridPos.y = gridParams.gridDensity.y - gridPos.y - 1; }
            if (dataTransform.posInvertZ) { gridPos.z = gridParams.gridDensity.z - gridPos.z - 1; }
            
            // Swizzle the grid index
            Cuda::ivec3 swizzledGridPos = SwizzleVector(gridPos, posSwiz);

            // Map back onto the data array
            const uint swizzledProbeIdx = ProbeIdxFromGridPos(swizzledGridPos, swizzledGridDensity);
            Assert(swizzledProbeIdx < gridParams.numProbes);

            // Copy the coefficient data and multiply by the orientation coefficients
            for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe; ++coeffIdx)
            {
                const auto sh = rawData[probeIdx * gridParams.coefficientsPerProbe + shSwiz[coeffIdx]] * shDirs[coeffIdx];

                if (coeffIdx < gridParams.shCoefficientsPerProbe)
                {
                    coeffMinMax = Cuda::MinMax(coeffMinMax, sh[0]);
                    coeffMinMax = Cuda::MinMax(coeffMinMax, sh[1]);
                    coeffMinMax = Cuda::MinMax(coeffMinMax, sh[2]);
                }
                
                swizzledData[swizzledProbeIdx * gridParams.coefficientsPerProbe + coeffIdx] = sh;
            }
        }

        // If the grid is empty i.e. has zero energy, signal a warning
        if (coeffMinMax[1] - coeffMinMax[0] < 1e-10f) { Log::Error("Warning: grid contains constant value %f", coeffMinMax[1] - coeffMinMax[0]); }
        else { Log::Write("Export USD: coefficient range [%f, %f]", coeffMinMax[0], coeffMinMax[1]); }

        const std::string ext = GetExtension(exportPath);

        if (ext == ".usd" || ext == ".usda")
        {
            WriteGridDataUSD(swizzledData, gridParams, exportPath, shFormat);
        }
        else if (ext == ".json")
        {
            WriteGridDataJSON(swizzledData, gridParams, exportPath, shFormat);
        }
        else
        {
            Log::Error("Error: cannot export light probe grid: unrecognised file extension '%s'", ext);
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
