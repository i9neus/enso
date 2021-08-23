#pragma once

#include "CudaRenderObject.cuh"

namespace Cuda
{
    namespace Host { class LightProbeGrid; }
    struct HitCtx;

    enum AxisSwizzle : int { kXYZ, kXZY, kYXZ, kYZX, kZXY, kZYX };

    __host__ __device__ inline ivec3 GridIdxFromProbeIdx(const int& probeIdx, const ivec3& gridDensity)
    {
        return ivec3(probeIdx % gridDensity.x, 
                     (probeIdx / gridDensity.x) % gridDensity.y,
                     probeIdx / (gridDensity.x * gridDensity.y));
    }

    __host__ __device__ inline int ProbeIdxFromGridIdx(const ivec3& gridIdx, const ivec3& gridDensity)
    {
        return gridIdx.z * gridDensity.x * gridDensity.y + gridIdx.y * gridDensity.x + gridIdx.x;
    }

    struct LightProbeGridParams
    {        
        __host__ __device__ LightProbeGridParams();
        __host__ LightProbeGridParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        BidirectionalTransform		transform;
        ivec3						gridDensity;
        int							shOrder;

        int                         axisSwizzle;
        vec3                        axisMultiplier;

        bool                        debugOutputPRef;
        bool                        debugOutputValidity;
        bool                        debugBakePRef;
        bool                        invertX, invertY, invertZ;

        int                         coefficientsPerProbe;
        int                         numProbes;
        vec3						aspectRatio;
    };

    namespace Device
    {
        template<typename T> class Array;
        
        class LightProbeGrid : public Device::RenderObject, public AssetTags<Host::LightProbeGrid, Device::LightProbeGrid>
        {
        public:
            __host__ __device__ LightProbeGrid();
            __device__ void Synchronise(const LightProbeGridParams& params);
            __device__ void Synchronise(Device::Array<vec3>* data) { cu_data = data; }

            __device__ void SetSHCoefficient(const int probeIdx, const int coeffIdx, const vec3& L);
            __device__ vec3 GetSHCoefficient(const int probeIdx, const int coeffIdx) const;
            __device__ vec3 Evaluate(const HitCtx& hitCtx) const;           

        private:
            __device__ vec3 InterpolateCoefficient(const ivec3 gridIdx, const uint coeffIdx, const vec3& delta) const;

            LightProbeGridParams    m_params;
            Device::Array<vec3>*    cu_data = nullptr;
        };
    }

    namespace Host
    {
        template<typename T> class Array;
        
        class LightProbeGrid : public Host::RenderObject, public AssetTags<Host::LightProbeGrid, Device::LightProbeGrid>
        {
        public:
            __host__ LightProbeGrid(const std::string& id);
            __host__ virtual ~LightProbeGrid();

            __host__ void Prepare(const LightProbeGridParams& params);
            __host__  virtual void OnDestroyAsset() override final;
            __host__ virtual void FromJson(const ::Json::Node& renderParamsJson, const uint flags) override final;
            __host__ Device::LightProbeGrid* GetDeviceInstance() { return cu_deviceData; }
            __host__ void IsConverged() const;
            __host__ bool IsValid() const;
            __host__ void GetRawData(std::vector<vec3>& data) const;
            __host__ const LightProbeGridParams& GetParams() const { return m_params; }
            __host__ const std::string& GetUSDExportPath() const { return m_usdExportPath; }

        private:
            Device::LightProbeGrid*         cu_deviceData = nullptr;

            AssetHandle<Host::Array<vec3>>  m_data;
            LightProbeGridParams            m_params;
            std::string                     m_usdExportPath;
        };
    }
}