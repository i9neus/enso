#pragma once

#include "../CudaRenderObject.cuh"

namespace Cuda
{
    namespace Host { class LightProbeGrid; }
    struct HitCtx;

    enum AxisSwizzle : int { kXYZ, kXZY, kYXZ, kYZX, kZXY, kZYX };

    enum ProbeGridOutputMode : int
    {
        kProbeGridIrradiance,
        kProbeGridValidity,
        kProbeGridHarmonicMean,
        kProbeGridPref
    };

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

        bool                        useValidity;
        int                         outputMode;
        bool                        invertX, invertY, invertZ;

        int                         coefficientsPerProbe;
        int                         numProbes;
        vec3						aspectRatio;
        int                         maxSamplesPerProbe;
    };

    namespace Device
    {
        template<typename T> class Array;
        
        class LightProbeGrid : public Device::RenderObject, public AssetTags<Host::LightProbeGrid, Device::LightProbeGrid>
        {
        public:
            struct Objects
            {
                Device::Array<vec3>* cu_shData = nullptr;
                Device::Array<uchar>* cu_validityData = nullptr;
            };

            __host__ __device__ LightProbeGrid();
            __device__ void Synchronise(const LightProbeGridParams& params);
            __device__ void Synchronise(const Objects& objects) { m_objects = objects; }

            __device__ void SetSHCoefficient(const int probeIdx, const int coeffIdx, const vec3& L);
            __device__ vec3 GetSHCoefficient(const int probeIdx, const int coeffIdx) const;
            __device__ vec3 Evaluate(const HitCtx& hitCtx) const;
            __device__ void PrepareValidityGrid();

        private:
            __device__ vec3 InterpolateCoefficient(const ivec3 gridIdx, const uint coeffIdx, const vec3& delta) const;
            __device__ vec3 WeightedInterpolateCoefficient(const ivec3 gridIdx, const uint coeffIdx, const vec3& delta, const uchar validity) const;
            __device__ __forceinline__ uchar GetValidity(const ivec3& gridIdx) const;

            LightProbeGridParams    m_params;
            Objects                 m_objects;
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

            __host__ void                               Prepare(const LightProbeGridParams& params);

            __host__  virtual void                      OnDestroyAsset() override final;
            __host__ virtual void                       FromJson(const ::Json::Node& renderParamsJson, const uint flags) override final;
            __host__ Device::LightProbeGrid*            GetDeviceInstance() { return cu_deviceData; }
            __host__ void                               IsConverged() const;
            __host__ bool                               IsValid() const;
            __host__ void                               GetRawData(std::vector<vec3>& data) const;
            __host__ const LightProbeGridParams&        GetParams() const { return m_params; }
            __host__ const std::string&                 GetUSDExportPath() const { return m_usdExportPath; }
            __host__ AssetHandle<Host::Array<vec3>>&    GetSHDataAsset() { return m_shData; }
            __host__ AssetHandle<Host::Array<uchar>>&   GetValidityDataAsset() { return m_validityData; }

        private:
            Device::LightProbeGrid*         cu_deviceData = nullptr;
            Device::LightProbeGrid::Objects m_deviceObjects;

            AssetHandle<Host::Array<vec3>>  m_shData;
            AssetHandle<Host::Array<uchar>>  m_validityData;
            LightProbeGridParams            m_params;
            std::string                     m_usdExportPath;
        };
    }
}