#pragma once

#include "../CudaRenderObjectContainer.cuh"
#include "../CudaDeviceObjectRAII.cuh"
#include "../cameras/CudaCamera.cuh"

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
        kProbeGridPref,
        kProbeGridConvergence,
        kProbeGridSqrError
    };

    enum ProbeGridMetadata : int
    {
        kProbeValidity = 0,
        kProbeFilterWeights = 1,
        kProbeNumSamples = 2
    };

    enum ProbeGridConvergenceFlags : uchar
    {
        kProbeUnconverged = 1,
        kProbeBelowSampleMin = 2,
        kProbeAtSampleMax = 4
    };

    __host__ __device__ __forceinline__ ivec3 GridPosFromProbeIdx(const int& probeIdx, const ivec3& gridDensity)
    {
        return ivec3(probeIdx % gridDensity.x, 
                     (probeIdx / gridDensity.x) % gridDensity.y,
                     probeIdx / (gridDensity.x * gridDensity.y));
    }

    __host__ __device__ __forceinline__ ivec3 GridPosFromProbeIdx(const int& probeIdx, const int& gridDensity)
    {
        return ivec3(probeIdx % gridDensity,
            (probeIdx / gridDensity) % gridDensity,
            probeIdx / (gridDensity * gridDensity));
    }

    __host__ __device__ __forceinline__ int ProbeIdxFromGridPos(const ivec3& gridIdx, const ivec3& gridDensity)
    {
        return gridIdx.z * gridDensity.x * gridDensity.y + gridIdx.y * gridDensity.x + gridIdx.x;
    }

    struct LightProbeGridParams
    {        
        __host__ __device__ LightProbeGridParams();
        __host__ LightProbeGridParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        __host__ void Prepare();

        __host__ __device__ bool operator!=(const LightProbeGridParams& rhs) const;

        BidirectionalTransform		transform;
        ivec3						gridDensity;
        ivec3                       clipRegion[2];
        int							shOrder;
        bool                        dilate;

        int                         posSwizzle;
        int                         shSwizzle;
        bool                        posInvertX, posInvertY, posInvertZ;
        bool                        shInvertX, shInvertY, shInvertZ;

        //vec3                        axisMultiplier;

        bool                        useValidity;
        int                         outputMode;
        int                         inputColourSpace;

        int                         shCoefficientsPerProbe;
        int                         coefficientsPerProbe;
        int                         numProbes;
        vec3						aspectRatio;
        ivec2                       minMaxSamplesPerProbe;

        CameraParams                camera;
    };

    namespace Device
    {
        template<typename T> class Array;
        
        class LightProbeGrid : public Device::RenderObject, public AssetTags<Host::LightProbeGrid, Device::LightProbeGrid>
        {
        public:
            struct AggregateStatistics
            {
                __host__ __device__ AggregateStatistics() : minMaxSamples(-1.0f), meanValidity(-1.0f), meanDistance(-1.0f), meanSamples(-1.0f) {}

                // Only L1 at this time
                static constexpr int kStatsNumCoeffs = 4;

                vec2	minMaxSamples;
                vec2	minMaxCoeffs[kStatsNumCoeffs]; 
                float   meanSamples;
                float   meanSqrIntensity[kStatsNumCoeffs];
                float	meanValidity;
                float	meanDistance;
            };           

            struct Objects
            {
                Device::Array<vec3>* cu_shData = nullptr;                
                union
                {
                    Device::Array<vec3>* cu_swapBuffer = nullptr;
                    Device::Array<vec3>* cu_shLaplacianData;
                };
                Device::Array<uchar>*   cu_validityData = nullptr;
                Device::Array<uchar>*   cu_adaptiveSamplingData = nullptr;
                Device::Array<vec2>*    cu_errorData = nullptr;
                float*                  cu_meanI = nullptr;
            };

            __host__ __device__ LightProbeGrid();
            __device__ void Synchronise(const LightProbeGridParams& params);
            __device__ void Synchronise(const Objects& objects) { m_objects = objects; }

            __device__ void PrepareValidityGrid();
            __device__ void Dilate();
            __device__ void SetSHCoefficient(const int probeIdx, const int coeffIdx, const vec3& L);
            __device__ void SetSHLaplacianCoefficient(const int probeIdx, const int coeffIdx, const vec3& L);
            __device__ vec3* At(const int probeIdx);
            __device__ const vec3* At(const int probeIdx) const { return const_cast<LightProbeGrid*>(this)->At(probeIdx); }
            __device__ vec3* LaplacianAt(const int probeIdx);
            __device__ const vec3* LaplacianAt(const int probeIdx) const { return const_cast<LightProbeGrid*>(this)->LaplacianAt(probeIdx); }
            __device__ vec3* At(const ivec3& gridIdx);
            __device__ const vec3* At(const ivec3& gridIdx) const { return const_cast<LightProbeGrid*>(this)->At(gridIdx); }
            __device__ int IdxAt(const ivec3& gridIdx) const;
            __device__ vec3 Evaluate(const HitCtx& hitCtx, const int maxSHOrder) const;
            __device__ const LightProbeGridParams& GetParams() const { return m_params; }
            __device__ void GetProbeGridAggregateStatistics(AggregateStatistics& result) const;
            __device__ void ComputeProbeGridHistograms(AggregateStatistics& result, uint* distanceHistogram) const;

        private:
            template<typename T>
            __device__ T InterpolateCoefficient(const Device::Array<T>& data, const ivec3 gridPos, const int coeffIdx, const int coeffsPerProbe, const vec3& delta) const
            {
                T vert[8];
                for (int z = 0, idx = 0; z < 2; z++)
                {
                    for (int y = 0; y < 2; y++)
                    {
                        for (int x = 0; x < 2; x++, idx++)
                        {
                            const ivec3 vertCoord = gridPos + ivec3(x, y, z);
                            const int sampleIdx = coeffsPerProbe * ProbeIdxFromGridPos(vertCoord, m_params.gridDensity);

                            assert(sampleIdx < data.Size());

                            vert[idx] = data[sampleIdx + coeffIdx];
                        }
                    }
                }

                // Trilinear interpolate
                return mix(mix(mix(vert[0], vert[1], delta.x), mix(vert[2], vert[3], delta.x), delta.y),
                    mix(mix(vert[4], vert[5], delta.x), mix(vert[6], vert[7], delta.x), delta.y), delta.z);
            }

            template<typename T>
            __forceinline__ __device__ const T& NearestNeighbourCoefficient(const Device::Array<T>& data, const ivec3& gridPos, const uint coeffIdx, const uint coeffPerProbe) const
            {
                return data[coeffPerProbe * ProbeIdxFromGridPos(gridPos, m_params.gridDensity) + coeffIdx];
            }

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
            struct AggregateStatistics : Device::LightProbeGrid::AggregateStatistics
            {
                AggregateStatistics() : isConverged(false) {}

                bool                                        isConverged;
                DeviceObjectRAII<uint, 200>			        coeffHistogram;
            };

            __host__ LightProbeGrid(const std::string& id);
            __host__ virtual ~LightProbeGrid();

            __host__ void                               Prepare();
            __host__ void                               Prepare(const LightProbeGridParams& params);
            __host__ void                               Integrate();
            __host__ void                               Replace(const LightProbeGrid& other);
            __host__ void                               Swap(LightProbeGrid& other);

            __host__  virtual void                      OnDestroyAsset() override final;
            __host__ virtual void                       FromJson(const ::Json::Node& renderParamsJson, const uint flags) override final;
            __host__ Device::LightProbeGrid*            GetDeviceInstance() { return cu_deviceData; }
            __host__ bool                               IsConverged() const { return m_statistics.isConverged; }
            __host__ bool                               HasSemaphore(const std::string& tag) const;
            __host__ int                                GetSemaphore(const std::string& tag) const;
            __host__ void                               SetSemaphore(const std::string& tag, const int data);
            __host__ bool                               IsValid() const;
            __host__ void                               GetRawData(std::vector<vec3>& data) const;
            __host__ void                               SetRawData(const std::vector<vec3>& data);
            __host__ void                               SetExternalBuffers(AssetHandle<Host::Array<uchar>> adaptiveSamplingData, 
                                                                           AssetHandle<Host::Array<vec2>> errorData, 
                                                                           DeviceObjectRAII<float>& meanI);
            __host__ void                               SetOutputMode(const int& outputMode);
            __host__ const LightProbeGridParams&        GetParams() const { return m_params; }
            __host__ const std::string&                 GetUSDExportPath() const { return m_usdExportPath; }
            __host__ AssetHandle<Host::Array<vec3>>&    GetSHDataAsset() { return m_shData; }
            __host__ AssetHandle<Host::Array<uchar>>&   GetValidityDataAsset() { return m_validityData; }
            __host__ const AggregateStatistics&         UpdateAggregateStatistics(const int maxSamples);
            __host__ const AggregateStatistics&         GetAggregateStatistics() const { return m_statistics; }
            __host__ void                               PushClipRegion(const ivec3* region);
            __host__ void                               PopClipRegion();

        private:
            Device::LightProbeGrid*         cu_deviceData = nullptr;
            Device::LightProbeGrid::Objects m_deviceObjects;

            AssetHandle<Host::Array<vec3>>  m_shData;            
            AssetHandle<Host::Array<uchar>> m_validityData;
            AssetHandle<Host::Array<vec3>>  m_shLaplacianData;
            AssetHandle<Host::Array<uchar>> m_adaptiveSamplingData;
            AssetHandle<Host::Array<vec2>>  m_errorData;
            DeviceObjectRAII<float>*        m_hostMeanI;
            
            LightProbeGridParams            m_params;
            std::string                     m_usdExportPath;
            std::unordered_map<std::string, int> m_semaphoreRegistry;
            std::vector<std::pair<ivec3, ivec3>> m_clipRegionStack;

            AggregateStatistics             m_statistics;
            DeviceObjectRAII<Device::LightProbeGrid::AggregateStatistics>	m_probeAggregateData;            
        };
    }
}