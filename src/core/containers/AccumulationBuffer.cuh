#pragma once

#include "core/assets/AssetAllocator.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/Math.cuh"

namespace Enso
{
    class Ray2D;
    class HitCtx2D;
    class RenderCtx;
    class UIViewCtx;

    struct AccumulationBufferParams
    {
        __host__ __device__ AccumulationBufferParams() {  }

        __device__ void Validate() const
        {
            CudaAssert(numProbes > 0);
            CudaAssert(numHarmonics > 0);
            CudaAssert(accumBufferSize > 0);

            CudaAssert(totalGridUnits > 0);
            CudaAssert(subprobesPerProbe > 0);
            CudaAssert(unitsPerProbe > 0);
            CudaAssert(totalAccumUnits > 0);
            CudaAssert(totalSubprobes > 0);
            CudaAssert(kernel.blockSize > 0);
            CudaAssert(kernel.grids.accumSize > 0);
            CudaAssert(kernel.grids.reduceSize > 0);
        }

        int      numProbes = 0;
        int      numHarmonics = 0;
        size_t   accumBufferSize = 0;

        int      totalGridUnits = 0; //             <-- The total number of units in the reduced grid
        int	     subprobesPerProbe = 0;	//			<-- A sub-probe is a set of SH coefficients + data. Multiple sub-probes are accumulated to make a full probe. 
        int      unitsPerProbe = 0; //				<-- The total number of accumulation units (coefficients + data) accross all sub-probes, per probe
        int      totalAccumUnits = 0; //		    <-- The total number of accumulation units in the grid
        int      totalSubprobes = 0; //				<-- The total number of subprobes in the grid 

        struct
        {
            int blockSize = 0;
            struct
            {
                int accumSize = 0;
                int reduceSize = 0;
            }
            grids;
        }
        kernel;
    };

    struct AccumulationBufferObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(accumBuffer);
            CudaAssert(reduceBuffer);
            CudaAssert(outputBuffer);
        }

        Device::Vector<vec3>* accumBuffer = nullptr;
        Device::Vector<vec3>* reduceBuffer = nullptr;
        Device::Vector<vec3>* outputBuffer = nullptr;
    };

    namespace Device
    {
        class AccumulationBuffer : public Device::Asset
        {
        public:
            __host__ __device__ AccumulationBuffer() {}

            __device__ void         Synchronise(const AccumulationBufferParams& params) { m_params = params; }
            __device__ void         Synchronise(const AccumulationBufferObjects& objects) { m_objects = objects; }

            __device__ void         Reduce(const uint batchSize, const uvec2 batchRange, const int norm);
            __device__ void         Accumulate(const vec3& L, const int& probeIdx, const int& subProbeIdx, const int& coeffIdx);
            __device__ void         Accumulate(const vec3& L, const int& probeIdx, const int& subProbeIdx);
            
            __device__ vec3&        operator[](const uint& idx);
            __device__ const vec3&  operator[](const uint& idx) const;

            __device__ const vec3   Evaluate(const int probeIdx, const int harmonicIdx) const;
            __device__ const vec3   Evaluate(const int probeIdx) const;

            __device__ inline int   GetNumHarmonics() const { return m_params.numHarmonics; }
            __device__ inline int   GetNumProbes() const { return m_params.numProbes; }
            __device__ inline int   GetSubprobesPerProbe() const { return m_params.subprobesPerProbe; }

        protected:
            AccumulationBufferParams    m_params;
            AccumulationBufferObjects   m_objects;
        };
    }

    namespace Host
    {
        class AccumulationBuffer : public Host::Asset
        {
        public:
            AccumulationBuffer(const Asset::InitCtx& initCtx, const int numProbes, const int numHarmonics, const size_t accumBufferSize);
            virtual ~AccumulationBuffer() noexcept;

            __host__ void Reduce();
            __host__ void Clear();  
            __host__ int  GetTotalAccumulatedSamples() const;
            
            __host__ const AccumulationBufferParams& GetParams() const { return m_params; }
            __host__ Device::AccumulationBuffer* GetDeviceInstance() const { return cu_deviceInstance; }

        private:
            __host__ void Synchronise(const int syncFlags);

            AssetHandle<Host::Vector<vec3>>         m_hostAccumBuffer;
            AssetHandle<Host::Vector<vec3>>         m_hostReduceBuffer;
            AssetHandle<Host::Vector<vec3>>         m_hostOutputBuffer;

            Device::AccumulationBuffer*             cu_deviceInstance;
            AccumulationBufferParams                m_params;
            AccumulationBufferObjects               m_deviceObjects;

            int                                     m_norm;
        };
    };
}