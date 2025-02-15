#include "AccumulationBuffer.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"
#include "core/assets/AssetAllocator.cuh"
#include "Vector.cuh"
#include "core/assets/GenericObject.cuh"

namespace Enso
{     
    /*
    * Memory ordering from smallest to largest:
    *  - RGBW tuples (as vec3s)
    *  - Harmonics
    *  - Batches
    *  - Probes
    */

    __device__ void Device::AccumulationBuffer::Accumulate(const vec3& L, const int& probeIdx, const int& subProbeIdx, const int& coeffIdx)
    {
        (*m_objects.accumBuffer)[(probeIdx * m_params.subprobesPerProbe + subProbeIdx) * m_params.numHarmonics + coeffIdx] += L;
    }

    __device__ void Device::AccumulationBuffer::Accumulate(const vec3& L, const int& probeIdx, const int& subProbeIdx)
    {
        (*m_objects.accumBuffer)[probeIdx * m_params.subprobesPerProbe + subProbeIdx] += L;
    }

    // Directly indexes the accumulation buffer
    __device__ vec3& Device::AccumulationBuffer::operator[](const uint& sampleIdx) { return (*m_objects.accumBuffer)[sampleIdx]; }
    __device__ const vec3& Device::AccumulationBuffer::operator[](const uint& sampleIdx) const { return (*m_objects.accumBuffer)[sampleIdx]; }

    // Evaluates the output buffer based on a probe index
    // FIXME: Pre-divide during reduce operation, not here.
    __device__ const vec3 Device::AccumulationBuffer::Evaluate(const int probeIdx, const int harmonicIdx) const 
    { 
        return (*m_objects.outputBuffer)[probeIdx * m_params.numHarmonics + harmonicIdx] / m_params.subprobesPerProbe;
    }
    
    // Evaluates the output buffer assuming only a single harmonic
    __device__ const vec3 Device::AccumulationBuffer::Evaluate(const int probeIdx) const
    {
        return (*m_objects.outputBuffer)[probeIdx] / m_params.subprobesPerProbe;
    }

    __device__ void Device::AccumulationBuffer::Reduce(const uint batchSize, const uvec2 batchRange, const int norm)
    {
        if (kKernelIdx >= m_params.totalAccumUnits) { return; }

        CudaAssertDebug(m_objects.reduceBuffer);
        CudaAssertDebug(m_objects.accumBuffer);

        auto& accumBuffer = *m_objects.accumBuffer;
        auto& reduceBuffer = *m_objects.reduceBuffer;
        const int subprobeIdx = (kKernelIdx / m_params.numHarmonics) % m_params.subprobesPerProbe;

        for (uint iterationSize = batchRange[0] / 2; iterationSize > batchRange[1] / 2; iterationSize >>= 1)
        {
            if (subprobeIdx < iterationSize)
            {
                // For the first iteration, copy the data out of the accumulation buffer
                if (iterationSize == batchSize / 2)
                {
                    auto& texel = reduceBuffer[kKernelIdx];
                    texel = accumBuffer[kKernelIdx];

                    if (subprobeIdx + iterationSize < m_params.subprobesPerProbe)
                    {
                        CudaAssertDebug(kKernelIdx + iterationSize * m_params.numHarmonics < m_params.accumBufferSize);
                        texel += accumBuffer[kKernelIdx + iterationSize * m_params.numHarmonics];
                    }
                }
                else
                {
                    CudaAssertDebug(kKernelIdx + iterationSize * m_params.numHarmonics < m_params.accumBufferSize);
                    CudaAssertDebug(subprobeIdx + iterationSize < m_params.subprobesPerProbe);

                    reduceBuffer[kKernelIdx] += reduceBuffer[kKernelIdx + iterationSize * m_params.numHarmonics];
                }
            }

            __syncthreads();
        }

        // After the last operation, cache the accumulated value in the probe grid
        if (subprobeIdx == 0 && batchRange[0] == 2)
        {
            const int probeIdx = kKernelIdx / m_params.unitsPerProbe;
            const int coeffIdx = kKernelIdx % m_params.numHarmonics;

            CudaAssertDebug(probeIdx < m_params.numProbes&& coeffIdx < m_params.numHarmonics);

            (*m_objects.outputBuffer)[probeIdx * m_params.numHarmonics + coeffIdx] = reduceBuffer[kKernelIdx] / float(max(1, norm));
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Reduce);

    Host::AccumulationBuffer::AccumulationBuffer(const Asset::InitCtx& initCtx, const int numProbes, const int numHarmonics, const size_t accumBufferSize) :
        Asset(initCtx),
        m_norm(0)
    {
        Assert(accumBufferSize >= numProbes * numHarmonics);
        
        // Establish the properties of the grid
        m_params.numProbes = numProbes;
        m_params.numHarmonics = (numHarmonics - 1) * 2 + 1;
        m_params.accumBufferSize = accumBufferSize;
        m_params.totalGridUnits = m_params.numProbes * m_params.numHarmonics;

        // Derive some more properties used when accumulating and reducing.
        m_params.subprobesPerProbe = std::min(m_params.accumBufferSize / m_params.numProbes,
            m_params.accumBufferSize / m_params.totalGridUnits);
        m_params.unitsPerProbe = m_params.subprobesPerProbe * m_params.numHarmonics;
        m_params.totalSubprobes = m_params.subprobesPerProbe * m_params.numProbes;
        m_params.totalAccumUnits = m_params.totalSubprobes * m_params.numHarmonics;

        // Create some assets
        m_hostAccumBuffer = AssetAllocator::CreateChildAsset<Host::Vector<vec3>>(*this, "accumBuffer", m_params.accumBufferSize);
        m_hostReduceBuffer = AssetAllocator::CreateChildAsset<Host::Vector<vec3>>(*this, "reduceBuffer", m_params.accumBufferSize);
        m_hostOutputBuffer = AssetAllocator::CreateChildAsset<Host::Vector<vec3>>(*this, "outputBuffer", m_params.totalGridUnits);

        // Set the device objects
        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.outputBuffer = m_hostOutputBuffer->GetDeviceInstance();

        // Set the parameters for the accumulate and reduce kernels
        m_params.kernel.blockSize = 256;
        m_params.kernel.grids.accumSize = (m_params.totalSubprobes + m_params.kernel.blockSize - 1) / m_params.kernel.blockSize;
        m_params.kernel.grids.reduceSize = (m_params.totalAccumUnits + m_params.kernel.blockSize - 1) / m_params.kernel.blockSize;

        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::AccumulationBuffer>(*this);
        Synchronise(kSyncParams | kSyncObjects);
    }

    Host::AccumulationBuffer::~AccumulationBuffer() noexcept
    {
        m_hostAccumBuffer.DestroyAsset();
        m_hostOutputBuffer.DestroyAsset();
        m_hostReduceBuffer.DestroyAsset();
    }

    __host__ void Host::AccumulationBuffer::Synchronise(const int syncFlags)
    {
        if (syncFlags & kSyncParams) { SynchroniseObjects<Device::AccumulationBuffer>(cu_deviceInstance, m_params); }
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::AccumulationBuffer>(cu_deviceInstance, m_deviceObjects); }
    }

    // Reduces the contents of the bins down into a single array of values
    __host__ void Host::AccumulationBuffer::Reduce()
    {
        // Used when parallel reducing the accumluation buffer
        const uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.subprobesPerProbe);

        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 1)
        {
            KernelReduce << < m_params.kernel.grids.reduceSize, m_params.kernel.blockSize >> > (cu_deviceInstance, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1), m_norm);
            batchSize >>= 1;
        }
        IsOk(cudaDeviceSynchronize());

        //Increment the normalisation factor
        ++m_norm;
    }

    // Erases the contents of the accumulation buffer bins
    __host__ void Host::AccumulationBuffer::Clear()
    {
        m_hostAccumBuffer->Wipe();
        m_norm = 0;
    }

    __host__ int Host::AccumulationBuffer::GetTotalAccumulatedSamples() const
    {
        return m_norm * m_params.subprobesPerProbe;
    }
}