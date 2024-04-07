#include "Camera2D.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"
#include "core/AssetAllocator.cuh"
#include "core/Vector.cuh"

namespace Enso
{
    constexpr size_t kAccumBufferSize = 1024 * 1024;
    
    __device__ void Device::Camera2D::ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange)
    {
        if (kKernelIdx >= m_params.totalAccumUnits) { return; }

        assert(m_reduceBuffer);
        assert(m_accumBuffer);

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
                        assert(kKernelIdx + iterationSize * m_params.numHarmonics < kAccumBufferSize);
                        texel += accumBuffer[kKernelIdx + iterationSize * m_params.numHarmonics];
                    }
                }
                else
                {
                    assert(kKernelIdx + iterationSize * m_params.numHarmonics < kAccumBufferSize);
                    assert(subprobeIdx + iterationSize < m_params.subprobesPerProbe);

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

            assert(probeIdx < m_params.numProbes&& coeffIdx < m_params.numHarmonics);

            (*m_objectsBuffer)[probeIdx * m_params.numHarmonics + coeffIdx] = reduceBuffer[kKernelIdx];
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(ReduceAccumulationBuffer);

    Host::Camera2D::Camera2D(const std::string& id, const int numProbes, const int numHarmonics) :
        GenericObject(id)
    {
        Assert(m_scene);

        // Establish the properties of the grid
        m_params.numProbes = numProbes;
        m_params.numHarmonics = (numHarmonics - 1) * 2 + 1;
        m_params.totalGridUnits = m_params.numProbes * m_params.numHarmonics;

        // Derive some more properties used when accumulating and reducing.
        m_params.subprobesPerProbe = std::min(kAccumBufferSize / m_params.numProbes,
            kAccumBufferSize / m_params.totalGridUnits);
        m_params.unitsPerProbe = m_params.subprobesPerProbe * m_params.numHarmonics;
        m_params.totalSubprobes = m_params.subprobesPerProbe * m_params.numProbes;
        m_params.totalAccumUnits = m_params.totalSubprobes * m_params.numHarmonics;

        // Create some assets
        m_hostAccumBuffer = CreateChildAsset<Host::Vector<vec3>>("accumBuffer", kAccumBufferSize, kVectorHostAlloc);
        m_hostReduceBuffer = CreateChildAsset<Host::Vector<vec3>>("reduceBuffer", kAccumBufferSize, kVectorHostAlloc);
        m_hostProxyGrid = CreateChildAsset<Host::Vector<vec3>>("proxyGrid", m_params.totalGridUnits, kVectorHostAlloc);

        // Set the device objects
        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.accumBuffer = m_hostProxyGrid->GetDeviceInstance();

        // Set the parameters for the accumulate and reduce kernels
        m_kernelParams.blockSize = 256;
        m_kernelParams.grids.accumSize = (m_params.totalSubprobes + m_kernelParams.blockSize - 1) / m_kernelParams.blockSize;
        m_kernelParams.grids.reduceSize = (m_params.totalAccumUnits + m_kernelParams.blockSize - 1) / m_kernelParams.blockSize;
    }

    // Reduces the contents of the bins down into a single array of values
    __host__ void Host::Camera2D::ReduceAccumulationBuffer()
    {
        // Used when parallel reducing the accumluation buffer
        const uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.subprobesPerProbe);

        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 1)
        {
            KernelReduceAccumulationBuffer << < m_kernelParams.grids.reduceSize, m_kernelParams.blockSize >> > (cu_deviceInstance, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
            batchSize >>= 1;
        }
        IsOk(cudaDeviceSynchronize());
    }

    // Erases the contents of the accumulation buffer bins
    __host__ void Host::Camera2D::ClearAccumulationBuffer()
    {
        m_hostAccumBuffer->Wipe();
    }
}