#include "CudaLightProbeDataTransform.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ ivec3 SwizzleIndices(const int type)
    {
        // Swizzle the axes
        switch (type)
        {
        case kXZY: return ivec3(0, 2, 1);
        case kYXZ: return ivec3(1, 0, 2);
        case kYZX: return ivec3(1, 2, 0);
        case kZXY: return ivec3(2, 0, 1);
        case kZYX: return ivec3(2, 1, 0);
        }
        return ivec3(0, 1, 2);
    };

    __host__ ivec3 SwizzleVector(const ivec3& v, const ivec3& i)
    {
        return ivec3(v[i[0]], v[i[1]], v[i[2]]);
    }

    __host__ void LightProbeDataTransform::DirectionalTransform::Initialise(const LightProbeGridParams& gridParams)
    {
        probeIdx.resize(gridParams.numProbes);
        coeffIdx.resize(gridParams.coefficientsPerProbe * 3);
        sh.resize(gridParams.coefficientsPerProbe);
    }
    
    __host__ void LightProbeDataTransform::ConstructSHIndices()
    {
        // Swizzle the indices for the L1 RGB coefficients
        std::vector<int> shSwiz(m_gridParams.coefficientsPerProbe);
        for (int i = 0; i < shSwiz.size(); ++i) { shSwiz[i] = i; }
        shSwiz[1] = SwizzleIndices(m_gridParams.dataTransform.shSwizzle)[0] + 1; 
        shSwiz[2] = SwizzleIndices(m_gridParams.dataTransform.shSwizzle)[1] + 1;
        shSwiz[3] = SwizzleIndices(m_gridParams.dataTransform.shSwizzle)[2] + 1;

        // Convert the coefficient offsets into per-channel offsets
        // E.g. { 0, 2, 1 } -> { 0, 1, 2, 6, 7, 8, 3, 4, 5 }
        for (int i = 0; i < m_forward.coeffIdx.size(); ++i)
        {
            m_forward.coeffIdx[i] = shSwiz[i / 3] * 3 + i % 3;
        }

        // Unity coefficient packing. L0 remains RGB, but L1 is grouped by R, G and B
        auto& f = m_forward.coeffIdx;
        std::vector<int> g = f;
        g[0] = f[0];   g[1] = f[1];   g[2] = f[2];      // L0
        g[3] = f[3];   g[4] = f[6];   g[5] = f[9];      // L1_1
        g[6] = f[4];   g[7] = f[7];   g[8] = f[10];     // L10
        g[9] = f[5];   g[10] = f[8];  g[11] = f[11];    // L11
        g[12] = f[12]; g[13] = f[13]; g[14] = f[14];    // Metadata
        f = g;

        // Derive the inverse coefficient mapping
        for (int i = 0; i < m_forward.coeffIdx.size(); ++i)
        {
            g[m_forward.coeffIdx[i]] = i;
        }
        m_inverse.coeffIdx = g;
    }

    __host__ void LightProbeDataTransform::ConstructSHTransforms()
    {
        // Factors for inverting L1 SH coefficients
        std::vector<float> shDirs(m_gridParams.coefficientsPerProbe, 1.0f);
        if (m_gridParams.dataTransform.shInvertX) { shDirs[1] = -1.0f; }
        if (m_gridParams.dataTransform.shInvertY) { shDirs[2] = -1.0f; }
        if (m_gridParams.dataTransform.shInvertZ) { shDirs[3] = -1.0f; }

        // Initialise the forward transformation with the directional scaling factors
        auto& fwdSh = m_forward.sh;
        for (int i = 0; i < m_forward.sh.size(); ++i)
        {
            fwdSh[i] = mat2(shDirs[i], 0.0, 0.0, 1.0);
        }

        // Unity coefficient rescaling
        fwdSh[0].i00 *= SH::Legendre(0);
        fwdSh[1].i00 *= SH::Legendre(1);
        fwdSh[2].i00 *= SH::Legendre(1);
        fwdSh[3].i00 *= SH::Legendre(1);

        // Unity validity is 1 minus the actual validity. 
        fwdSh[4] = mat2(-1.0, 0.0, 0.0, 1.0);

        // Invert the forward transformation
        auto& invSh = m_inverse.sh;
        for (int i = 0; i < invSh.size(); ++i)
        {
            invSh[i] = inverse(fwdSh[i]);
        }
    }

    __host__ void LightProbeDataTransform::ConstructProbePositionIndices()
    {
        // Generate swizzle indices for probe positions
        const ivec3 posSwiz = SwizzleIndices(m_gridParams.dataTransform.posSwizzle);

        // Swizzle the grid density
        const ivec3 swizzledGridDensity = SwizzleVector(m_gridParams.gridDensity, posSwiz);

        // Swizzle the axes
        for (int probeIdx = 0; probeIdx < m_gridParams.numProbes; ++probeIdx)
        {
            ivec3 gridPos = GridPosFromProbeIdx(probeIdx, m_gridParams.gridDensity);

            // Invert the axes where appropriate
            if (m_gridParams.dataTransform.posInvertX) { gridPos.x = m_gridParams.gridDensity.x - gridPos.x - 1; }
            if (m_gridParams.dataTransform.posInvertY) { gridPos.y = m_gridParams.gridDensity.y - gridPos.y - 1; }
            if (m_gridParams.dataTransform.posInvertZ) { gridPos.z = m_gridParams.gridDensity.z - gridPos.z - 1; }

            // Swizzle the grid index
            ivec3 swizzledGridPos = SwizzleVector(gridPos, posSwiz);

            // Map back onto the data array
            const uint swizzledProbeIdx = ProbeIdxFromGridPos(swizzledGridPos, swizzledGridDensity);
            Assert(swizzledProbeIdx < m_gridParams.numProbes);

            // Store the indirections
            m_forward.probeIdx[probeIdx] = swizzledProbeIdx;
            m_inverse.probeIdx[swizzledProbeIdx] = probeIdx;
        }
    }

    __host__ void LightProbeDataTransform::Construct(const LightProbeGridParams& gridParams)
    {
        // Forward transformation operations are applied in this order:
        //   1. Swizzle probe positions
        //   2. Transform coefficients
        //   3. Swizzle coefficients

        m_gridParams = gridParams;
        if (m_gridParams.shOrder != 1)
        {
            Log::Error("Error: light probe data transform currently only supports L1 probes.");
            return;
        }

        m_forward.Initialise(gridParams);
        m_inverse.Initialise(gridParams);

        // Construct each component of the transform
        ConstructProbePositionIndices();
        ConstructSHIndices();
        ConstructSHTransforms();
    }

    __host__ void LightProbeDataTransform::Forward(const std::vector<vec3>& inputData, std::vector<vec3>& outputData) const
    {
        if (inputData.size() < m_forward.probeIdx.size())
        {
            Log::Error("Cannot forward transform probe grid data. Size of input data buffer is smaller the transform (%ix%ix%i).",
                m_gridParams.gridDensity.x, m_gridParams.gridDensity.y, m_gridParams.gridDensity.z);
            return;
        }

        std::vector<float> swapBufferA(3 * m_gridParams.coefficientsPerProbe);
        std::vector<float> swapBufferB(3 * m_gridParams.coefficientsPerProbe);

        for (int probeIdx = 0; probeIdx < m_gridParams.numProbes; ++probeIdx)
        {
            // Read in the block of SH coefficients
            const int inputIdx = probeIdx * m_gridParams.coefficientsPerProbe;
            std::memcpy(&swapBufferA[0], &inputData[inputIdx], sizeof(float) * 3 * m_gridParams.coefficientsPerProbe);
  
            // Forward transform SH coefficients
            for (int shIdx = 0; shIdx < m_gridParams.coefficientsPerProbe; ++shIdx)
            {
                swapBufferB[shIdx * 3] = (m_forward.sh[shIdx] * vec2(swapBufferA[shIdx * 3], 1.0)).x;
                swapBufferB[shIdx * 3 + 1] = (m_forward.sh[shIdx] * vec2(swapBufferA[shIdx * 3 + 1], 1.0)).x;
                swapBufferB[shIdx * 3 + 2] = (m_forward.sh[shIdx] * vec2(swapBufferA[shIdx * 3 + 2], 1.0)).x;
            }
            
            // Forward transform SH positions
            for (int shIdx = 0; shIdx < 3 * m_gridParams.coefficientsPerProbe; ++shIdx)
            {
                swapBufferA[shIdx] = swapBufferB[m_forward.coeffIdx[shIdx]];
            }

            // Write the probe positions to the transformed destination
            const int outputIdx = m_forward.probeIdx[probeIdx] * m_gridParams.coefficientsPerProbe;
            std::memcpy(&outputData[outputIdx], &swapBufferA[0], sizeof(float) * 3 * m_gridParams.coefficientsPerProbe);
        }
    }

    __host__ void LightProbeDataTransform::Inverse(const std::vector<vec3>& inputData, std::vector<vec3>& outputData) const
    {
        if (inputData.size() < m_inverse.probeIdx.size())
        {
            Log::Error("Cannot inverse transform probe grid data. Size of input data buffer is smaller the transform (%ix%ix%i).",
                m_gridParams.gridDensity.x, m_gridParams.gridDensity.y, m_gridParams.gridDensity.z);
            return;
        }
        
        std::vector<float> swapBufferA(3 * m_gridParams.coefficientsPerProbe);
        std::vector<float> swapBufferB(3 * m_gridParams.coefficientsPerProbe);

        for (int probeIdx = 0; probeIdx < m_gridParams.numProbes; ++probeIdx)
        {
            // Read in the block of SH coefficients
            const int inputIdx = probeIdx * m_gridParams.coefficientsPerProbe;
            std::memcpy(&swapBufferA[0], &inputData[inputIdx], sizeof(vec3) * m_gridParams.coefficientsPerProbe);

            // Invert transform SH positions
            for (int shIdx = 0; shIdx < 3 * m_gridParams.coefficientsPerProbe; ++shIdx)
            {
                swapBufferB[shIdx] = swapBufferA[m_inverse.coeffIdx[shIdx]];
            }
            
            // Inverse transform SH coefficients
            for (int shIdx = 0; shIdx < m_gridParams.coefficientsPerProbe; ++shIdx)
            {
                swapBufferA[shIdx * 3] = (m_inverse.sh[shIdx] * vec2(swapBufferB[shIdx * 3], 1.0)).x;
                swapBufferA[shIdx * 3 + 1] = (m_inverse.sh[shIdx] * vec2(swapBufferB[shIdx * 3 + 1], 1.0)).x;
                swapBufferA[shIdx * 3 + 2] = (m_inverse.sh[shIdx] * vec2(swapBufferB[shIdx * 3 + 2], 1.0)).x;
            }

            // Write the probe positions to the transformed destination
            const int outputIdx = m_inverse.probeIdx[probeIdx] * m_gridParams.coefficientsPerProbe;
            std::memcpy(&outputData[outputIdx], &swapBufferA[0], sizeof(vec3) * m_gridParams.coefficientsPerProbe);
        }
    }


}