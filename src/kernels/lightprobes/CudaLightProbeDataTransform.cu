#include "CudaLightProbeDataTransform.cuh"

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
    
    __host__ void GenerateSHIndices(const LightProbeGridParams& gridParams, LightProbeDataTransform& transform)
    {
        // Swizzle the indices for the L1 RGB coefficients
        std::vector<int> shSwiz(gridParams.coefficientsPerProbe);
        for (int i = 0; i < shSwiz.size(); ++i) { shSwiz[i] = i; }
        shSwiz[1] = SwizzleIndices(gridParams.shSwizzle)[0] + 1;
        shSwiz[2] = SwizzleIndices(gridParams.shSwizzle)[1] + 1;
        shSwiz[3] = SwizzleIndices(gridParams.shSwizzle)[2] + 1;

        // Convert the coefficient offsets into per-channel coefficients
        // E.g. { 0, 2, 1 } -> { 0, 1, 2, 6, 7, 8, 3, 4, 5 }
        for (int i = 0; i < transform.forward.coeffIdx.size(); ++i)
        {
            transform.forward.coeffIdx[i] = shSwiz[i / 3] + i % 3;
        }

        // Unity coefficient packing. L0 remains RGB, but L1 is grouped by R, G and B
        auto& f = transform.forward.coeffIdx;
        auto g = f;
        g[0] = f[1];   g[1] = f[1];   g[2] = f[2];      // L0
        g[3] = f[3];   g[4] = f[6];   g[5] = f[9];      // L1_1
        g[6] = f[4];   g[7] = f[7];   g[8] = f[10];     // L10
        g[9] = f[5];   g[10] = f[8];  g[11] = f[11];    // L11
        g[12] = f[12]; g[13] = f[13]; g[14] = f[14];    // Metadata
        f = g;

        // Derive the inverse coefficient mapping
        for (int i = 0; i < transform.forward.coeffIdx.size(); ++i)
        {
            g[transform.forward.coeffIdx[i]] = i;
        }
        transform.inverse.coeffIdx = g;
    }

    __host__ void GenerateSHTransforms(const LightProbeGridParams& gridParams, LightProbeDataTransform& transform)
    {
        // Factors for inverting L1 SH coefficients
        std::vector<float> shDirs(gridParams.coefficientsPerProbe, 1.0f);
        if (gridParams.shInvertX) { shDirs[1] = -1.0f; }
        if (gridParams.shInvertY) { shDirs[2] = -1.0f; }
        if (gridParams.shInvertZ) { shDirs[3] = -1.0f; }

        // Initialise the forward transformation with the directional scaling factors
        auto& fwdSh = transform.forward.sh;
        for (int i = 0; i < transform.forward.coeffIdx.size(); ++i)
        {
            fwdSh[i] = mat2(shDirs[i], 0.0, 0.0, 1.0);
        }

        // Unity coefficient rescaling
        fwdSh[0].i00 *= SH::Legendre(0);
        fwdSh[1].i00 *= SH::Legendre(1);
        fwdSh[2].i00 *= SH::Legendre(1);
        fwdSh[3].i00 *= SH::Legendre(1);
        fwdSh[4] = mat2(-1.0, 0.0, 0.0, 1.0);

        // Invert the forward transformation
        auto& invSh = transform.inverse.sh;
        for (int i = 0; i < invSh.size(); ++i)
        {
            invSh[i] = inverse(fwdSh[i]);
        }
    }

    __host__ void GenerateProbePositionIndices(const LightProbeGridParams& gridParams, LightProbeDataTransform& transform)
    {
        // Generate swizzle indices for probe positions
        const ivec3 posSwiz = SwizzleIndices(gridParams.posSwizzle);

        // Swizzle the grid density
        const ivec3 swizzledGridDensity = SwizzleVector(gridParams.gridDensity, posSwiz);

        // Swizzle the axes
        for (int probeIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx)
        {
            ivec3 gridPos = GridPosFromProbeIdx(probeIdx, gridParams.gridDensity);

            // Invert the axes where appropriate
            if (gridParams.posInvertX) { gridPos.x = gridParams.gridDensity.x - gridPos.x - 1; }
            if (gridParams.posInvertY) { gridPos.y = gridParams.gridDensity.y - gridPos.y - 1; }
            if (gridParams.posInvertZ) { gridPos.z = gridParams.gridDensity.z - gridPos.z - 1; }

            // Swizzle the grid index
            ivec3 swizzledGridPos = SwizzleVector(gridPos, posSwiz);

            // Map back onto the data array
            const uint swizzledProbeIdx = ProbeIdxFromGridPos(swizzledGridPos, swizzledGridDensity);
            Assert(swizzledProbeIdx < gridParams.numProbes);

            // Store the indirections
            transform.forward.probeIdx[probeIdx] = swizzledProbeIdx;
            transform.inverse.probeIdx[swizzledProbeIdx] = probeIdx;
        }
    }

    __host__ LightProbeDataTransform GenerateLightProbeDataTransform(const LightProbeGridParams& gridParams)
    {
        // Forward transformation operations are applied in this order:
        //   1. Swizzle probe positions
        //   2. Transform coefficients
        //   3. Swizzle coefficients

        LightProbeDataTransform transform(gridParams);          

        GenerateProbePositionIndices(gridParams, transform);
        GenerateSHIndices(gridParams, transform);
        GenerateSHTransforms(gridParams, transform);

        return transform;
    }
}