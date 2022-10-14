#pragma once

#include "SuiteBase.h"

namespace Cuda
{
    class LineSegment;
    namespace Host
    {
        template<typename T> class Vector;
        class BIH2DAsset;
    }
}

namespace Tests
{
    class CudaBIH2DTestsImpl : public SuiteBase
    {
    public:
        __host__ CudaBIH2DTestsImpl() = default;

        __host__ void BuildSimpleGeometry();

        __host__ void PointTestSimpleGeometry();
        __host__ void RayTestSimpleGeometry();
        __host__ void RayTestRandomGeometry();

    private:
        __host__ void CreateCircleSegments(Cuda::Host::Vector<Cuda::LineSegment>& segments);
        __host__ void CreateRowSegments(Cuda::Host::Vector<Cuda::LineSegment>& segments);

        __host__ void BuildBIH(Cuda::AssetHandle<Host::BIH2DAsset>& bih, Cuda::Host::Vector<Cuda::LineSegment>& segments, const bool printStats);
    };
}
