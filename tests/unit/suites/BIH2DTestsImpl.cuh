#pragma once

#include "SuiteBase.h"

namespace Enso
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
    class BIH2DTestsImpl : public SuiteBase
    {
    public:
        __host__ BIH2DTestsImpl() = default;

        __host__ void BuildSimpleGeometry();

        __host__ void PointTestSimpleGeometry();
        __host__ void RayTestSimpleGeometry();
        __host__ void RayTestRandomGeometry();

    private:
        __host__ void CreateCircleSegments(Host::Vector<LineSegment>& segments);
        __host__ void CreateRowSegments(Host::Vector<LineSegment>& segments);

        __host__ void BuildBIH(AssetHandle<Host::BIH2DAsset>& bih, Host::Vector<LineSegment>& segments, const bool printStats);
    };
}
