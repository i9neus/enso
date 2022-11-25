#include "Math.cuh"
#include "io/Log.h"

#include <vector>

namespace Enso
{
    enum GeodesicClass : int
    {
        kGeoClassTetrahedron,
        kGeoClassIcosahedron
    };

    __host__ void GenerateGeodesicDistribution(std::vector<vec3>& output, const GeodesicClass geoClass, int numIterations);
}