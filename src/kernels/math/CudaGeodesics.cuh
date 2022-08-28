#include "CudaMath.cuh"
#include "generic/Log.h"

#include <vector>

namespace Cuda
{
    enum GeodesicClass : int
    {
        kGeoClassTetrahedron,
        kGeoClassIcosahedron
    };

    __host__ void GenerateGeodesicDistribution(std::vector<vec3>& output, const GeodesicClass geoClass, int numIterations);
}