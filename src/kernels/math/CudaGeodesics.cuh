#include "CudaMath.cuh"

namespace Cuda
{
    enum GeodesicClass : int
    {
        kGeoClassTetrahedron,
        kGeoClassIcosahedron
    };

    __host__ void GenerateGeodesicDistribution(std::vector<vec3>& output, const GeodesicClass geoClass, int numIterations);
}