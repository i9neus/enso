#include "CudaLambert.cuh"

namespace Cuda
{
    __host__ Host::LambertBRDF::LambertBRDF() : 
        cu_deviceData(nullptr)
    {
        InstantiateOnDevice(&cu_deviceData);
    }
    
    __host__ void Host::LambertBRDF::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}