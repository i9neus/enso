#include "CudaLambert.cuh"

namespace Cuda
{
    __host__ Host::LambertBRDF::LambertBRDF() : 
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::LambertBRDF>();
    }
    
    __host__ void Host::LambertBRDF::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}