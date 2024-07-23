#include "../CudaHeaders.cuh"
#include <unordered_map>

#define ENABLE_CUDA_PROFILER

namespace Enso
{
    namespace CudaProfiler
    {
#ifdef ENABLE_CUDA_PROFILER
        __host__ void                   Clear();
        __host__ bool                   Flush();
        __host__ void                   Register(const std::string& id);
        __host__ void                   Report();
        __host__ void                   Poll(const std::string& id);
#else
        __host__ bool                   Flush() { return true; }
        __host__ void                   Register(const std::string&) {}
        __host__ void                   Report() {}
        __host__ void                   Poll(const std::string&) {}
#endif
    };
}