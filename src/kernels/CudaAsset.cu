#include "CudaAsset.cuh"
#include "generic/Hash.h"
#include "thirdparty/tinyformat/tinyformat.h"
#include <map>

namespace Cuda
{    
    __host__ std::string Host::Asset::MakeTemporaryID()
    {
        static uint temporaryIdx = 0;
        return tfm::format("tempAsset_%x", HashOf(temporaryIdx++));
    }
}

