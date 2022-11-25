#include "math/Math.cuh"
#include "Asset.cuh"
#include "Hash.h"
#include "thirdparty/tinyformat/tinyformat.h"

namespace Enso
{    
    __host__ std::string Host::Asset::MakeTemporaryID()
    {
        static uint temporaryIdx = 0;
        return tfm::format("tempAsset_%x", HashOf(temporaryIdx++));
    }
}

