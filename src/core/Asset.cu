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

    __host__ std::string Host::Asset::GetParentAssetID() const
    {
        AssertMsgFmt(!m_parentAssetHandle.expired(), "The parent asset belonging to '%s' is invalid. It may not exist or may have been cleaned up.", m_assetId.c_str());

        return m_parentAssetHandle.lock()->GetAssetID();
    }
}

