#include "core/math/Math.cuh"
#include "Asset.cuh"
#include "core/math/Hash.cuh"
#include "thirdparty/tinyformat/tinyformat.h"
#include "io/json/JsonUtils.h"

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

    __host__ std::string Host::Asset::GetAssetDAGPath() const
    {
        if (!m_parentAssetHandle.expired())
        {
            // If this asset has a valid parent, get its DAG path and append this asset's ID onto it
            return m_parentAssetHandle.lock()->GetAssetDAGPath() + Json::Node::kDAGDelimiter + GetAssetID();
        }
        else
        {
            return GetAssetID();
        }
    }

    __host__ std::string Host::Asset::GetAssetStem() const
    {
        const size_t delimPos = m_assetId.find_last_of(GetDAGPathDelimeter());
        return (delimPos == std::string::npos) ? m_assetId : m_assetId.substr(delimPos + 1);
    }
}

