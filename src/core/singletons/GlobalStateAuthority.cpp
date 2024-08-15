#include "GlobalStateAuthority.h"
#include "io/FilesystemUtils.h"

namespace Enso
{
    GlobalStateAuthority::GlobalStateAuthority()
    {
        try
        {
            Log::Write("Looking for config.json...\n");

            m_configJson.Deserialise("config.json");
            m_configJson.GetValue("scene", m_defaultScenePath, Json::kRequiredAssert);

            m_defaultSceneDirectory = GetParentDirectory(m_defaultScenePath);
        }
        catch (const std::runtime_error& err)
        {
            throw std::runtime_error(tfm::format("Unable to initialise global state authority: %s", err.what()));
        }
    }

    GlobalStateAuthority& GlobalStateAuthority::Get()
    {
        static GlobalStateAuthority singleton;
        return singleton;
    }
}