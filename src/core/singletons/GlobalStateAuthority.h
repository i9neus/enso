#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>
#include <array>
#include "core/math/Constants.h"
#include "thirdparty/tinyformat/tinyformat.h"

#include "io/json/JsonUtils.h"

namespace Enso
{
    class GlobalStateAuthority
    {
    public:
        const Json::Document& GetConfigJson() { return m_configJson; }
        const std::string& GetDefaultScenePath() const { return m_defaultScenePath; }
        const std::string& GetDefaultSceneDirectory() const { return m_defaultSceneDirectory; }

        static GlobalStateAuthority& Get();

    private:
        GlobalStateAuthority();

    private:
        Json::Document m_configJson;

        std::string m_defaultScenePath;
        std::string m_defaultSceneDirectory;
    };

    inline GlobalStateAuthority& GSA() { return GlobalStateAuthority::Get(); }
}