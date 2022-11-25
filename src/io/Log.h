#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>
#include <array>
#include <set>
#include "core/math/Constants.h"
#include "thirdparty/tinyformat/tinyformat.h"

namespace Enso
{
    enum LogLevel : uint32_t
    {
        kLogDebug = 0,
        kLogNormal,
        kLogWarning,
        kLogError,
        kLogCritical,
        kLogSystem,
        kNumLogLevels
    };

    namespace Log
    {
        enum ANSIColourCode : uint32_t
        {
            kFgBlack = 30,
            kFgRed = 31,
            kFgGreen = 32,
            kFgYellow = 33,
            kFgBlue = 34,
            kFgPurple = 35,
            kFgTeal = 36,
            kFgWhite = 37,
            kFgDefault = 39,
            kBgRed = 41,
            kBgGreen = 42,
            kBgYellow = 43,
            kBgBlue = 44,
            kBgPurple = 45,
            kBgTeal = 46,
            kBgWhite = 47,
            kBgDefault = 49
        };

        // Little class to indent the log and un-indent automatically on destruction
        class Indent
        {
        public:
            Indent(const std::string& onIndent = "",
                const std::string& onRestore = "",
                const std::string& onException = "");
            ~Indent();
            void Restore();
        private:
            int32_t             m_logIndentation;
            const std::string   m_onRestore;
            const std::string   m_onException;
        };

        // Keeps track of the number of errors and warnings
        class Snapshot
        {
        private:
            std::array<uint, kNumLogLevels>               m_numMessages;
        public:
            Snapshot();
            uint operator[](const int i) const;
            uint& operator[](const int i);
            Snapshot operator-(const Snapshot& rhs) const;
        };

        void        NL();
        Snapshot    GetMessageState();
        void        EnableLevel(const uint32_t flags, bool set);
        void        WriteImpl(const char* file, const int line, const std::string& formatted, const uint32_t colour, const LogLevel level);

#define LOG_TYPE(Name, Colour, Type) template<typename... Args> \
                                     inline void Name(const std::string& message, const Args&... args) { WriteImpl(nullptr, -1, tfm::format(message.c_str(), args...), Colour, Type); } \
                                     inline void Name(const std::string& message) { WriteImpl(nullptr, -1, message, Colour, Type); }

#define LOG_TYPE_ONCE(Name, Colour, Type) template<typename... Args> \
                                     inline void Name(const std::string& message, const Args&... args) { WriteImpl(__FILE__, __LINE__, tfm::format(message.c_str(), args...), Colour, Type); } \
                                     inline void Name(const std::string& message) { WriteImpl(__FILE__, __LINE__, message, Colour, Type); }

        LOG_TYPE(Write, kFgDefault, kLogNormal)
            LOG_TYPE(Success, kFgGreen, kLogNormal)
            LOG_TYPE(Debug, kFgBlue, kLogDebug)
            LOG_TYPE(Warning, kFgYellow, kLogWarning)
            LOG_TYPE(Error, kBgRed, kLogError)
            LOG_TYPE(System, kFgTeal, kLogSystem)

            LOG_TYPE_ONCE(WriteOnce, kFgDefault, kLogNormal)
            LOG_TYPE_ONCE(SuccessOnce, kFgGreen, kLogNormal)
            LOG_TYPE_ONCE(DebugOnce, kFgBlue, kLogDebug)
            LOG_TYPE_ONCE(WarningOnce, kFgYellow, kLogWarning)
            LOG_TYPE_ONCE(ErrorOnce, kBgRed, kLogError)
            LOG_TYPE_ONCE(SystemOnce, kFgTeal, kLogSystem)
    }
}