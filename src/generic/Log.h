#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>
#include <array>
#include <set>
#include "Constants.h"
#include "thirdparty/tinyformat/tinyformat.h"

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

class Log
{
public:
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

    class Snapshot
    {
    private:
        std::array<uint, kNumLogLevels>               m_numMessages;

    public:
        Snapshot() : m_numMessages{}
        {}

        uint operator[](const int i) const 
        { 
            return (i < 0 || i >= kNumLogLevels) ? 0 : m_numMessages[i]; 
        }
        uint& operator[](const int i)
        {
            return m_numMessages[i];
        }

        Snapshot operator-(const Snapshot& rhs) const
        {
            Snapshot delta;
            for (int i = 0; i < kNumLogLevels; i++)
            {
                delta.m_numMessages[i] = m_numMessages[i] - rhs.m_numMessages[i];
            }
            return delta;
        }
    };

public:
    static constexpr int32_t kMaxIndent = 5;
    static constexpr int32_t kIndentChars = 3;

    static Log& Get();
    static void NL();   
    static Snapshot GetMessageState();

    void EnableLevel(const uint32_t flags, bool set);

#define LOG_TYPE(Name, Colour, Type) template<typename... Args> \
                                     static inline void Name(const std::string& message, const Args&... args) { Log::StaticWrite(nullptr, -1, tfm::format(message.c_str(), args...), Colour, Type); } \
                                     static void Name(const std::string& message) { Log::StaticWrite("", -1, message, Colour, Type); }

#define LOG_TYPE_ONCE(Name, Colour, Type) template<typename... Args> \
                                     static inline void Name(const std::string& message, const Args&... args) { Log::StaticWrite(__FILE__, __LINE__, tfm::format(message.c_str(), args...), Colour, Type); } \
                                     static void Name(const std::string& message) { Log::StaticWrite(__FILE__, __LINE__, message, Colour, Type); }

    LOG_TYPE(Write, kFgDefault, kLogNormal)
    LOG_TYPE(Debug, kFgGreen, kLogDebug)
    LOG_TYPE(Warning, kFgYellow, kLogWarning)
    LOG_TYPE(Error, kBgRed, kLogError)
    LOG_TYPE(System, kFgTeal, kLogSystem)

    LOG_TYPE_ONCE(WriteOnce, kFgDefault, kLogNormal)
    LOG_TYPE_ONCE(DebugOnce, kFgGreen, kLogDebug)
    LOG_TYPE_ONCE(WarningOnce, kFgYellow, kLogWarning)
    LOG_TYPE_ONCE(ErrorOnce, kBgRed, kLogError)
    LOG_TYPE_ONCE(SystemOnce, kFgTeal, kLogSystem)

private:
    Log();
    ~Log();

    static void StaticWrite(const char* file, const int line, const std::string& formatted, const uint32_t colour, const LogLevel level);
    void WriteImpl(const char* file, const int line, const std::string& formatted, const uint32_t colour, const LogLevel level);

    std::ostream&       m_logTerminalOut;
    std::mutex          m_logFileMutex;
    int32_t             m_logIndentation;
    int32_t             m_logVerbosity;
    uint32_t            m_logFlags;
    
    Snapshot            m_stats;

    std::set<std::string> m_triggeredSet; 
};