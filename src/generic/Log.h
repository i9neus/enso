#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>
#include <array>
#include "Constants.h"

enum LogLevel : uint32_t { kLogDebug = 0, kLogNormal, kLogWarning, kLogError, kLogCritical, kNumLogLevels };

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
    Log();
    ~Log();

    static constexpr int32_t kMaxIndent = 5;
    static constexpr int32_t kIndentChars = 3;

    static void NL();

    template<typename... Args>
    static inline void Write(const std::string& message, const Args&... args) { Log::Write(tfm::format(message.c_str(), args...)); }
    static void Write(const std::string& message);

    template<typename... Args>
    static inline void Debug(const std::string& message, const Args&... args)  { Log::Debug(tfm::format(message.c_str(), args...)); }
    static void Debug(const std::string& message);

    template<typename... Args>
    static inline void Warning(const std::string& message, const Args&... args) { Log::Warning(tfm::format(message.c_str(), args...)); }
    static void Warning(const std::string& message);

    template<typename... Args>
    static inline void Error(const std::string& message, const Args&... args) { Log::Error(tfm::format(message.c_str(), args...)); }
    static void Error(const std::string& message);

    static Snapshot GetMessageState();

private:
    static Log& Singleton();

    void WriteImpl(const std::string& formatted, const uint32_t colour, const LogLevel level);

    std::ostream&       m_logTerminalOut;
    std::mutex          m_logFileMutex;
    int32_t             m_logIndentation;
    int32_t             m_logVerbosity;
    
    Snapshot            m_stats;
};