#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>

enum LogLevel : uint32_t { kLogDebug, kLogNormal, kLogWarning, kLogError, kLogCritical };

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

public:
    Log();
    ~Log();

    static constexpr int32_t kMaxIndent = 5;
    static constexpr int32_t kIndentChars = 3;

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

private:
    static Log& Singleton();

    void WriteImpl(const std::string& formatted, const uint32_t colour, const LogLevel level);

    std::ostream&       m_logTerminalOut;
    std::mutex          m_logFileMutex;
    int32_t             m_logIndentation;
    int32_t             m_logVerbosity;
};