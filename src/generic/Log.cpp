#include "Log.h"

Log::Indent::Indent(const std::string& onIndent,
    const std::string& onRestore,
    const std::string& onException)
    : m_onRestore(onRestore),
    m_onException(onException)
{
    if (!onIndent.empty()) { Singleton().Write(onIndent); }

    m_logIndentation = (Singleton().m_logIndentation < kMaxIndent) ?
        Singleton().m_logIndentation++ :
        Singleton().m_logIndentation;
}

Log::Indent::~Indent()
{
    Restore();
}

void Log::Indent::Restore()
{
    if (m_logIndentation < 0) { return; }

    Singleton().m_logIndentation = m_logIndentation;
    m_logIndentation = -1;

    if (std::uncaught_exception() && !m_onException.empty())
    {
        Singleton().WriteImpl(m_onException, kFgYellow, kLogWarning);
    }
    else if (!m_onRestore.empty())
    {
        Singleton().Write(m_onRestore);
    }
}

Log::Log() :
    m_logTerminalOut(std::cout)
{
}

Log::~Log() { }

void Log::Write(const std::string& message)
{
    Singleton().WriteImpl(message, kFgDefault, kLogNormal);
}

void Log::Debug(const std::string& message)
{
    Singleton().WriteImpl(message, kFgGreen, kLogDebug);
}

void Log::Warning(const std::string& message)
{
    Singleton().WriteImpl(message, kFgYellow, kLogWarning);
}

void Log::Error(const std::string& message)
{
    Singleton().WriteImpl(message, kBgRed, kLogError);
}

Log& Log::Singleton()
{
    static Log state;
    return state;
}

void Log::WriteImpl(const std::string& messageStr, const uint32_t colour, const LogLevel level)
{
    std::string sanitisedStr, formattedStr;
    bool carriageReturn = false, newLine = false;

    // Scrub new-lines and carriage returns from the string
    for (auto& c : messageStr)
    {
        if (c == '\r') { carriageReturn = true; }
        else if (c == '\n') { newLine = true; }
        else { sanitisedStr += c; }
    }

    auto repeat = [&](char c, int32_t t) { for (int32_t i = 0; i < t; ++i) { formattedStr += c; } };
    auto indent = [&]() { for (int32_t i = 0; i < m_logIndentation * kIndentChars; ++i) { formattedStr += ' '; } };

    if (carriageReturn) { formattedStr += '\r'; } // CR

    indent(); // Indent

    formattedStr += sanitisedStr; // Add the sanitised message string

    if (newLine) { formattedStr += '\n'; } // NL

    // Lock
    std::lock_guard<std::mutex> lock(m_logFileMutex);

    m_logTerminalOut << "\033[" << colour << "m";
    m_logTerminalOut << formattedStr;
    m_logTerminalOut << "\033[" << kFgDefault << "m";  
    m_logTerminalOut.flush();
}
