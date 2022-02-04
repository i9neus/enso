#include "Log.h"

Log::Indent::Indent(const std::string& onIndent,
    const std::string& onRestore,
    const std::string& onException)
    : m_onRestore(onRestore),
    m_onException(onException)
{
    if (!onIndent.empty()) { Get().Write(onIndent); }

    m_logIndentation = (Get().m_logIndentation < kMaxIndent) ?
        Get().m_logIndentation++ :
        Get().m_logIndentation;
}

Log::Indent::~Indent()
{
    Restore();
}

void Log::Indent::Restore()
{
    if (m_logIndentation < 0) { return; }

    Get().m_logIndentation = m_logIndentation;
    m_logIndentation = -1;

    if (std::uncaught_exception() && !m_onException.empty())
    {
        Get().WriteImpl(m_onException, kFgYellow, kLogWarning);
    }
    else if (!m_onRestore.empty())
    {
        Get().Write(m_onRestore);
    }
}

Log::Log() :
    m_logTerminalOut(std::cout),
    m_logFlags(0)
{
    EnableLevel(kLogNormal, true);
    EnableLevel(kLogWarning, true);
    EnableLevel(kLogError, true);
    EnableLevel(kLogCritical, true);
}

Log::~Log() { }

Log::Snapshot Log::GetMessageState()
{
    return Get().m_stats;
}

void Log::EnableLevel(const uint32_t flag, const bool set)
{
    if (set) { m_logFlags |= (1 << flag); }
    else { m_logFlags &= ~(1 << flag); }
}

void Log::NL() { Get().WriteImpl("\n", kFgDefault, kLogNormal); }

void Log::StaticWrite(const std::string& message, const uint32_t colour, const LogLevel level)
{
    Get().WriteImpl(message, colour, level);
}

Log& Log::Get()
{
    static Log state;
    return state;
}

void Log::WriteImpl(const std::string& messageStr, const uint32_t colour, const LogLevel level)
{
    if (messageStr.empty() || !(m_logFlags & (1 << level))) { return; }
    
    // Apply indentation
    std::string formattedStr;
    formattedStr.reserve(m_logIndentation * kIndentChars + messageStr.length());
    for (int32_t i = 0; i < m_logIndentation * kIndentChars; ++i)
    {
        formattedStr += ' ';
    }
    formattedStr += messageStr; 
    
    bool addNewline = true;
    switch (formattedStr.back())
    {        
    case '\b':
        formattedStr.pop_back();
    case '\n':
        addNewline = false;
        break;
    };     

    // Always add a newline unless there's an excape character
    if (addNewline) { formattedStr += '\n'; }

    // Lock
    //std::lock_guard<std::mutex> lock(m_logFileMutex);

    //std::printf("\033[%im%s\033[%im\033[%im", colour, formattedStr.c_str(), kFgDefault, kBgDefault);

    m_logTerminalOut << "\033[" << colour << "m";
    m_logTerminalOut << formattedStr;
    m_logTerminalOut << "\033[" << kFgDefault << "m\033[" << kBgDefault << "m";
    m_logTerminalOut.flush();

    m_stats[level]++;
}
