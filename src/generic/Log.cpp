#include "Log.h"

#ifdef MS_TEST_FRAMEWORK
#include "CppUnitTest.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
#endif

namespace Log
{
    struct GlobalState
    {
        GlobalState() :
            flags(kLogNormal | kLogWarning | kLogError | kLogCritical),
            verbosity(0),
            indentation(0),
            terminalOut(std::cout)
        {
        }

        std::ostream&           terminalOut;
        std::mutex              fileMutex;
        int32_t                 verbosity;
        uint32_t                flags;
        int                     indentation;
        Log::Snapshot           stats;
        std::set<std::string>   triggeredSet;
    }
    state;

    constexpr int32_t kMaxIndent = 5;
    constexpr int32_t kIndentChars = 3;

    Indent::Indent(const std::string& onIndent,
        const std::string& onRestore,
        const std::string& onException)
        : m_onRestore(onRestore),
        m_onException(onException)
    {
        if (!onIndent.empty()) { Write(onIndent); }

        m_logIndentation = (state.indentation < kMaxIndent) ?
            state.indentation++ :
            state.indentation;
    }

    Indent::~Indent()
    {
        Restore();
    }

    void Indent::Restore()
    {
        if (m_logIndentation < 0) { return; }

        state.indentation = m_logIndentation;
        m_logIndentation = -1;

        if (std::uncaught_exception() && !m_onException.empty())
        {
            WriteImpl(nullptr, -1, m_onException, kFgYellow, kLogWarning);
        }
        else if (!m_onRestore.empty())
        {
            Write(m_onRestore);
        }
    }

    Snapshot::Snapshot() : m_numMessages{} {}

    uint Snapshot::operator[](const int i) const
    {
        return (i < 0 || i >= kNumLogLevels) ? 0 : m_numMessages[i];
    }

    uint& Snapshot::operator[](const int i)
    {
        return m_numMessages[i];
    }

    Snapshot Snapshot::operator-(const Snapshot& rhs) const
    {
        Snapshot delta;
        for (int i = 0; i < kNumLogLevels; i++)
        {
            delta.m_numMessages[i] = m_numMessages[i] - rhs.m_numMessages[i];
        }
        return delta;
    }

    Snapshot GetMessageState()
    {
        return state.stats;
    }

    void EnableLevel(const uint32_t flag, const bool set)
    {
        if (set) { state.flags |= (1 << flag); }
        else { state.flags &= ~(1 << flag); }
    }

    void NL() { WriteImpl(nullptr, -1, "\n", kFgDefault, kLogNormal); }

    void WriteImpl(const char* file, const int line, const std::string& messageStr, const uint32_t colour, const LogLevel level)
    {
        if (messageStr.empty() || !(state.flags & (1 << level))) { return; }

        // If this message has an ID associated with it, it's once-only. Check to see if it's already been triggered and bail if so.
        if (file != nullptr)
        {
            const std::string id = tfm::format("%s:%i", file, line);
            if (state.triggeredSet.find(id) != state.triggeredSet.end()) { return; }
            state.triggeredSet.insert(id);
        }

        // Apply indentation
        std::string formattedStr;
        formattedStr.reserve(state.indentation * kIndentChars + messageStr.length());
        for (int32_t i = 0; i < state.indentation * kIndentChars; ++i)
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

#ifdef MS_TEST_FRAMEWORK
    // If this code is being invoked through the MS test framework, push the message to the logger
        Logger::WriteMessage(formattedStr.c_str());
#else
        state.terminalOut << "\033[" << colour << "m";
        state.terminalOut << formattedStr;
        state.terminalOut << "\033[" << kFgDefault << "m\033[" << kBgDefault << "m";
        state.terminalOut.flush();
#endif

        state.stats[level]++;
    }
}
