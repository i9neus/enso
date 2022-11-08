#include "HighResolutionTimer.h"

HighResolutionTimer::HighResolutionTimer() : m_startTime(std::chrono::high_resolution_clock::now()) {}

HighResolutionTimer::HighResolutionTimer(const char* msg) :
    m_message(msg),
    m_startTime(std::chrono::high_resolution_clock::now()) { }

HighResolutionTimer::HighResolutionTimer(std::function<std::string(float)> lambda) :
    m_lambda(lambda),
    m_startTime(std::chrono::high_resolution_clock::now()) {}

HighResolutionTimer::~HighResolutionTimer()
{
    if (m_lambda)
    {
        Log::Debug("%s\n", m_lambda(Get()).c_str()); 
    }
    else if (!m_message.empty())
    {
        Log::Debug("%s\n", tfm::format(m_message.c_str(), Get()).c_str());
    }
}

inline float HighResolutionTimer::Get() const
{
    return float(std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_startTime).count());
}

inline void HighResolutionTimer::Reset()
{
    m_startTime = std::chrono::high_resolution_clock::now();
}

inline void HighResolutionTimer::Write(const std::string& format) const
{
    Log::Debug(format, Get());
}