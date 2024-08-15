#pragma once

#include <chrono>
#include <functional>
#include "io/Log.h"

namespace Enso
{
    class HighResolutionTimer
    {
    public:
        HighResolutionTimer() : m_startTime(std::chrono::high_resolution_clock::now()) {}

        HighResolutionTimer(const char* msg) :
            m_message(msg),
            m_startTime(std::chrono::high_resolution_clock::now()) { }

        HighResolutionTimer(std::function<std::string(float)> lambda) :
            m_lambda(lambda),
            m_startTime(std::chrono::high_resolution_clock::now()) {}

        ~HighResolutionTimer()
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

        inline float Get() const
        {
            return float(std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_startTime).count());
        }

        inline void Reset()
        {
            m_startTime = std::chrono::high_resolution_clock::now();
        }

        inline void Write(const std::string& format) const
        {
            Log::Debug(format, Get());
        }

    private:
        const std::string m_message;
        std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
        std::function<std::string(float)> m_lambda;
    };
}
