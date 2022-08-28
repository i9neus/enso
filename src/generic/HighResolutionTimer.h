#pragma once

#include <chrono>
#include <functional>
#include "generic/Log.h"

class HighResolutionTimer
{
public:
    HighResolutionTimer();
    HighResolutionTimer(const char* msg);
    HighResolutionTimer(std::function<std::string(float)> lambda);
    ~HighResolutionTimer();

    inline float Get() const;
    inline void Reset();
    inline void Write(const std::string& format) const;

private:
    const std::string m_message;
    std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
    std::function<std::string(float)> m_lambda;
};