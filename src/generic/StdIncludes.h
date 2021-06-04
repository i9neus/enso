#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#define NOMINMAX

#include <Windows.h>

#include <string>
#include <vector>
#include <wrl.h>
#include <shellapi.h>
#include <thread>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <condition_variable>
#include "Assert.h"
//#include <unmap>
#include <unordered_map>

#include "thirdparty/tinyformat/tinyformat.h"
#include "Assert.h"

using Microsoft::WRL::ComPtr;

template<typename T> inline void echo(const char* str) { std::printf("%s\n", str); }

class Timer
{
public:
    Timer() : m_startTime(std::chrono::high_resolution_clock::now()) {}
    Timer(const char* msg) :
        m_message(msg),
        m_startTime(std::chrono::high_resolution_clock::now()) { }

    Timer(std::function<std::string(float)> lambda) :
        m_lambda(lambda),
        m_startTime(std::chrono::high_resolution_clock::now()) {}

    inline float Get() const 
    {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - m_startTime).count();
    }

    ~Timer()
    {
        
        if (m_lambda)
        {
            std::printf("%s\n", m_lambda(Get()).c_str());
        }
        else if(!m_message.empty())
        {
            std::printf("%s\n", tfm::format(m_message.c_str(), Get()).c_str());
        }
    }

private:
    const std::string m_message;
    const std::chrono::time_point<std::chrono::high_resolution_clock>  m_startTime;
    std::function<std::string(float)> m_lambda;
};