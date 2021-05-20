#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

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

#include "thirdparty/tinyformat/tinyformat.h"

using Microsoft::WRL::ComPtr;

#define ASSERTS

#ifdef ASSERTS

#define Assert(condition) \
        if(!(condition)) {  \
            throw std::runtime_error(tfm::format("%s File %s, line %d.", #condition, __FILE__, __LINE__)); \
        }

#define AssertMsg(condition, message) \
        if(!(condition)) {  \
            throw std::runtime_error(tfm::format("%s File %s, line %d.", message, __FILE__, __LINE__)); \
        }

#define AssertMsgFmt(condition, message, ...) \
        if(!(condition)) {  \
            char buffer[1024]; \
            std::snprintf(buffer, 1024, message, __VA_ARGS__); \
            throw std::runtime_error(tfm::format("%s File %s, line %d.", buffer, __FILE__, __LINE__)); \
        }

#define AssertRethrow(condition, message) \
        try \
        { \
            if(!(condition)) {  \
                throw std::runtime_error(""); \
            } \
        } \
        catch(std::runtime_error& e) { \
            throw std::runtime_error(tfm::format("%s File %s, line %d.", buffer, __FILE__, __LINE__)); \
        }

#else
#define Assert(condition, message)
#define AssertMsg(condition, message, ...)
#define AssertRethrow(condition, message)
#endif