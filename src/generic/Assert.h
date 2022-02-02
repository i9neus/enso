#pragma once

#ifdef DISABLE_ASSERTS

#define Assert(condition, message)
#define AssertMsg(condition, message, ...)
#define AssertRethrow(condition, message)

#else

#include "Backtrace.h"

#ifdef DISABLE_BACKTRACE_ON_ASSERT
#define _AssertCacheStackBacktrace
#else
#define _AssertCacheStackBacktrace StackBacktrace::Cache()
#endif

#define Assert(condition) \
        if(!(condition)) {  \
            _AssertCacheStackBacktrace; \
            throw std::runtime_error(tfm::format("%s in %s (%d)", #condition, __FILE__, __LINE__)); \
        }

#define AssertMsg(condition, message) \
        if(!(condition)) {  \
            _AssertCacheStackBacktrace; \
            throw std::runtime_error(tfm::format("%s in %s (%d)", message, __FILE__, __LINE__)); \
        }

#define AssertMsgFmt(condition, message, ...) \
        if(!(condition)) {  \
            _AssertCacheStackBacktrace; \
            char buffer[1024]; \
            std::snprintf(buffer, 1024, message, __VA_ARGS__); \
            throw std::runtime_error(tfm::format("%s in %s (%d)", buffer, __FILE__, __LINE__)); \
        }

#define AssertRethrow(condition, message) \
        try \
        { \
            if(!(condition)) {  \
                throw std::runtime_error(""); \
            } \
        } \
        catch(std::runtime_error& e) { \
            throw std::runtime_error(tfm::format("%s in %s (%d)", buffer, __FILE__, __LINE__)); \
        }
#endif