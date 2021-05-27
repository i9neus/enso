#pragma once

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