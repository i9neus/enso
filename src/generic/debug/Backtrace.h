#pragma once

#include <string>
#include <Windows.h>

namespace StackBacktrace
{
    void                        Clear();
    void                        Cache();
    std::vector<std::string>    Get();
}
    //std::string                 ExceptionBacktrace(_EXCEPTION_POINTERS* exPtrs);