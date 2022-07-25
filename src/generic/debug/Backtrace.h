#pragma once

#include <string>
#include <Windows.h>

namespace StackBacktrace
{
    void                        Clear();
    void                        Cache();
    std::vector<std::string>    Get();
    void                        Print();
}
    //std::string                 ExceptionBacktrace(_EXCEPTION_POINTERS* exPtrs);