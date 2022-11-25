#pragma once

#include <vector>
#include <string>

namespace StackBacktrace
{
    void                        Clear();
    void                        Cache();
    std::vector<std::string>    Get();
    void                        Print();
}
    //std::string                 ExceptionBacktrace(_EXCEPTION_POINTERS* exPtrs);