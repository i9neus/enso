#pragma once

#include <string>

namespace Enso
{
    void ClearThreadName(const std::string& name);
    void SetThreadName(const std::string& name);
    void AssertInThread(const std::string& name);
}