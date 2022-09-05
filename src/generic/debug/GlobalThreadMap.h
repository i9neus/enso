#pragma once

#include <string>

void ClearThreadName(const std::string& name);
void SetThreadName(const std::string& name);
void AssertInThread(const std::string& name);