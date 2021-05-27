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
#include "Assert.h"
//#include <unmap>
#include <unordered_map>

#include "thirdparty/tinyformat/tinyformat.h"
#include "Assert.h"

using Microsoft::WRL::ComPtr;

template<typename T> inline void echo(const char* str) { std::printf("%s\n", str); }

