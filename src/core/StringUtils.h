#pragma once

#include "Lexer.h"

namespace Enso
{
    // Converts an 8-bit ASCII string into a 16-bit wide unicode string
    std::wstring Widen(const std::string& str);

    // Capitalises the first letter of the input string
    std::string CapitaliseFirst(const std::string& input);

    // Removes training whitespace from the end of a string
    void ClipTrailingWhitespace(std::string& input);

    // Makes a string lowercase
    void MakeLowercase(std::string& input);

    // Returns a lowercase copy of a string
    std::string Lowercase(const std::string& input);

    // Checks to see whether a string is all lowercase
    bool IsLowercase(const std::string& input);
    
    // Makes a string uppercase
    void MakeUppercase(std::string& input);

    // Returns an uppercase copy of a string
    std::string Uppercase(const std::string& input);
    
    // Checks to see whether a string is all uppercase
    bool IsUppercase(const std::string& input);

    // Formats a floating point value with digit grouping to dp decimal places
    std::string FormatPrettyFloat(const float value, int dp);

    // Formats a time value in seconds in the style hh:mm::ss.xxx
    std::string FormatElapsedTime(const float time);

    // Formats a data size in using the nearest denomination of B, kB, MB, GB
    std::string FormatDataSize(const float inputMB, const int dp);    
}