#include "StringUtils.h"
#include <cstdlib>
#include <clocale>
#include <math.h>

namespace Enso
{
    std::wstring Widen(const std::string& mbstr)
    {
        if (mbstr.empty()) { return std::wstring(); }

        std::setlocale(LC_ALL, "en_US.utf8");
        wchar_t* wstr = new wchar_t[mbstr.length() + 1];
        std::mbstowcs(wstr, mbstr.c_str(), mbstr.length() + 1);
        std::wstring stdwstr(wstr);
        delete[] wstr;

        return stdwstr;
    }

    // Capitalises the first letter of the input string
    std::string CapitaliseFirst(const std::string& input)
    {
        std::string copyInput = input;
        copyInput[0] = std::toupper(copyInput[0]);
        return copyInput;
    }

    void ClipTrailingWhitespace(std::string& input)
    {
        // Clips trailing whitespace from the beginning and end of the input string
        int32_t startIdx = 0;
        while (startIdx < signed(input.size()) && std::isspace(input[startIdx])) { ++startIdx; }
        int32_t endIdx = input.size() - 1;
        while (endIdx >= startIdx && std::isspace(input[endIdx])) { --endIdx; }

        input = input.substr(startIdx, std::max(0, 1 + endIdx - startIdx));
    }

    void MakeLowercase(std::string& input)
    {
        for (auto& c : input) { c = std::tolower(c); }
    }

    std::string Lowercase(const std::string& input)
    {
        std::string output;
        output.resize(input.length());
        for (size_t idx = 0; idx < output.length(); idx++)
        {
            output[idx] = std::tolower(input[idx]);
        }
        return output;
    }

    bool IsLowercase(const std::string& input)
    {
        for (auto c : input) { if (std::isupper(c)) { return false; } }
        return true;
    }

    void MakeUppercase(std::string& input)
    {
        for (auto& c : input) { c = std::toupper(c); }
    }

    std::string Uppercase(const std::string& input)
    {
        std::string output;
        output.resize(input.length());
        for (size_t idx = 0; idx < output.length(); idx++)
        {
            output[idx] = std::toupper(input[idx]);
        }
        return output;
    }

    bool IsUppercase(const std::string& input)
    {
        for (auto c : input) { if (std::islower(c)) { return false; } }
        return true;
    }

    std::string FormatElapsedTime(const float time)
    {
        const int seconds = int(time);
        std::string formatted;

        if (seconds >= 86000) { formatted += tfm::format("%i:", seconds / 86000); }
        if (seconds >= 3600) { formatted += tfm::format("%2.i:", (seconds / 3600) % 24); }
        return formatted + tfm::format("%.2i:%.2i", (seconds / 60) % 60, seconds % 60);
    }

    // Formats a floating point value with digit grouping to dp decimal places
    std::string FormatPrettyFloat(const float value, int dp)
    {
        std::string strIntValue = tfm::format("%i", int(value));
        std::string fmtValue;
        fmtValue.reserve(strIntValue.length());

        for (int idx = 0; idx < strIntValue.length(); ++idx)
        {
            if (idx != strIntValue.length() - 1 && (strIntValue.length() - idx - 1) % 3 == 0) { fmtValue += ','; }
            fmtValue += strIntValue[idx];
        }

        const std::string strFloatValue = tfm::format(tfm::format("%%0.%if", dp).c_str(), std::fmod(value, 1.0f));
        return fmtValue + strFloatValue.substr(1, strFloatValue.length() - 1);
    }

    std::string FormatDataSize(const float inputMB, const int dp)
    {
        const int64_t bytes = inputMB * 1048576.f;
        if (bytes < 1024ll) { return tfm::format("%sB", bytes); }
        else if (bytes < 1048576ll) { return FormatPrettyFloat(bytes / 1024.f, 2) + "kB"; }
        else if (bytes < 1073741824ll) { return FormatPrettyFloat(bytes / 1048576.f, 2) + "MB"; }
        return FormatPrettyFloat(bytes / 1073741824.f, 2) + "GB";
    }
}