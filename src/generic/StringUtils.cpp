#include "StringUtils.h"
#include <cstdlib>
#include <clocale>
#include "generic/Math.h"

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

    input = input.substr(startIdx, math::max(0, 1 + endIdx - startIdx));
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