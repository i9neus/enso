#include "StringUtils.h"
#include <cstdlib>
#include <clocale>

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