#pragma once

#include "StdIncludes.h"

std::wstring Widen(const std::string& str);

// Capitalises the first letter of the input string
std::string CapitaliseFirst(const std::string& input);

void ClipTrailingWhitespace(std::string& input);

void MakeLowercase(std::string& input);
std::string Lowercase(const std::string& input);
bool IsLowercase(const std::string& input);

void MakeUppercase(std::string& input);
std::string Uppercase(const std::string& input);
bool IsUppercase(const std::string& input);

std::string FormatPrettyFloat(const float value, int dp);
std::string FormatElapsedTime(const float time);
std::string FormatDataSize(const float inputMB, const int dp);

class Lexer
{
private:
    const std::string& m_string;
    std::string::const_iterator   m_it;

public:
    enum Flags
    {
        ExcludeLimiter = 1,
        IncludeDelimiter = 1 << 1,
        AwareOfStringLiterals = 1 << 2,
        AwareOfNesting = 1 << 3
    };

    Lexer() = delete;
    Lexer(const std::string& input) : m_string(input), m_it(m_string.begin()) {}

    inline operator bool() const { return m_it != m_string.end(); }
    //inline bool operator !() const { return !(*this); }

    const std::string& sourceString() const { return m_string; }

    void begin()
    {
        m_it = m_string.begin();
    }

    // Seeks to the next non-whitespace character or EOL
    inline bool SeekNext()
    {
        return SeekTo([](char c) { return !std::isspace(c); });
    }

    // Seeks until check returns true or EOL
    template<typename CheckLambda>
    bool SeekTo(CheckLambda check)
    {
        while (!eol())
        {
            if (check(*m_it)) { break; }
            ++m_it;
        }
        return !eol();
    }

    inline bool eol() const { return m_it == m_string.end(); }

    // Checks to see if the next non-whitespace characer is c. Skips over it regardless.
    bool ReadNext(char c)
    {
        AssertMsg(SeekNext(), "Unexpected end-of-line encountered.");
        return *(m_it++) == c;
    }

    // Checks to see if the next non-whitespace characeter is c. Skips over it ONLY if it is.
    bool PeekNext(char c)
    {
        AssertMsg(SeekNext(), "Unexpected end-of-line encountered.");
        if (!eol() && *m_it == c) { ++m_it; return true; }
        else { return false; }
    }

    // Seeks until check is true, parses resulting token
    template<typename CheckLambda>
    bool SeekAndParseToken(std::string& token, CheckLambda check, int32_t flags = 0)
    {
        if (!SeekTo(check)) { return false; }
        return ParseToken(token, check, flags);
    }

    // Seeks until a non-whitespace character is found, then parses the subsequent token
    inline bool SeekAndParseToken(std::string& token, int32_t flags = 0)
    {
        return SeekAndParseToken(token, [](char c) { return !std::isspace(c); });
    }

    inline bool ParseToken(std::string& token, char delimiter, int32_t flags = 0)
    {
        return SeekAndParseToken(token, [&](char c) { return c != delimiter; });
    }

    // Reads characters into token until check returns false. Flags allow the first or
    // last characters to be skipped.
    template<typename CheckLambda>
    bool ParseToken(std::string& token, CheckLambda check, int32_t flags = 0)
    {
        token.clear();
        int32_t validChars = 0;
        while (!eol())
        {
            if (!check(*m_it))
            {
                if (flags & IncludeDelimiter) { token += *(m_it++); }
                break;
            }

            if (++validChars == 1 && (flags & ExcludeLimiter)) { m_it++; }
            else { token += *(m_it++); }

        }
        return !token.empty();
    }

    // Reads a token delimited by the input parameters and also excludes the limiters/delimiters
    // from the token string. E.g. running ParseDelimitedToken(token, '{', '}') on "{hello}world"
    // will return "hello" and leave "world" in the input buffer
    bool ParseDelimitedToken(std::string& token, const char limiter,
        const char delimiter, const int32_t flags = 0)
    {
        int32_t indentation = 0;
        bool inQuotes = false;
        auto parseRules = [&](char c)
        {
            if (c == '\"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == limiter && (!inQuotes || !(flags & AwareOfStringLiterals)))
            {
                indentation++;
            }
            else if (c == delimiter && (!inQuotes || !(flags & AwareOfStringLiterals)))
            {
                AssertMsg(--indentation >= 0, "Mismatched limiters/delimiters in token.");
            }
            return indentation > 0;
        };

        if (!ParseToken(token, parseRules, ExcludeLimiter | IncludeDelimiter)) { return false; }
        AssertMsg(!inQuotes || !(flags & AwareOfStringLiterals), "Mismatched quotes in token.");
        token.pop_back();
        return true;
    }

    // Tries to parse an integer value
    template<typename Type>
    bool parseInteger(Type& value)
    {
        static_assert(std::numeric_limits<Type>::is_integer, "Requires integer type.");
        std::string token;
        ParseToken(token, [](char c) { return std::isdigit(c) || c == '-'; });
        if (token.empty()) { return false; }

        try
        {
            value = Type(std::stoi(token.c_str()));
        }
        catch (...) { return false; }

        return true;
    }

    // Tries to parse a real value
    template<typename Type>
    bool parseReal(Type& value)
    {
        static_assert(std::is_floating_point<Type>::value, "Requires real type.");
        std::string token;
        ParseToken(token, [](char c) { return std::isdigit(c) || c == '-' || c == '.'; });
        if (token.empty()) { return false; }

        try
        {
            value = Type(std::stof(token.c_str()));
        }
        catch (...) { return false; }

        return true;
    }

    char operator*() const
    {
        AssertMsg(!eol(), "Unexpected end-of-line encountered.");
        return *m_it;
    }

    std::string::const_iterator& operator++()
    {
        AssertMsg(!eol(), "Unexpected end-of-line encountered.");
        return ++m_it;
    }

    std::string::const_iterator operator++(int)
    {
        AssertMsg(!eol(), "Unexpected end-of-line encountered.");
        return m_it++;
    }
};