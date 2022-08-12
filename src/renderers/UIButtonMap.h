#pragma once

#include <map>
#include "generic/Hash.h"

enum UIButtonStates : uint { kButtonUp = 1, kOnButtonDepressed = 2, kButtonDown = 4, kOnButtonReleased = 8 };

template<typename T>
inline std::string bstr(const T& t, const bool truncate = false)
{
    static_assert(std::is_integral<T>::value, "Not an integer");

    std::string str;
    for (int j = sizeof(t) * 8 - 1; j >= 0; j--)
    {
        str += (t & (T(1) << j)) ? '1' : '0';

        if (j != 0 && j % 8 == 0) { str += ','; }
    }
    return str;
}

template<size_t NumButtons>
class UIButtonMap
{
    enum _attrs : uint { kAllKeysUp = 0x0, kAllKeysDown = 0xaaaaaaaa, kKeyCodeArraySize = 1 + ((NumButtons - 1) >> 4) };

public:
    UIButtonMap()
    {
        std::memset(m_codes.data(), 0, sizeof(uint) * kKeyCodeArraySize);
    }

    UIButtonMap(const UIButtonMap& other)
    {
        std::memcpy(m_codes.data(), other.m_codes.data(), sizeof(uint) * kKeyCodeArraySize);
    }

    inline uint StateToBits(const uint state) const
    {        
        return ((state >> 1) & 1) | ((state >> 1) & 2) | ((state >> 3) * 3);
    }

    void Update(const uint code, const bool isDown)
    {
        Assert(code < NumButtons);

        // If any keys previously registered as either actively onDown or onUp, move them to passively down or up 
        for (int blockIdx = 0; blockIdx < kKeyCodeArraySize; ++blockIdx)
        {
            if (m_codes[blockIdx] != kAllKeysUp && m_codes[blockIdx] != kAllKeysDown)
            {
                for (int bitIdx = 0; bitIdx < 32; bitIdx += 2)
                {
                    uint code = (m_codes[blockIdx] >> bitIdx) & 3;
                    switch (code)
                    {
                    case 0: code = 1; break;
                    case 2: code = 3; break;
                    }                    
                    m_codes[blockIdx] = (m_codes[blockIdx] & ~(3 << bitIdx)) | (m_codes[blockIdx] << bitIdx);
                }
            }
        }

        // Set the new state
        SetState(code, isDown ? kOnButtonDepressed : kOnButtonReleased);

        /*Log::Write("----------");
        for (int blockIdx = 0; blockIdx < kKeyCodeArraySize; ++blockIdx)
        {
            Log::Write("%i: %s", blockIdx, bstr(m_codes[blockIdx]));
        }*/
    }

    uint GetState(const uint code) const
    {        
        Assert(code < NumButtons);
        const uint bitIdx = code * 2;
        return 1 << ((m_codes[bitIdx >> 5] >> (bitIdx & 31)) & 3u);
    }

    void SetState(const uint code, const uint state)
    {        
        Assert(code < NumButtons);
        uint bitIdx = code * 2, blockIdx = bitIdx >> 5;
        bitIdx &= 31;
        m_codes[blockIdx] = (m_codes[blockIdx] & ~(3 << bitIdx)) | (StateToBits(state) << bitIdx);
    }

    inline bool operator==(const UIButtonMap& rhs) const
    {
        for (int idx = 0; idx < kKeyCodeArraySize; ++idx)
        {
            if (m_codes[idx] != rhs.m_codes[idx]) { return false; }
        }
        return true;
    }

    inline bool operator!=(const UIButtonMap& rhs) const { return !(this->operator==(rhs)); }

    uint HashOf() const
    {
        uint hash = 0x8f23aba7;
        for (const auto& code : m_codes)
        {
            hash = HashCombine(hash, std::hash<uint>{}(code));
        }
        return hash;
    }

private:
    std::array < uint, kKeyCodeArraySize > m_codes;
};

using MouseButtonMap = UIButtonMap<256>;
using KeyboardButtonMap = UIButtonMap<256>;