#pragma once

#include <map>
#include "core/math/Hash.cuh"
#include "VirtualKeyStates.h"

#include <string>

namespace Enso
{    
    enum UIButtonStates : uint
    {
        kButtonUp = 1,
        kOnButtonDepressed = 2,
        kButtonDown = 4,
        kOnButtonReleased = 8
    };

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

    struct UICodeStatePair
    {
        uint code;
        uint state;
    };

    template<size_t NumButtons>
    class UIButtonMap
    {
        enum _attrs : uint { kAllKeysUp = 0x0, kAllKeysDown = 0xaaaaaaaa, kKeyCodeArraySize = 1 + ((NumButtons - 1) >> 4) };
        enum _flags : uint { kUndefined = 0u, kWildcard = 1u };

        UIButtonMap(uint flags) : UIButtonMap() 
        {
            m_flags = flags;
        }

    public:        

        UIButtonMap() : m_flags(kUndefined)
        {
            std::memset(m_codes.data(), 0, sizeof(uint) * kKeyCodeArraySize);
        }

        UIButtonMap(const uint code, const uint state = kOnButtonDepressed) : UIButtonMap()
        {
            SetState(code, state);
        }

        UIButtonMap(const std::initializer_list<UICodeStatePair>& initList) : UIButtonMap()
        {
            for (const auto& init : initList)
            {
                *this |= UIButtonMap(init.code, init.state);
            }
        }

        UIButtonMap(const UIButtonMap& other)
        {
            std::memcpy(m_codes.data(), other.m_codes.data(), sizeof(uint) * kKeyCodeArraySize);
        }

        UIButtonMap(UIButtonMap&& other)
        {
            m_codes = std::move(other.m_codes);
        }

        ~UIButtonMap() = default;

        void Clear()
        {
            for (auto code : m_codes) { code = 0u; }
        }

        inline uint StateToBits(const uint state) const
        {
            switch (state)
            {
            case kOnButtonDepressed: return 1;
            case kButtonDown: return 2;
            case kOnButtonReleased: return 3;
            };
            return 0;
            //return ((state >> 1) & 1) | ((state >> 1) & 2) | ((state >> 3) * 3);
        }

        void Echo() const
        {
            Log::Error("----------");
            for (int blockIdx = 0; blockIdx < kKeyCodeArraySize; ++blockIdx)
            {
                Log::Error("%i: %s", blockIdx, bstr(m_codes[blockIdx]));
            }
        }

        void Update(bool echo = true)
        {
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
                        case 1: code = 2; break;
                        case 3: code = 0; break;
                        }
                        m_codes[blockIdx] = (m_codes[blockIdx] & ~(3 << bitIdx)) | (code << bitIdx);
                    }
                }
            }

            //if(echo) Echo();
        }

        inline void Update(const uint code, const bool isDown)
        {
            Assert(code < NumButtons);

            Update(false);

            // Set the new state
            SetState(code, isDown ? kOnButtonDepressed : kOnButtonReleased);

            //Echo();
        }

        // Returns the state of the specified code as a value in UIButtonStates
        uint GetState(const uint code) const
        {
            Assert(code < NumButtons);
            const uint bitIdx = code * 2;
            return 1 << ((m_codes[bitIdx >> 5] >> (bitIdx & 31)) & 3u);
        }

        bool IsSet(const uint code) const
        {
            const uint state = GetState(code);
            return state == kOnButtonDepressed || state == kButtonDown;
        }

        void SetState(const uint code, const uint state)
        {
            // FIXME: Alt-tabbing out of the app breaks things. 
            if (code == KEY_MENU) { return; }

            Assert(code < NumButtons);
            uint bitIdx = code * 2, blockIdx = bitIdx >> 5;
            bitIdx &= 31;
            m_codes[blockIdx] = (m_codes[blockIdx] & ~(3 << bitIdx)) | (StateToBits(state) << bitIdx);
        }

        inline static UIButtonMap Nothing() { return UIButtonMap(); }
        inline static UIButtonMap Wildcard() { return UIButtonMap(kWildcard); }

        const bool IsWildcard() const { return m_flags == kWildcard; }

        inline bool operator==(const UIButtonMap& rhs) const
        {
            for (int idx = 0; idx < kKeyCodeArraySize; ++idx)
            {
                if (m_codes[idx] != rhs.m_codes[idx]) { return false; }
            }
            return true;
        }

        inline bool operator!=(const UIButtonMap& rhs) const { return !(this->operator==(rhs)); }

        UIButtonMap& operator|=(const UIButtonMap& rhs)
        {
            for (int buttonIdx = 0; buttonIdx < NumButtons; ++buttonIdx)
            {
                const uint stateLHS = GetState(buttonIdx);
                if (stateLHS == kButtonUp)
                {
                    SetState(buttonIdx, rhs.GetState(buttonIdx));
                }
            }
            return *this;
        }

        uint HashOf() const
        {
            uint hash = 0x8f23aba7;
            //Log::Write("------");
            for (const auto& code : m_codes)
            {
                hash = HashCombine(hash, std::hash<uint>{}(code));
                //Log::Write("0x%x: 0x%x", code, hash);
            }
            return hash;
        }

    private:
        std::array < uint, kKeyCodeArraySize > m_codes; 
        uint m_flags;
    };

    // ORs two maps together
    template<size_t NumButtons>
    UIButtonMap<NumButtons> operator|(const UIButtonMap<NumButtons>& lhs, const UIButtonMap<NumButtons>& rhs)
    {
        return UIButtonMap<NumButtons>(lhs) |= rhs;
    }

    using VirtualKeyMap = UIButtonMap<256>;
}