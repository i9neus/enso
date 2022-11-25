#pragma once

#include <atomic>
#include "Assert.h"
#include <thread>

namespace Enso
{
    class Semaphore
    {
    public:
        Semaphore(const unsigned int& initialState) : m_state(initialState) {}

        bool Try(const unsigned int& expectedState, const unsigned int& newState, bool assertOnFail)
        {
            unsigned int actualState = expectedState;
            bool success = m_state.compare_exchange_strong(actualState, newState, std::memory_order_release, std::memory_order_relaxed);
            if (!success && assertOnFail)
            {
                AssertMsgFmt(success, "Semaphore failed to transition to state %i. Expected %i, found %i.", newState, expectedState, actualState);
            }

            return success;
        }

        void Wait(const unsigned int& expectedState, const unsigned int& newState)
        {
            unsigned int actualState = expectedState;
            while (!m_state.compare_exchange_strong(actualState, newState, std::memory_order_release, std::memory_order_relaxed))
            {
                std::this_thread::yield();
                actualState = expectedState;
            }
        }

        bool WaitFor(const unsigned int& expectedState, const unsigned int& newState, const std::chrono::duration<double>& duration)
        {
            unsigned int actualState = expectedState;
            const auto spinStart = std::chrono::high_resolution_clock::now();
            while (!m_state.compare_exchange_strong(actualState, newState, std::memory_order_release, std::memory_order_relaxed))
            {
                std::this_thread::yield();
                if (std::chrono::high_resolution_clock::now() - spinStart > duration) { return false; }
                actualState = expectedState;
            }

            return true;
        }

        explicit operator unsigned int() const
        {
            return m_state;
        }

    private:
        std::atomic<unsigned int> m_state;
    };
}