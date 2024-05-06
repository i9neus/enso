#include "Profiler.cuh"

namespace Enso
{
    Profiler& Profiler::Get()
    {
        static Profiler singleton;
        return singleton;
    }

    void Profiler::SetState(const int state)
    {
        m_state.store(state);
    }

    void Profiler::AtomicMsg(const std::string& msg)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Log::Debug(msg);
    }
}