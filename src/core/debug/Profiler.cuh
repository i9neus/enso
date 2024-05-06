#include <atomic>
#include "Assert.h"
#include <thread>
#include "io/Log.h"

namespace Enso
{
    class Profiler
    {
    private:
        std::atomic<int>    m_state;
        std::mutex          m_mutex;

    public:
        static Profiler&    Get();

        void                SetState(const int);
        int                 GetState() const { return m_state.load(); }
        void                AtomicMsg(const std::string& msg);

    private:
        Profiler() = default;
    };
}