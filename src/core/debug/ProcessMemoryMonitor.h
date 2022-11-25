#pragma once

#include <memory>

struct _PROCESS_MEMORY_COUNTERS;

namespace Enso
{
    class ProcessMemoryMonitor
    {
    public:
        static ProcessMemoryMonitor& Get();
        void                            Snapshot();
        void                            Clear();

        int64_t                         GetWorkingSetSize() const;

    private:
        ProcessMemoryMonitor();

    private:
        std::unique_ptr<_PROCESS_MEMORY_COUNTERS> m_pmcStart;
        bool                                     m_isGood;
    };
}