#pragma once

#include "../StdIncludes.h"

class ProcessMemoryMonitor
{
public:
    static ProcessMemoryMonitor&    Get();
    void                            Snapshot();
    void                            Clear();

    int64_t                         GetWorkingSetSize() const;

private:
    ProcessMemoryMonitor();

private:
    PROCESS_MEMORY_COUNTERS m_pmcStart;
    bool                    m_isGood;
};