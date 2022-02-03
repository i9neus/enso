#include "ProcessMemoryMonitor.h"
#include <psapi.h>

ProcessMemoryMonitor::ProcessMemoryMonitor() : m_isGood(false)
{
    Clear();
}

void ProcessMemoryMonitor::Clear()
{
    std::memset(&m_pmcStart, 0, sizeof(PROCESS_MEMORY_COUNTERS));
}

void ProcessMemoryMonitor::Snapshot()
{
    if (GetProcessMemoryInfo(GetCurrentProcess(), &m_pmcStart, sizeof(m_pmcStart)))
    {
        m_isGood = true;
    }
}

int64_t ProcessMemoryMonitor::GetWorkingSetSize() const
{
    PROCESS_MEMORY_COUNTERS pmcNow;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmcNow, sizeof(pmcNow))) { return 0; }
    
    return int64_t(pmcNow.WorkingSetSize) - int64_t(m_pmcStart.WorkingSetSize);
}

ProcessMemoryMonitor& ProcessMemoryMonitor::Get()
{
    static ProcessMemoryMonitor singleton;
    return singleton;
}