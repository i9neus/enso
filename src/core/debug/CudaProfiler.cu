#include "CudaProfiler.cuh"
#include "io/Log.h"

namespace Enso
{
    class CudaProfilerData
    {
    public:
        struct Event
        {
            Event() : numCalls(1) {}
            cudaEvent_t     handle;
            int             numCalls;
        };

        static CudaProfilerData& Get()
        {
            static CudaProfilerData singleton;
            return singleton;
        }

        void Clear()
        {
            if (!eventMap.empty())
            {
                std::lock_guard<std::mutex> lock(mapMutex);

                Log::Warning("Destroyed %i unflushed events...", eventMap.size());
                for (auto& event : eventMap)
                {
                    Log::Warning("  - %s", event.first);
                    IsOk(cudaEventDestroy(event.second.handle));
                }
                eventMap.clear();
            }
        }

        std::unordered_map<std::string, Event>      eventMap;
        std::set<std::thread::id>                   threadMap;        
        std::mutex                                  mapMutex;

    private:        
        CudaProfilerData() = default;

        ~CudaProfilerData()
        {            
            Clear();
        }
    }; 

    void CudaProfiler::Clear()
    {
        CudaProfilerData::Get().Clear();
    }

    bool CudaProfiler::Flush()
    {
        auto& data = CudaProfilerData::Get();
        std::lock_guard<std::mutex> lock(data.mapMutex);


        for (auto it = data.eventMap.begin(); it != data.eventMap.end();)
        {
            if (cudaEventQuery(it->second.handle) == cudaSuccess)
            {
                Log::Write("Flushed event '%s'", it->first);
                cudaEventDestroy(it->second.handle);
                it = data.eventMap.erase(it);
            }            
            else
            {
                ++it;
            }
        }

        return data.eventMap.empty();
    }

    void CudaProfiler::Report()
    {
        auto& data = CudaProfilerData::Get();
        
        std::lock_guard<std::mutex> lock(data.mapMutex);
        Log::Write("%i CUDA events over %i threads", data.eventMap.size(), data.threadMap.size());
        for (auto& event : data.eventMap)
        {
            Log::Write("  - %i calls: %s", event.second.numCalls, event.first);
        }
    }

    void CudaProfiler::Register(const std::string& id)
    {
        auto& data = CudaProfilerData::Get();
        std::lock_guard<std::mutex> lock(data.mapMutex);

        data.threadMap.insert(std::this_thread::get_id());

        auto it = data.eventMap.find(id);
        if (it != data.eventMap.end())
        {
            it->second.numCalls++;
            return;
        }

        CudaProfilerData::Event newEvent;
        IsOk(cudaEventCreate(&newEvent.handle));
        IsOk(cudaEventRecord(newEvent.handle));
        data.eventMap[id] = newEvent;
    } 

    void CudaProfiler::Poll(const std::string& id)
    {
        auto& data = CudaProfilerData::Get();
        std::lock_guard<std::mutex> lock(data.mapMutex);

        auto it = data.eventMap.find(id);
        if (it != data.eventMap.end())
        {
            if (cudaEventQuery(it->second.handle) == cudaSuccess)
            {
                Log::Write("'%s' is complete!", id);
            }
            else
            {
                Log::Warning("'%s' is waiting (%i calls)", it->first, it->second.numCalls);
            }
            
        }
    }
}