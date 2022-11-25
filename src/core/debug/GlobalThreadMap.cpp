#include "GlobalThreadMap.h"
#include "../Assert.h"

#include <thread>
#include <unordered_map>
#include <string>
#include <mutex>
#include <functional>

namespace Enso
{
    class GlobalThreadMap
    {
    public:
        GlobalThreadMap() {}

        void ClearThreadName(const std::string& name)
        {
            const size_t idHash = std::hash<std::thread::id>{}(std::this_thread::get_id());

            AssertMsgFmt(m_idToName.find(idHash) != m_idToName.end(), "Trying to clear '%s' but it was not previously named.", name.c_str());

            m_idToName.erase(idHash);
            m_nameToID.erase(name);
        }

        void SetThreadName(const std::string& name)
        {
            Assert(!name.empty());

            const size_t idHash = std::hash<std::thread::id>{}(std::this_thread::get_id());

            auto it = m_idToName.find(idHash);
            if (it != m_idToName.end())
            {
                AssertMsgFmt(false, "Thread '%s' has already been named '%s'. Clear it before setting it again.", name.c_str(), it->second.c_str());
            }

            m_idToName[idHash] = name;
            m_nameToID[name] = idHash;
        }

        void AssertInThread(const std::string& expectedName)
        {
            auto nameIt = m_nameToID.find(expectedName);
            AssertMsgFmt(nameIt != m_nameToID.end(), "Error: no thread with name '%s' was previously registered", expectedName.c_str());

            const size_t idHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
            if (idHash != nameIt->second)
            {
                auto idIt = m_idToName.find(idHash);
                const std::string actualName = (idIt == m_idToName.end()) ? "[Unregistered]" : idIt->second;

                throw std::runtime_error(tfm::format("Error: expected to be in thread '%s'. Actually in thread '%s'", expectedName, actualName));
            }
        }

        size_t Size() const { m_idToName.size(); }

    private:
        std::unordered_map<size_t, std::string> m_idToName;
        std::unordered_map<std::string, size_t> m_nameToID;

        std::mutex m_mutex;
    };

    GlobalThreadMap gThreadMap;

    // Only monitor threading in debug mode
#ifdef _DEBUG || WATCH_GLOBAL_THREADS

    void SetThreadName(const std::string& name)
    {
        gThreadMap.SetThreadName(name);
    }

    void AssertInThread(const std::string& name)
    {
        gThreadMap.AssertInThread(name);
    }

    void ClearThreadName(const std::string& name)
    {
        gThreadMap.ClearThreadName(name);
    }

#else

    void SetThreadName(const std::string&) {}
    void AssertInThread(const std::string&) {}
    void ClearThreadName(const std::string&) {}

#endif
}
