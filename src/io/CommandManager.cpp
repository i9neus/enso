#include "CommandManager.h"

namespace Enso
{
    CommandManager::CommandManager()
    {}    

    // Flushes the command queue and processes the commands contained within it
    void CommandManager::Flush(CommandQueue& inCmd, const bool debug)
    {
        // Copy the data from the command queue and purge it
        Json::Document rootNode;
        if (!inCmd.Flush(rootNode)) { return; }

        if (debug) { Log::Debug(rootNode.Stringify(true)); }

        try
        {
            // Process each command in turn
            for (Json::Document::Iterator nodeIt = rootNode.begin(); nodeIt != rootNode.end(); ++nodeIt)
            {
                Json::Node childNode = *nodeIt;
                if (!childNode.IsObject()) { continue; }

                const auto functorIt = m_eventMap.find(nodeIt.Name());
                if (functorIt == m_eventMap.end())
                {
                    Log::Debug("Flush: command '%s' was not registered", nodeIt.Name());
                    continue;
                }

                // Call the event functor
                (functorIt->second)(childNode);
            }
        }
        catch (std::runtime_error& err)
        {
            Log::Error("Command error: %s", err.what());
        }
    }
}