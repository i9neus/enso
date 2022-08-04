#include "Job.h"
#include "JsonCommandQueue.h"

/*void Probegen::Dispatch(const Json::Document& rootJson)
{


	Assert(rootJson.NumMembers() != 0);

	//Log::Debug(rootJson.Stringify(true));

	if (rootJson.GetChildObject("patches", Json::kSilent | Json::kLiteralID))
	{
		std::lock_guard<std::mutex> lock(m_jsonInputMutex);

		// Overwrite the command list with the new data
		m_patchJson = rootJson;

		// Found a scene object parameter parameter patch, so signal that the scene graph is dirty
		EmplaceRenderCommand(kRenderMangerUpdateParams);

		//Log::Debug("Updated! %s\n", m_patchJson.Stringify(true));
	}

	const Json::Node commandsJson = rootJson.GetChildObject("commands", Json::kSilent | Json::kLiteralID);
	if (commandsJson)
	{
		// Examine the command list
		for (auto nodeIt = commandsJson.begin(); nodeIt != commandsJson.end(); ++nodeIt)
		{
			auto commandIt = m_jobMap.find(nodeIt.Name());
			if (commandIt != m_jobMap.end())
			{
				auto& job = commandIt->second;
				try
				{
					// Copy any JSON data that accompanies this command
					job.json = *nodeIt;

					// Call the dispatch functor
					Assert(job.onDispatch);
					job.onDispatch(job);

					//Log::System("Dispatched new job: %s", nodeIt.Name());
				}
				catch (const std::runtime_error& err)
				{
					Log::Error("Error: render manager command '%s' failed: %s", nodeIt.Name(), err.what());
				}
			}
			else
			{
				Log::Error("Error: '%s' is not a valid render manager command", nodeIt.Name());
			}
		}
	}
}*/