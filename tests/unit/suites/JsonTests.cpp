#include "JsonTests.h"

#include "io/json/JsonUtils.h"

namespace Tests
{
	void JsonTestsImpl::AddChildDAGInEmptyTree()
	{
		Json::Document doc;
		try
		{
			// Add a new node based on a DAG
			Json::Node addNode = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

			// Verify the presense of the node
			Json::Node findNode = doc.GetChild("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

			Logger::WriteMessage(doc.Stringify(true).c_str());
		}
		catch (std::runtime_error& err)
		{
			Assert::Fail(Widen(tfm::format("Failed with assertion '%s'", err.what())).c_str());
		}
		catch (...) { Assert::Fail(L"AddChildObject threw the wrong error."); }
	}

	void JsonTestsImpl::AddChildDAGInValidTree()
	{
		Json::Document doc;
		try
		{
			// Add a new node based on a DAG
			Json::Node addNode = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

			// Add a new node based on a DAG
			addNode = doc.AddChildObject("childA/childD/childE", Json::kPathIsDAG | Json::kRequiredAssert);

			// Verify the presense of the node
			Json::Node findNode = doc.GetChild("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);
			findNode = doc.GetChild("childA/childD/childE", Json::kPathIsDAG | Json::kRequiredAssert);

			Logger::WriteMessage(doc.Stringify(true).c_str());
		}
		catch (std::runtime_error& err)
		{
			Assert::Fail(Widen(tfm::format("Failed with assertion '%s'", err.what())).c_str());
		}
		catch (...) { Assert::Fail(L"AddChildObject threw the wrong error."); }
	}

	void JsonTestsImpl::AddChildMalformedDAG()
	{
		Json::Document doc;
		try
		{
			// Add a new node based on a malformed DAG
			Json::Node addNode = doc.AddChildObject("childA/childB//childC", Json::kPathIsDAG | Json::kRequiredAssert);

			Logger::WriteMessage(doc.Stringify(true).c_str());
			Assert::Fail(L"AddChildObject should not have succeeded.");
		}
		catch (std::runtime_error& err)
		{
			Logger::WriteMessage(tfm::format("Exception thrown: %s", err.what()).c_str());
		}
		catch (...){ Assert::Fail(L"AddChildObject threw the wrong error.");	}
	}
	
	void JsonTestsImpl::AddChildDAGInInvalidTree()
	{
		Json::Document doc;
		try
		{
			// Add a new node based on a DAG
			Json::Node node = doc.AddChildObject("childA", Json::kRequiredAssert);
			node.AddValue("childB", 1);

			node = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

			Logger::WriteMessage(doc.Stringify(true).c_str());
			Assert::Fail(L"AddChildObject should not have succeeded.");
		}
		catch (std::runtime_error& err) 
		{
			Logger::WriteMessage(tfm::format("Exception thrown: %s", err.what()).c_str());
		}
		catch (...) { Assert::Fail(L"AddChildObject threw the wrong error."); }
	}	
}