#include "JsonTests.h"

#include "io/json/JsonUtils.h"

namespace Tests
{
	template<typename Lambda>
	void TryFailOnSucceed(Json::Document& doc, Lambda lambda)
	{
		try
		{
			lambda();

			Logger::WriteMessage(doc.Stringify(true).c_str());
			Assert::Fail(L"Function should not have succeeded.");
		}
		catch (std::runtime_error& err)
		{
			Logger::WriteMessage(tfm::format("Expected exception thrown: %s", err.what()).c_str());
		}
		catch (...) { Assert::Fail(L"Threw an unexpected error."); }
	}

	template<typename Lambda>
	void Try(Json::Document& doc, Lambda lambda)
	{
		try
		{
			lambda();
		}
		catch (std::runtime_error& err)
		{
			Logger::WriteMessage(doc.Stringify(true).c_str());
			Assert::Fail(Widen(tfm::format("Failed with assertion '%s'", err.what())).c_str());
		}
		catch (...) { Assert::Fail(L"Threw an unexpected error."); }
	}
	
	void JsonTestsImpl::AddChildDAGInEmptyTree()
	{
		Json::Document doc;
		Try(doc, [&]()
			{
				// Add a new node based on a DAG
				Json::Node addNode = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

				// Verify the presense of the node
				Json::Node findNode = doc.GetChild("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);
			});	
	}

	void JsonTestsImpl::AddChildDAGInValidTree()
	{
		Json::Document doc;
		Try(doc, [&]()
			{
				// Add a new node based on a DAG
				Json::Node addNode = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

				// Add a new node based on a DAG
				addNode = doc.AddChildObject("childA/childD/childE", Json::kPathIsDAG | Json::kRequiredAssert);

				// Verify the presense of the node
				Json::Node findNode = doc.GetChild("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);
				findNode = doc.GetChild("childA/childD/childE", Json::kPathIsDAG | Json::kRequiredAssert);
			});
	}

	void JsonTestsImpl::AddChildMalformedPath()
	{
		Json::Document doc;
		TryFailOnSucceed(doc, [&]()
			{
				// Add a new node based on a malformed DAG
				Json::Node addNode = doc.AddChildObject("childA/childB//childC", Json::kPathIsDAG | Json::kRequiredAssert);
			});		

		doc.Clear();
		TryFailOnSucceed(doc, [&]()
			{
				// Add a new node based on a malformed name
				doc.AddValue("childD@", 10);
			});
	}
	
	void JsonTestsImpl::AddChildDAGInInvalidTree()
	{
		Json::Document doc;
		TryFailOnSucceed(doc, [&]()
			{
				// Add a new node based on a DAG
				Json::Node node = doc.AddChildObject("childA", Json::kRequiredAssert);
				node.AddValue("childB", 1);

				node = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG | Json::kRequiredAssert);

				Logger::WriteMessage(doc.Stringify(true).c_str());
				Assert::Fail(L"AddChildObject should not have succeeded.");
			});		
	}	

	void JsonTestsImpl::MoveCtor()
	{
		Json::Document doc;
		Json::Node node = doc.AddChildObject("childA/childB/childC", Json::kPathIsDAG);

	}
}