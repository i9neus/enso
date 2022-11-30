#pragma once

#include "SuiteBase.h"

namespace Tests
{
	class JsonTestsImpl : public SuiteBase
	{
	public:
		JsonTestsImpl() = default;

		void AddChildDAGInEmptyTree();
		void AddChildDAGInValidTree();
		void AddChildDAGInInvalidTree();
		void AddChildMalformedDAG();
	};
	
	TEST_CLASS(JsonTests)
	{
	public:
		EXTERNAL_TEST_METHOD(JsonTestsImpl, AddChildDAGInEmptyTree);
		EXTERNAL_TEST_METHOD(JsonTestsImpl, AddChildDAGInValidTree);
		EXTERNAL_TEST_METHOD(JsonTestsImpl, AddChildDAGInInvalidTree);
		EXTERNAL_TEST_METHOD(JsonTestsImpl, AddChildMalformedDAG);
	};
}