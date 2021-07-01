/*
* Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "generic/StdIncludes.h"

#include "dx12/D3DContainer.h"
#include "Win32Application.h"

#include "generic/StringUtils.h"
#include "generic/JsonUtils.h"

// Call various method to guarante we have object files generated for the test suit. 
void DummyMethodCalls()
{
	auto wstr = Widen("");
}

int main(int argc, char* argv[])
{	
	DummyMethodCalls();

	/*Json::Document document;
	document.AddValue("value1", 1);
	document.AddValue("value2", "hello");
	Json::Node node = document.AddChildObject("child");
	node.AddValue("test", 1.0f); 
	std::vector<float> arr = { 1.0f, 2.0f, 3.0f };
	node.AddArray("array", arr);

	std::cout << document.Stringify() << std::endl;*/
	
	//try
	{
		D3DContainer sample(1280, 720, "D3D12 Hello Texture");
		auto rValue = Win32Application<D3DWindowInterface>::Run(&sample, GetModuleHandle(NULL), SW_SHOW);

		Cuda::AR().VerifyEmpty();

		return rValue;
	}
	/*catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());
	}*/
}

/*_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	D3DContainer sample(1280, 720, "D3D12 Hello Texture");
	return Win32Application<D3DWindowInterface>::Run(&sample, hInstance, nCmdShow);
}*/
