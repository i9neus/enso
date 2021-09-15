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
#include "generic/GlobalStateAuthority.h"

//#include "utils/BulkUSDProcessor.h"

// Call various method to guarante we have object files generated for the test suit. 
void DummyMethodCalls()
{
	auto wstr = Widen("");
}

int main(int argc, char* argv[])
{	
	DummyMethodCalls();

	auto& log = Log::Singleton();
	log.EnableLevel(kLogSystem, true);
	log.EnableLevel(kLogDebug, true);
	
	try
	{
		/*BulkConvertUSDProbeGrds();
		return 0;*/

		// Initialise the global state authority singleton
		GSA();
		
		D3DContainer sample(1280, 720, "D3D12 Hello Texture");
		auto rValue = Win32Application<D3DWindowInterface>::Run(&sample, GetModuleHandle(NULL), SW_SHOW);

		Cuda::AR().VerifyEmpty();

		return rValue;
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());
	}
	catch (...)
	{
		Log::Error("Unhandled error");
	}
}

/*_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	D3DContainer sample(1280, 720, "D3D12 Hello Texture");
	return Win32Application<D3DWindowInterface>::Run(&sample, hInstance, nCmdShow);
}*/
