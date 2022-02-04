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
#include "generic/debug/ProcessMemoryMonitor.h"

//#include "utils/BulkUSDProcessor.h"

// Call various method to guarante we have object files generated for the test suite. 
void DummyMethodCalls()
{
	auto wstr = Widen("");
}

void InitialiseGlobalObjects()
{
	// Global state authority holds config data that all objects may occasionally access but which is 
	// too cumbersome to pass in via contexts.
	GlobalStateAuthority::Get();

	// Platform-specific code which monitors the size and contents of memory at the process level. 
	ProcessMemoryMonitor::Get();
}

int main(int argc, char* argv[])
{
	DummyMethodCalls();

	// Initialise any objects that have global scope. 
	InitialiseGlobalObjects();

	Log::Get().EnableLevel(kLogSystem, true);
	Log::Get().EnableLevel(kLogDebug, true);
	
	try
	{			
		D3DContainer sample(1280, 720, "D3D12 Hello Texture");
		auto rValue = Win32Application<D3DWindowInterface>::Run(&sample, GetModuleHandle(NULL), SW_SHOW);

		Cuda::AR().VerifyEmpty();

		return rValue;
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());

		const auto backtrace = StackBacktrace::Get();
		if (!backtrace.empty())
		{
			Log::Debug("Stack backtrace:");
			for (const auto& frame : backtrace)
			{
				Log::Debug(frame);
			}
		}
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
