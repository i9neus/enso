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

int main(int argc, char* argv[])
{
	std::shared_ptr<AssetBase> object;
	
	try
	{
		D3DContainer sample(1280, 720, "D3D12 Hello Texture");
		return Win32Application<D3DWindowInterface>::Run(&sample, GetModuleHandle(NULL), SW_SHOW);

		GlobalAssetRegistry::Get().VerifyEmpty();
	}
	catch (const std::runtime_error& err)
	{
		std::printf("Runtime error: %s\n", err.what());
	}
}

/*_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	D3DContainer sample(1280, 720, "D3D12 Hello Texture");
	return Win32Application<D3DWindowInterface>::Run(&sample, hInstance, nCmdShow);
}*/
