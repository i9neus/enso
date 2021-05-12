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

#include "generic/Includes.h"

#include "dx12/simpleD3D12.h"
#include "dx12/D3D12HelloTexture.h"
#include "dx12/D3DContainer.h"

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	D3DContainer sample(1280, 720, "D3D12 Hello Texture");
	return Win32Application<D3DWindowInterface>::Run(&sample, hInstance, nCmdShow);
	
	//D3D12HelloTexture sample(1280, 720, L"D3D12 Hello Texture");
	//return Win32Application<DXSample>::Run(&sample, hInstance, nCmdShow);
	
	//DX12CudaInterop sample(1280, 720, "D3D12 CUDA Interop");
	//return Win32Application::Run(&sample, hInstance, nCmdShow);
}
