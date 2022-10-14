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

#include "Win32Application.h"

#include "D3DContainer.h"
#include "Win32Application.h"

#include "generic/StringUtils.h"
#include "generic/JsonUtils.h"
#include "generic/GlobalStateAuthority.h"
#include "generic/debug/ProcessMemoryMonitor.h"
#include "generic/WindowsHeaders.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>

#include <string>
#include <wrl.h>
#include <shellapi.h>
#include <iostream>

#include "Win32Application.h"

#include "thirdparty/imgui/backends/imgui_impl_win32.h"

void Win32Application::InitialiseIMGUI(HWND hWnd)
{
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer backends
	ImGui::ImplWin32_Init(hWnd);
}

void Win32Application::DestroyIMGUI()
{
	ImGui::ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

int Win32Application::Run(D3DContainer& d3dContainer, HINSTANCE hInstance, int nCmdShow)
{
	// Parse the command line parameters
	/*int argc;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	pSample->ParseCommandLineArgs(argv, argc);
	LocalFree(argv);*/

	// Initialize the window class.
	WNDCLASSEX windowClass = { 0 };
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = WindowProc;
	windowClass.hInstance = hInstance;
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.lpszClassName = "DX12CudaSampleClass";
	RegisterClassEx(&windowClass);

	constexpr LONG kStartupWidth = 1920;
	constexpr LONG kStartupHeight = 1080;

	RECT windowRect = { 0, 0, kStartupWidth, kStartupHeight };
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

	// Create the window and store a handle to it.
	GetHwnd() = CreateWindow(
		windowClass.lpszClassName,
		"",
		WS_OVERLAPPEDWINDOW | WS_MAXIMIZE,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr,		// We have no parent window.
		nullptr,		// We aren't using menus.
		hInstance,
		&d3dContainer);

	InitialiseIMGUI(GetHwnd());

	ShowWindow(GetHwnd(), nCmdShow);

	// Initialize the sample. OnInit is defined in each child-implementation of DXSample.
	d3dContainer.OnCreate(GetHwnd());

	// Main sample loop.
	MSG msg = {};
	while (msg.message != WM_QUIT)
	{
		// Process any messages in the queue.
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	d3dContainer.OnDestroy();

	// Return this part of the WM_QUIT message to Windows.
	return static_cast<char>(msg.wParam);
}

// Forward declare message handler from imgui_impl_win32.cpp
namespace ImGui
{
	extern IMGUI_IMPL_API LRESULT ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
}

// Main message handler for the sample.
LRESULT CALLBACK Win32Application::WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (ImGui::ImplWin32_WndProcHandler(hWnd, message, wParam, lParam)) { return 1; }

	try
	{
		D3DContainer* d3dContainer = reinterpret_cast<D3DContainer*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

		// Only process UI messages if the container object has been successfully created
		if (d3dContainer)
		{
			switch (message)
			{
		
			case WM_KEYDOWN:	d3dContainer->OnKey(wParam, false, true); return 0;
			case WM_KEYUP:		d3dContainer->OnKey(wParam, false, false); return 0;
			case WM_SYSKEYDOWN:	d3dContainer->OnKey(wParam, true, true); return 0;
			case WM_SYSKEYUP:	d3dContainer->OnKey(wParam, true, false); return 0;

			case WM_LBUTTONDOWN: case WM_LBUTTONDBLCLK: d3dContainer->OnMouseButton(VK_LBUTTON, true); return 0;
			case WM_RBUTTONDOWN: case WM_RBUTTONDBLCLK: d3dContainer->OnMouseButton(VK_RBUTTON, true); return 0;
			case WM_MBUTTONDOWN: case WM_MBUTTONDBLCLK: d3dContainer->OnMouseButton(VK_MBUTTON, true); return 0;
			case WM_XBUTTONDOWN: case WM_XBUTTONDBLCLK: d3dContainer->OnMouseButton(VK_XBUTTON1, true); return 0;
		
			case WM_LBUTTONUP: d3dContainer->OnMouseButton(VK_LBUTTON, false); return 0;
			case WM_RBUTTONUP: d3dContainer->OnMouseButton(VK_RBUTTON, false); return 0;
			case WM_MBUTTONUP: d3dContainer->OnMouseButton(VK_MBUTTON, false); return 0;
			case WM_XBUTTONUP: d3dContainer->OnMouseButton(VK_XBUTTON1, false); return 0;

			case WM_MOUSEMOVE: d3dContainer->OnMouseMove(int(GET_X_LPARAM(lParam)), int(GET_Y_LPARAM(lParam)), wParam); return 0;

			case WM_MOUSEWHEEL: d3dContainer->OnMouseWheel((float)GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA);
				return 0;

			case WM_PAINT:
				//Timer frameTimer;
				d3dContainer->OnUpdate();
				d3dContainer->OnRender();

				//SetWindowText(hWnd, tfm::format("%.2f FPS", 1.0f / frameTimer.Get()).c_str());
				return 0;

			case WM_SIZE:
				d3dContainer->OnClientResize(hWnd, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), wParam);
				return 0;
			}
		}

		// Critical window messages
		switch (message)
		{
		case WM_CREATE:
		{
			// Save the DXSample* passed in to CreateWindow.
			LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
			SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		}
		return 0;

		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
		case WM_CLOSE:
			DestroyWindow(hWnd);
			return 0;
		}
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s", err.what());
		StackBacktrace::Print();
	}
	catch (...)
	{
		Log::Error("Unhandled exception.");
		StackBacktrace::Print();
	}

	// Handle any messages the switch statement didn't.
	return DefWindowProc(hWnd, message, wParam, lParam);
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
	// Initialise any objects that have global scope. 
	InitialiseGlobalObjects();

	Log::EnableLevel(kLogSystem, true);
	//Log::Get().EnableLevel(kLogDebug, true);
	
	try
	{			
		D3DContainer d3dContainer("Probegen");
		auto rValue = Win32Application::Run(d3dContainer, GetModuleHandle(NULL), SW_SHOW);

		Cuda::AR().VerifyEmpty();

		return rValue;
	}
	catch (const std::runtime_error& err)
	{
		Log::Error("Runtime error: %s\n", err.what());

		StackBacktrace::Print();
	}
	catch (...)
	{
		Log::Error("Unhandled error");
	}
}
