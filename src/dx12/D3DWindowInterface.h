#pragma once

#include "generic/StdIncludes.h"

class D3DWindowInterface
{
public:
	D3DWindowInterface(UINT width, UINT height, std::string name)
	{
		m_clientWidth = width;
		m_clientHeight = height;
		m_name = name;
		m_aspectRatio = float(width) / float(height);
	}

    virtual void OnInit(HWND hWnd) = 0;
    virtual void OnUpdate() = 0;
    virtual void OnRender() = 0;
    virtual void OnDestroy() = 0;

	virtual void OnKeyDown(UINT8 /*key*/) {}
	virtual void OnKeyUp(UINT8 /*key*/) {}

	virtual void OnClientResize(HWND hWnd, UINT width, UINT height, WPARAM wParam) {}

	UINT GetClientWidth() const { return m_clientWidth; }
	UINT GetClientHeight() const { return m_clientHeight; }
	const CHAR* GetTitle() const { return "Container"; }

protected:
	UINT		m_clientWidth;
	UINT		m_clientHeight;
	float		m_aspectRatio;
	std::string m_name;
};