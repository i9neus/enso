#pragma once

#include "generic/StdIncludes.h"

class D3DWindowInterface
{
public:
	D3DWindowInterface(UINT width, UINT height, std::string name)
	{
		m_width = width;
		m_height = height;
		m_name = name;
		m_aspectRatio = float(width) / float(height);
	}

    virtual void OnInit() = 0;
    virtual void OnUpdate() = 0;
    virtual void OnRender() = 0;
    virtual void OnDestroy() = 0;

	virtual void OnKeyDown(UINT8 /*key*/) {}
	virtual void OnKeyUp(UINT8 /*key*/) {}

	UINT GetWidth() const { return m_width; }
	UINT GetHeight() const { return m_height; }
	const CHAR* GetTitle() const { return "Container"; }

protected:
	UINT		m_width;
	UINT		m_height;
	float		m_aspectRatio;
	std::string m_name;
};