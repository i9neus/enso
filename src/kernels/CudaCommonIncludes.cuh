#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "generic/thirdparty/nvidia/helper_cuda.h"

using namespace DirectX;

struct Vertex
{
	Vertex(const XMFLOAT3& p, const XMFLOAT4& c) : position(p), color(c) {}
	XMFLOAT3 position;
	XMFLOAT4 color;
};

struct VertexUV
{
	VertexUV(const XMFLOAT3& p, const XMFLOAT2& u) : position(p), uv(u) {}
	XMFLOAT3 position;
	XMFLOAT2 uv;
};

struct CudaImage
{
	CudaImage() : m_width(0u), m_height(0u), c_data(nullptr) {}

	void create(unsigned int width, unsigned int height);
	void destroy();

	unsigned int m_width;
	unsigned int m_height;
	float4*		 c_data;
};