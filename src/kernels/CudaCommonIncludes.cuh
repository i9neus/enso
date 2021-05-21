#pragma once

#include <DirectXMath.h>
#include <cuda_runtime.h>
#include "Assert.h"
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