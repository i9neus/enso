#pragma once

#include "D3DHeaders.h"

#include "modules/ModuleManager.cuh"
#include "ui/UIModuleManager.h"

#include <vector>
#include <memory>

using namespace DirectX;

namespace Enso
{
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

	static const char* shaderstrs =
		" struct PSInput \n" \
		" { \n" \
		"  float4 position : SV_POSITION; \n" \
		"  float4 color : COLOR; \n" \
		" } \n" \
		" PSInput VSMain(float3 position : POSITION, float4 color : COLOR) \n" \
		" { \n" \
		"  PSInput result;\n" \
		"  result.position = float4(position, 1.0f);\n" \
		"  result.color = color;\n"	\
		"  return result; \n" \
		" } \n" \
		" float4 PSMain(PSInput input) : SV_TARGET \n" \
		" { \n" \
		"  return input.color;\n" \
		" } \n";

	class D3DContainer
	{
	public:
		D3DContainer(std::string name);

		void OnCreate(HWND hWnd);
		void OnRender();
		void OnDestroy();
		void OnUpdate();

		void OnClientResize(HWND hWnd, UINT width, UINT height, WPARAM wParam);
		void OnFocusChange(HWND hWnd, bool isSet);

		void OnKey(const WPARAM code, const bool isSysKey, const bool isDown);
		void OnMouseButton(const int button, const bool isDown);
		void OnMouseMove(const int mouseX, const int mouseY, const WPARAM flags);
		void OnMouseWheel(const float degrees);

		UINT GetClientWidth() const { return m_clientWidth; }
		UINT GetClientHeight() const { return m_clientHeight; }
		const CHAR* GetTitle() const { return "Probegen"; }

	private:

		void CreateDevice();
		void DestroyDevice();

		void CreateRenderTargets();
		void DestroyRenderTargets();

		void CreateSwapChain();
		void DestroySwapChain();

		void CreateAssets();
		void DestroyAssets();

		void CreateSynchronisationObjects();
		void DestroySynchronisationObjects();

		void UpdateAssetDimensions();

		static const UINT				kFrameCount = 2;
		static const UINT				kTexturePixelSize = 16;    // The number of bytes used to represent a pixel in the texture.
		std::string						shadersSrc = shaderstrs;

		// Pipeline objects.
		D3D12_VIEWPORT					m_viewport;
		ComPtr<IDXGIFactory4>			m_factory;
		ComPtr<IDXGISwapChain3>			m_swapChain;
		ComPtr<ID3D12Device>			m_device;
		ComPtr<ID3D12Resource>			m_renderTargets[kFrameCount];
		ComPtr<ID3D12CommandAllocator>	m_commandAllocators[kFrameCount];
		ComPtr<ID3D12CommandQueue>		m_commandQueue;
		ComPtr<ID3D12RootSignature>		m_rootSignature;
		ComPtr<ID3D12DescriptorHeap>	m_rtvHeap;
		ComPtr<ID3D12DescriptorHeap>	m_srvHeap;
		ComPtr<ID3D12PipelineState>		m_pipelineState;
		ComPtr<ID3D12PipelineState>		m_trianglePipelineState;
		ComPtr<ID3D12GraphicsCommandList> m_commandList;
		CD3DX12_RECT					m_scissorRect;
		UINT							m_rtvDescriptorSize;
		LUID							m_dx12deviceluid;

		// App resources.
		ComPtr<ID3D12Resource>			m_vertexBuffer;
		D3D12_VERTEX_BUFFER_VIEW		m_vertexBufferView;

		// App resources.
		ComPtr<ID3D12Resource>			m_triangleVertexBuffer;
		D3D12_VERTEX_BUFFER_VIEW		m_triangleVertexBufferView;
		ComPtr<ID3D12Resource>			m_texture;

		// Synchronization objects.
		UINT							m_frameIndex;
		HANDLE							m_fenceEvent;
		ComPtr<ID3D12Fence>				m_fence;
		UINT64							m_fenceValues[kFrameCount];

		std::shared_ptr<ModuleManager>	m_moduleManager;
		std::unique_ptr<UIModuleManager> m_ui;
		HWND							m_hWnd;

		UINT							m_quadTexWidth;
		UINT							m_quadTexHeight;

		UINT							m_clientWidth;
		UINT							m_clientHeight;

	private:
		void							CreatePipeline();
		void							InitCuda();
		void							LoadAssets();
		void							PopulateCommandList();
		void							MoveToNextFrame();
		void							WaitForGpu();
		std::vector<float>				GenerateTextureData();
		void							CreateRootSignature();
		void							CreateViewportQuad();
		void							CreateViewportTexture();
		void							OnRenderGUI();

		void							GetHardwareAdapter(_In_ IDXGIFactory2* pFactory, _Outptr_result_maybenull_ IDXGIAdapter1** ppAdapter);
	};
}