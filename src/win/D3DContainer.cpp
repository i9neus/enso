#include "D3DContainer.h"

#include "SecurityAttributes.h"
#include "win/Win32Application.h"
#include "thirdparty/nvidia/helper_cuda.h"

namespace Enso
{
	D3DContainer::D3DContainer(std::string name) :
		m_clientWidth(0),
		m_clientHeight(0),
		m_frameIndex(0),
		m_scissorRect(0, 0, 0, 0),
		m_fenceValues{ 0, 0 },
		m_rtvDescriptorSize(0)
	{
		m_viewport = { 0.0f, 0.0f, 0.0f, 0.0f, };
	}

	void D3DContainer::OnCreate(HWND hWnd)
	{
		/*
			- Create device
			-
		*/

		m_hWnd = hWnd;

		UpdateAssetDimensions();

		// Crtate the renderer
		m_moduleManager = std::make_shared<ModuleManager>();

		// Set up the D3D pipeline
		CreatePipeline();

		// Load the renderer
		m_moduleManager->Initialise(m_dx12deviceluid, GetClientWidth(), GetClientHeight());
		m_moduleManager->LoadRenderer("2dgi");

		// Create the GUI interface
		m_ui = std::make_unique<UIModuleManager>(m_hWnd, m_moduleManager);

		// Setup IMGUI objects
		m_ui->CreateD3DDeviceObjects(m_rootSignature, m_device, 2);

		m_moduleManager->GetRenderer().Start();
	}

	void D3DContainer::OnDestroy()
	{
		// Ensure that the GPU is no longer referencing resources that are about to be
		// cleaned up by the destructor.
		WaitForGpu();

		m_ui->Destroy();
		m_ui.reset();

		m_moduleManager->Destroy();
		m_moduleManager.reset();

		for (int i = 0; i < kFrameCount; ++i)
		{
			ReleaseResource(m_renderTargets[kFrameCount]);
			ReleaseResource(m_commandAllocators[kFrameCount]);
		}

		ReleaseResource(m_triangleVertexBuffer);
		ReleaseResource(m_factory);
		ReleaseResource(m_swapChain);
		ReleaseResource(m_commandQueue);
		ReleaseResource(m_rootSignature);
		ReleaseResource(m_rtvHeap);
		ReleaseResource(m_srvHeap);
		ReleaseResource(m_pipelineState);
		ReleaseResource(m_trianglePipelineState);
		ReleaseResource(m_commandList);
		ReleaseResource(m_vertexBuffer);
		ReleaseResource(m_triangleVertexBuffer);
		ReleaseResource(m_texture);
		ReleaseResource(m_fence);
		CloseHandle(m_fenceEvent);

		ReleaseResource(m_device);
	}

	void D3DContainer::OnUpdate() {}

	void D3DContainer::UpdateAssetDimensions()
	{
		// Update the window dimensions
		RECT clientRect;
		GetClientRect(m_hWnd, &clientRect);
		m_clientWidth = clientRect.right;
		m_clientHeight = clientRect.bottom;
		m_viewport = D3D12_VIEWPORT{ 0.0f, 0.0f, float(m_clientWidth), float(m_clientHeight) };
		m_scissorRect = CD3DX12_RECT(0, 0, LONG(m_clientWidth), LONG(m_clientHeight));

		// Calcualate nearest power-of-two values for the texture
		m_quadTexWidth = max<UINT>(128u, min<UINT>(4096u, 1u << UINT(std::ceil(std::log2(float(m_clientWidth))))));
		m_quadTexHeight = max<UINT>(128u, min<UINT>(4096u, 1u << UINT(std::ceil(std::log2(float(m_clientHeight))))));

		Log::System("D3D quad texture: %i x %i", m_quadTexWidth, m_quadTexHeight);
	}

	void D3DContainer::CreateDevice()
	{
		UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
		// Enable the debug layer (requires the Graphics Tools "optional feature").
		// NOTE: Enabling the debug layer after device creation will invalidate the active device.
		{
			ComPtr<ID3D12Debug> debugController;
			if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
			{
				debugController->EnableDebugLayer();

				// Enable additional debug layers.
				dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
			}
		}
#endif

		ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&m_factory)));

		{
			ComPtr<IDXGIAdapter1> hardwareAdapter;
			GetHardwareAdapter(m_factory.Get(), &hardwareAdapter);

			ThrowIfFailed(D3D12CreateDevice(
				hardwareAdapter.Get(),
				D3D_FEATURE_LEVEL_11_0,
				IID_PPV_ARGS(&m_device)
			));
			DXGI_ADAPTER_DESC1 desc;
			hardwareAdapter->GetDesc1(&desc);
			m_dx12deviceluid = desc.AdapterLuid;
		}

		// Describe and create the command queue.
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

		// Create descriptor heaps.
		{
			// Describe and create a render target view (RTV) descriptor heap.
			D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
			rtvHeapDesc.NumDescriptors = kFrameCount;
			rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

			// Describe and create a shader resource view (SRV) heap for the texture.
			D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
			srvHeapDesc.NumDescriptors = 1;
			srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			ThrowIfFailed(m_device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvHeap)));

			m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}

		// Create the command list.
		{
			// Create command allocator for each frame.
			for (UINT n = 0; n < kFrameCount; n++)
			{
				ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[n])));
			}

			ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocators[0].Get(), m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList)));
		}
	}

	void D3DContainer::CreateRenderTargets()
	{
		// Create frame resources.
		{
			CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

			// Create a RTV and a command allocator for each frame.
			for (UINT n = 0; n < kFrameCount; n++)
			{
				ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
				m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
				rtvHandle.Offset(1, m_rtvDescriptorSize);
			}
		}
	}

	void D3DContainer::CreateSwapChain()
	{
		// Describe and create the swap chain.
		DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
		swapChainDesc.BufferCount = kFrameCount;
		swapChainDesc.Width = m_clientWidth;
		swapChainDesc.Height = m_clientHeight;
		swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.SampleDesc.Count = 1;

		ComPtr<IDXGISwapChain1> swapChain;
		ThrowIfFailed(m_factory->CreateSwapChainForHwnd(
			m_commandQueue.Get(),		// Swap chain needs the queue so that it can force a flush on it.
			m_hWnd,
			&swapChainDesc,
			nullptr,
			nullptr,
			&swapChain
		));

		// This sample does not support fullscreen transitions.
		ThrowIfFailed(m_factory->MakeWindowAssociation(m_hWnd, DXGI_MWA_NO_ALT_ENTER));

		ThrowIfFailed(swapChain.As(&m_swapChain));
		m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	}

	// Load the rendering pipeline dependencies.
	void D3DContainer::CreatePipeline()
	{
		CreateDevice();

		CreateSwapChain();

		CreateRenderTargets();

		CreateRootSignature();

		CreateAssets();

		CreateSynchronisationObjects();
	}

	void D3DContainer::CreateRootSignature()
	{
		D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

		// This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
		featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

		if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
		{
			featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
		}

		CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE);

		CD3DX12_ROOT_PARAMETER1 rootParameters[1];
		rootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);

		D3D12_STATIC_SAMPLER_DESC sampler = {};
		sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.MipLODBias = 0;
		sampler.MaxAnisotropy = 0;
		sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
		sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		sampler.MinLOD = 0.0f;
		sampler.MaxLOD = D3D12_FLOAT32_MAX;
		sampler.ShaderRegister = 0;
		sampler.RegisterSpace = 0;
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, featureData.HighestVersion, &signature, &error));
		ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
	}

	void D3DContainer::CreateViewportQuad()
	{
		float uvX = float(m_clientWidth) / float(m_quadTexWidth);
		float uvY = float(m_clientHeight) / float(m_quadTexHeight);

		// Define the geometry for a triangle.
		VertexUV triangleVertices[] =
		{
			{ { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
			{ { -1.0f, 1.0f, 0.0f }, { 0.0f, uvY } },
			{ { 1.0f, -1.0f, 0.0f }, { uvX, 0.0f } },
			{ { -1.0f, 1.0f, 0.0f }, { 0.0f, uvY } },
			{ { 1.0f, 1.0f, 0.0f }, { uvX, uvY } },
			{ { 1.0f, -1.0f, 0.0f }, { uvX, 0.0f } }
		};

		const UINT vertexBufferSize = sizeof(triangleVertices);

		// Note: using upload heaps to transfer static data like vert buffers is not 
		// recommended. Every time the GPU needs it, the upload heap will be marshalled 
		// over. Please read up on Default Heap usage. An upload heap is used here for 
		// code simplicity and because there are very few verts to actually transfer.
		ThrowIfFailed(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_triangleVertexBuffer)));

		// Copy the triangle data to the vertex buffer.
		UINT8* pVertexDataBegin;
		CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
		ThrowIfFailed(m_triangleVertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
		memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
		m_triangleVertexBuffer->Unmap(0, nullptr);

		// Initialize the vertex buffer view.
		m_triangleVertexBufferView.BufferLocation = m_triangleVertexBuffer->GetGPUVirtualAddress();
		m_triangleVertexBufferView.StrideInBytes = sizeof(VertexUV);
		m_triangleVertexBufferView.SizeInBytes = vertexBufferSize;
	}

	void D3DContainer::CreateViewportTexture()
	{
		// Note: ComPtr's are CPU objects but this resource needs to stay in scope until
		// the command list that references it has finished executing on the GPU.
		// We will flush the GPU at the end of this method to ensure the resource is not
		// prematurely destroyed.
		ComPtr<ID3D12Resource> textureUploadHeap;

		// Create the texture.
		{
			// Describe and create a Texture2D.
			D3D12_RESOURCE_DESC textureDesc = {};
			textureDesc.MipLevels = 1;
			textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			textureDesc.Width = m_quadTexWidth;
			textureDesc.Height = m_quadTexHeight;
			textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
			textureDesc.DepthOrArraySize = 1;
			textureDesc.SampleDesc.Count = 1;
			textureDesc.SampleDesc.Quality = 0;
			textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

			ThrowIfFailed(m_device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_SHARED,
				&textureDesc,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(&m_texture)));

			const UINT64 uploadBufferSize = GetRequiredIntermediateSize(m_texture.Get(), 0, 1);

			// Create the GPU upload buffer.
			ThrowIfFailed(m_device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&textureUploadHeap)));

			// Copy data to the intermediate upload heap and then schedule a copy 
			// from the upload heap to the Texture2D.
			/*std::vector<float> texture = GenerateTextureData();

			D3D12_SUBRESOURCE_DATA textureData = {};
			textureData.pData = &texture[0];
			textureData.RowPitch = m_quadTexWidth * TexturePixelSize;
			textureData.SlicePitch = textureData.RowPitch * m_quadTexHeight;*/

			// Describe and create a SRV for the texture.
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srvDesc.Format = textureDesc.Format;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = 1;

			m_device->CreateShaderResourceView(m_texture.Get(), &srvDesc, m_srvHeap->GetCPUDescriptorHandleForHeapStart());

			m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

			m_moduleManager->LinkD3DOutputTexture(m_device, m_texture, m_quadTexWidth, m_quadTexHeight, m_clientWidth, m_clientHeight);
		}
	}

	void D3DContainer::CreateSynchronisationObjects()
	{
		// Create synchronization objects and wait until assets have been uploaded to the GPU.
		{
			ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex], D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_fence)));

			// Create an event handle to use for frame synchronization.
			m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
			if (m_fenceEvent == nullptr)
			{
				ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
			}

			m_moduleManager->LinkSynchronisationObjects(m_device, m_fence, m_fenceEvent);

			m_fenceValues[m_frameIndex]++;

			// Wait for the command list to execute; we are reusing the same command 
			// list in our main loop but for now, we just want to wait for setup to 
			// complete before continuing.
			WaitForGpu();
		}
	}

	// Load the sample assets.
	void D3DContainer::CreateAssets()
	{
		// Create the pipeline state, which includes compiling and loading shaders.
		{
			ComPtr<ID3DBlob> vertexShader;
			ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
			// Enable better shader debugging with the graphics debugging tools.
			UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
			UINT compileFlags = 0;
#endif

			static const char* shadersHLSL =
				"struct PSInput\
            {\
                float4 position : SV_POSITION;\
                float2 uv : TEXCOORD;\
            };\
            \
            Texture2D g_texture : register(t0);\
            SamplerState g_sampler : register(s0);\
            \
            PSInput VSMain(float4 position : POSITION, float4 uv : TEXCOORD)\
            {\
                PSInput result;\
                result.position = position;\
                result.uv = uv;\
                return result;\
            }\
            \
            float4 PSMain(PSInput input) : SV_TARGET\
            {\
                float4 tx = g_texture.Sample(g_sampler, input.uv);\
                tx.w = 1.0;\
                return tx;\
            }";

			ThrowIfFailed(D3DCompile(shadersHLSL, strlen(shadersHLSL), nullptr, nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
			ThrowIfFailed(D3DCompile(shadersHLSL, strlen(shadersHLSL), nullptr, nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));

			// Define the vertex input layout.
			D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
			{
				{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
				{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
			};

			// Describe and create the graphics pipeline state object (PSO).
			D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
			psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
			psoDesc.pRootSignature = m_rootSignature.Get();
			psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
			psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
			psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
			psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
			psoDesc.DepthStencilState.DepthEnable = FALSE;
			psoDesc.DepthStencilState.StencilEnable = FALSE;
			psoDesc.SampleMask = UINT_MAX;
			psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psoDesc.NumRenderTargets = 1;
			psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
			psoDesc.SampleDesc.Count = 1;
			ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_trianglePipelineState)));
		}

		CreateViewportQuad();

		CreateViewportTexture();

		// Close the command list and execute it to begin the initial GPU setup.
		ThrowIfFailed(m_commandList->Close());
		ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
		m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
	}

	void D3DContainer::PopulateCommandList()
	{
		// Command list allocators can only be reset when the associated 
		// command lists have finished execution on the GPU; apps should use 
		// fences to determine GPU execution progress.
		ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

		// However, when ExecuteCommandList() is called on a particular command 
		// list, that command list can then be reset at any time and must be before 
		// re-recording.
		ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));

		m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());

		ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap.Get() };
		m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

		// Set necessary state.
		m_commandList->SetGraphicsRootDescriptorTable(0, m_srvHeap->GetGPUDescriptorHandleForHeapStart());
		m_commandList->RSSetViewports(1, &m_viewport);
		m_commandList->RSSetScissorRects(1, &m_scissorRect);

		// Indicate that the back buffer will be used as a render target.
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

		///////////////////////////////////

		// Perpare to render the quad
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
		m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

		m_commandList->SetPipelineState(m_trianglePipelineState.Get());

		m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		m_commandList->IASetVertexBuffers(0, 1, &m_triangleVertexBufferView);
		m_commandList->DrawInstanced(6, 1, 0, 0);

		// Indicate that the back buffer will now be used to present.
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

		///////////////////////////////////

		if (m_ui) { m_ui->PopulateCommandList(m_commandList, m_frameIndex); }

		///////////////////////////////////

		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

		ThrowIfFailed(m_commandList->Close());
	}

	// Render the scene.
	void D3DContainer::OnRender()
	{
		// Record all the commands we need to render the scene into the command list.
		PopulateCommandList();

		// Execute the command list.
		ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
		m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		// Present the frame.
		ThrowIfFailed(m_swapChain->Present(1, 0));

		// Schedule a Signal command in the queue.
		const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
		ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

		// After everything's rendered, dispatch any commands that IMGUI may have emitted
		//m_ui.DispatchRenderCommands();

		m_moduleManager->UpdateD3DOutputTexture(m_fenceValues[m_frameIndex]);
		//m_commandQueue->Signal(m_fence.Get(), currentFenceValue + 1);

		// Update the frame index.
		m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

		// If the next frame is not ready to be rendered yet, wait until it is ready.
		if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
		{
			//std::printf("%i is waiting for %i (%i)\n", m_frameIndex, m_fenceValues[m_frameIndex], currentFenceValue);
			ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
			WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
		}

		// Set the fence value for the next frame.
		m_fenceValues[m_frameIndex] = currentFenceValue + 2;

		//std::printf("Frame: %i\n", m_frameIndex);
	}

	// Wait for pending GPU work to complete.
	void D3DContainer::WaitForGpu()
	{
		// Schedule a Signal command in the queue.
		ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

		// Wait until the fence has been processed.
		ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
		WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

		// Increment the fence value for the current frame.
		m_fenceValues[m_frameIndex]++;
	}

	_Use_decl_annotations_
		void D3DContainer::GetHardwareAdapter(IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter)
	{
		ComPtr<IDXGIAdapter1> adapter;
		*ppAdapter = nullptr;

		for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter); ++adapterIndex)
		{
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			{
				// Don't select the Basic Render Driver adapter.
				// If you want a software adapter, pass in "/warp" on the command line.
				continue;
			}

			// Check to see if the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
			{
				break;
			}
		}

		*ppAdapter = adapter.Detach();
	}

	void D3DContainer::OnClientResize(HWND hWnd, UINT width, UINT height, WPARAM wParam)
	{
		if (!m_device || wParam == SIZE_MINIMIZED) { return; }

		//WaitForGpu();

		std::cout << "Here\n";

		/*	ImGui_ImplDX12_InvalidateDeviceObjects();
			CleanupRenderTarget();
			ResizeSwapChain(hWnd, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam));
			CreateRenderTarget();
			ImGui_ImplDX12_CreateDeviceObjects();
		}
		return 0;*/
	}

	void D3DContainer::OnKey(const WPARAM code, const bool isSysKey, const bool isDown)
	{
		if (m_moduleManager) { m_moduleManager->GetRenderer().SetKey(code, isSysKey, isDown); }
	}

	void D3DContainer::OnMouseButton(const int button, const bool isDown)
	{
		if (m_moduleManager) { m_moduleManager->GetRenderer().SetMouseButton(button, isDown); }
	}

	void D3DContainer::OnMouseMove(const int mouseX, const int mouseY, const WPARAM flags)
	{
		if (m_moduleManager) { m_moduleManager->GetRenderer().SetMousePos(mouseX, mouseY, flags); }
	}

	void D3DContainer::OnMouseWheel(const float degrees)
	{
		if (m_moduleManager) { m_moduleManager->GetRenderer().SetMouseWheel(degrees); }
	}
}
