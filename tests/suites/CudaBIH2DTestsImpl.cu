#include "CudaBIH2DTestsImpl.cuh"

#include <kernels/gi2d/tracables/primitives/LineSegment.cuh>
#include <kernels/gi2d/CudaBIH2D.cuh>

using namespace Cuda;

namespace Tests
{
	__host__ void CudaBIH2DTestsImpl::CreateCircleSegments(Cuda::Host::Vector<Cuda::LineSegment>& segments)
	{
		constexpr int kCircleSegs = 10;
		segments.Resize(kCircleSegs);
		for (uint idx = 0; idx < kCircleSegs; ++idx)
		{
			const float theta0 = kTwoPi * float(idx) / float(kCircleSegs);
			const float theta1 = kTwoPi * float(idx + 1) / float(kCircleSegs);
			segments[idx] = GI2D::LineSegment(vec2(std::cos(theta0), std::sin(theta0)) * 0.25f, vec2(std::cos(theta1), std::sin(theta1)) * 0.25f, 0, vec3(1.0f));
		}
		
		// Make sure the line segments are synchronised
		segments.Synchronise(kVectorSyncUpload);
	}

	__host__ void CudaBIH2DTestsImpl::CreateRowSegments(Cuda::Host::Vector<Cuda::LineSegment>& segments)
	{
		constexpr int kRowSegments = 10;
		segments.Resize(kRowSegments);
		for (uint idx = 0; idx < kRowSegments; ++idx)
		{			
			float px = mix(-0.5f, 0.5f, float(idx) / float(kRowSegments - 1));
			segments[idx] = GI2D::LineSegment(vec2(px, -0.1), vec2(px, 0.1), 0, vec3(1.0f));
		}

		// Make sure the line segments are synchronised
		segments.Synchronise(kVectorSyncUpload);
	}

	__host__ void CudaBIH2DTestsImpl::BuildBIH(AssetHandle<Host::BIH2DAsset>& bih, Cuda::Host::Vector<Cuda::LineSegment>& segments, const bool printStats)
	{
		// Prime the list of indices ready for building
		auto& primIdxs = bih->GetPrimitiveIndices();
		primIdxs.resize(segments.Size());
		for (int idx = 0; idx < segments.Size(); ++idx)
		{
			primIdxs[idx] = idx;
		}
		
		/*bih->m_debugFunctor = [](const char* str)
		{
			Logger::WriteMessage(str);
		};*/

		// Construct the BVH
		std::function<BBox2f(uint)> getPrimitiveBBox = [&segments](const uint& idx) -> BBox2f
		{
			return Grow(segments[idx].GetBoundingBox(), 1e-3f);
		};
		bih->Build(getPrimitiveBBox);

		// Print out the stats to the log incase we need to update the baselines
		if (printStats)
		{
			const auto& stats = bih->GetTreeStats();
			Logger::WriteMessage(tfm::format("Build time: %i\n", stats.buildTime).c_str());
			Logger::WriteMessage(tfm::format("Tree depth: %i\n", stats.maxDepth).c_str());
			Logger::WriteMessage(tfm::format("Inner nodes: %i\n", stats.numInnerNodes).c_str());
			Logger::WriteMessage(tfm::format("Leaf nodes: %i\n", stats.numLeafNodes).c_str());
		}
	}
	
	__host__ void CudaBIH2DTestsImpl::BuildSimpleGeometry()
	{
		AssetHandle<Host::BIH2DAsset> bih = CreateAsset<Host::BIH2DAsset>("id_gi2DBIH");
		Host::Vector<LineSegment> segments("cudaVector", 0u, kVectorHostAlloc, nullptr);		
		
		CreateCircleSegments(segments);
		BuildBIH(bih, segments, true);

		// Check that the tree is as expected for this configuration
		constexpr int kRefDepth = 4;
		constexpr int kRefInnerNodes = 10;
		constexpr int kRefLeafNodes = 10;
		const auto& stats = bih->GetTreeStats();
		Assert::IsTrue(stats.maxDepth == kRefDepth, Widen(tfm::format("Tree depth is %i. Expected %i.", stats.maxDepth, kRefDepth)).c_str());
		Assert::IsTrue(stats.numInnerNodes == kRefInnerNodes, Widen(tfm::format("Num inner nodes is %i. Expected %i.", stats.numInnerNodes, kRefInnerNodes)).c_str());
		Assert::IsTrue(stats.numLeafNodes == kRefLeafNodes, Widen(tfm::format("Num leaf nodes is %i. Expected %i.", stats.numLeafNodes, kRefLeafNodes)).c_str());

		bih.DestroyAsset();
	}

	__host__ void CudaBIH2DTestsImpl::PointTestSimpleGeometry()
	{
		AssetHandle<Host::BIH2DAsset> bih = CreateAsset<Host::BIH2DAsset>("id_gi2DBIH");
		Host::Vector<LineSegment> segments("cudaVector", 0u, kVectorHostAlloc, nullptr);

		CreateCircleSegments(segments);
		BuildBIH(bih, segments, true);

		/*const auto& hostNodes = bih->GetHostNodes();
		for (int idx = 0; idx < hostNodes.Size(); ++idx)
		{
			const auto& node = hostNodes[idx];
			if (node.IsLeaf())
			{
				Logger::WriteMessage(tfm::format("Leaf: %i -> %i\n", idx, node.idx).c_str());
			}
			else
			{
				Logger::WriteMessage(tfm::format("Inner: %i -> %i\n", idx, node.GetChildIndex()).c_str());
			}
		}*/

		// Generate random points on each of the line segments and test the tree for a hit
		for (int idx = 0; idx < segments.Size(); ++idx)
		{
			bool isHit = false;
			vec2 p = segments[idx].PointAt(RandFlt());
			bih->TestPrimitive(p,
				[&isHit, &segments, &p](const uint idx) -> void
				{
					isHit |= segments[idx].TestPoint(p, 0.01f);
				});

			Assert::IsTrue(isHit, Widen(tfm::format("Point %s at index %i did not register an intersection.", p.format(), idx)).c_str());
		}

		bih.DestroyAsset();
	}

	__host__ void CudaBIH2DTestsImpl::RayTestSimpleGeometry() 
	{
		AssetHandle<Host::BIH2DAsset> bih = CreateAsset<Host::BIH2DAsset>("id_gi2DBIH");
		Host::Vector<LineSegment> segments("cudaVector", 0u, kVectorHostAlloc, nullptr);

		CreateCircleSegments(segments);
		BuildBIH(bih, segments, true);

		const vec2 o(0.0f);
		const vec2 d = normalize(vec2(1.0f));
				
		int hitSegment = -1;
		auto onRayIntersectLeaf = [&](const uint& idx, float& tNear) -> void
		{
			float t = segments[idx].TestRay(o, d);
			if (t < tNear)
			{
				tNear = t;
				hitSegment = idx;
			}
		};		
		bih->TestRay(o, d, onRayIntersectLeaf);

		bih.DestroyAsset();
	}
	__host__ void CudaBIH2DTestsImpl::RayTestRandomGeometry()
	{
		constexpr int kNumIterations = 1;
		constexpr int kNumRays = 1000;
		constexpr float kSizeLower = 0.001f;
		constexpr float kSizeUpper = 0.5f;
		constexpr int kNumSegmentsLower = 10;
		constexpr int kNumSegmentsUpper = 1000;

		AssetHandle<Host::BIH2DAsset> bih = CreateAsset<Host::BIH2DAsset>("id_gi2DBIH");
		Host::Vector<LineSegment> segments("cudaVector", 0u, kVectorHostAlloc, nullptr);
		
		for (int iterIdx = 0; iterIdx < kNumIterations; ++iterIdx)
		{
			GenerateRandomLineSegments(segments, BBox2f(vec2(-0.5f, 0.5f), vec2(-0.5f, 0.5f)), ivec2(kNumSegmentsLower, kNumSegmentsUpper), vec2(kSizeLower, kSizeUpper), 1);

			// Make sure the line segments are synchronised
			segments.Synchronise(kVectorSyncUpload);
			BuildBIH(bih, segments, false);
			
			for (int rayIdx = 0; rayIdx < kNumRays; ++rayIdx)
			{			
				const vec2 o = RandVec2(-1.0f, 1.0f);
				const float theta = RandFlt() * kPi;
				const vec2 d = vec2(std::cos(theta), std::sin(theta));
				
				// Test the BIH
				int bihHitIdx = -1;
				auto onRayIntersectLeaf = [&, this](const uint& idx, float& tNearest) -> void
				{
					float t = segments[idx].TestRay(o, d);
					if (t < tNearest)
					{
						tNearest = t;
						bihHitIdx = idx;
					}
				};
				bih->TestRay(o, d, onRayIntersectLeaf);
				 
				// Brute-force test the segment list
				float tNear = kFltMax;
				int referenceHitIdx = -1;
				for (int segIdx = 0; segIdx < segments.Size(); ++segIdx)
				{
					const float t = segments[segIdx].TestRay(o, d);
					if (t > 0.0f && t < tNear)
					{
						referenceHitIdx = segIdx;
						tNear = t;
					}
				}

				Assert::IsTrue(bihHitIdx == referenceHitIdx, 
					Widen(tfm::format("BIH reported ray [%s, %s] hit primitive %i. Expected %i.", o.format(), d.format(), bihHitIdx, referenceHitIdx)).c_str());
			}
		}	

		bih.DestroyAsset();
	}
}

/*TEST_CLASS(BoundingIntervalHierarchy2DTests), public SuiteBase
{
public:
	TEST_METHOD(RandomPrimitives)
	{
		constexpr int kNumPrims = 1000;
		constexpr float kBoundSize = 1.0f;
		constexpr float kPrimSize = 0.01f;
		std::vector<BBox2f> primBBoxes(kNumPrims);
		std::vector<uint> primIdxs(kNumPrims);

		// Construct an array of random primitives
		for (int idx = 0; idx < primBBoxes.size(); ++idx)
		{
			vec2 p = RandVec2(-0.5f, 0.5f) * kBoundSize;
			vec2 dp = RandVec2(0.1, 1.0) * kBoundSize * kPrimSize;
			primBBoxes[idx] = BBox2f(p - dp, p + dp);
			primIdxs[idx] = idx;
		}

		AssetHandle<Host::BIH2D> bih = CreateAsset<Host::BIH2D>("bih");
		Host::BIH2DBuilder builder(*bih);

		builder.Build(primIdxs, [&primBBoxes](const uint idx) -> BBox2f& { Assert(idx < primBBoxes.size());  return primBBoxes[idx]; });

		bih->DestroyAsset();
	}
};*/