#include "SceneBuilder.cuh"
#include "SceneContainer.cuh"

#include "cameras/PinholeCamera.cuh"
#include "materials/Material.cuh"
#include "lights/QuadLight.cuh"
#include "textures/TextureMap.cuh"
#include "tracables/Primitives.cuh"

#include "core/containers/Vector.cuh"

#define kEmitterPos vec3(0., 0.5, 0.5)
#define kEmitterRot vec3(kHalfPi * 1.5, 0., 0.)
#define kEmitterSca 1.
#define kEmitterPower 2.
#define kEmitterRadiance (kOne * kEmitterPower / sqr(kEmitterSca))

namespace Enso
{
    __host__ Host::SceneBuilder::SceneBuilder()
    {
    }

    __host__ bool Host::SceneBuilder::Rebuild(AssetHandle<Host::SceneContainer>& scene)
    {
        Log::Write("Rebuilding scene '%s'...", scene->GetAssetID());
        
        scene->DestroyManagedObjects();
        
        auto& cameras = scene->Cameras();
        auto& tracables = scene->Tracables();
        auto& textures = scene->Textures();

        // Create the primary camera
        const float cameraPhi = -kPi;
        const vec3 cameraLookAt = vec3(0., 0.1, -0.);
        const vec3 cameraPos = vec3(cos(cameraPhi), 0.5, sin(cameraPhi)) * 2. + cameraLookAt;
        cameras.push_back(AssetAllocator::CreateChildAsset<Host::PinholeCamera>(*scene, "pinholecamera", cameraPos, cameraLookAt, 35.f));

        // Create some textures
        textures.push_back(AssetAllocator::CreateChildAsset<Host::TextureMap>(*scene, "floortexture", "C:\\projects\\enso\\data\\Texture1.exr"));
        textures.push_back(AssetAllocator::CreateChildAsset<Host::TextureMap>(*scene, "grace", "C:\\projects\\enso\\data\\Grace.exr"));

        constexpr int kNumPrims = 7;
        BidirectionalTransform transform;
        tracables.resize(7);
        for (int primIdx = 0; primIdx < kNumPrims; ++primIdx)
        {
            float phi = kTwoPi * (0.75f + float(primIdx) / float(kNumPrims));
            transform = BidirectionalTransform(vec3(cos(phi), 0.f, sin(phi)) * 0.7f, kZero, 0.2f);
            
            switch (primIdx % 3)
            {
            case 0:
                tracables[primIdx] = AssetAllocator::CreateChildAsset<Host::Primitive<UnitSphereParams>>(*scene, tfm::format("sphere%i", primIdx), transform, 5, UnitSphereParams());
                break;
            case 1:
                tracables[primIdx] = AssetAllocator::CreateChildAsset<Host::Primitive<BoxParams>>(*scene, tfm::format("box%i", primIdx), transform, 5, BoxParams(vec3(1.0f)));
                break;
            case 2:
                tracables[primIdx] = AssetAllocator::CreateChildAsset<Host::Primitive<CylinderParams>>(*scene, tfm::format("cylinder%i", primIdx), transform, 5, CylinderParams(1.f));
                break;
            }       
        }

        // Ground plane        
        transform = BidirectionalTransform(vec3(0.f, -0.2f, 0.f), vec3(-kHalfPi, 0.f, 0.f), 2.f);
        tracables.push_back(AssetAllocator::CreateChildAsset<Host::Primitive<PlaneParams>>(*scene, "groundplane", transform, 5, PlaneParams{ true }));
       
        // Emitter plane
        transform = BidirectionalTransform(kEmitterPos, kEmitterRot, kEmitterSca);
        tracables.push_back(AssetAllocator::CreateChildAsset<Host::Primitive<PlaneParams>>(*scene, "emitterplane", transform, 5, PlaneParams{ true }));

        for (auto& t : tracables) Log::Debug("  - %i", t.GetReferenceCount());


        // Synchronise the scene
        scene->Synchronise(kSyncObjects);
        return true;
    }    
}