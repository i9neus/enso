#include "SceneBuilder.cuh"
#include "SceneContainer.cuh"

#include "cameras/PinholeCamera.cuh"
#include "materials/Material.cuh"
#include "lights/QuadLight.cuh"
#include "textures/ProceduralTexture.cuh"
#include "textures/TextureMap.cuh"
#include "tracables/Primitives.cuh"
#include "materials/Lambertian.cuh"

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
       
        // Create the primary camera
        const float cameraPhi = -kPi;
        const vec3 cameraLookAt = vec3(0., 0.1, -0.);
        const vec3 cameraPos = vec3(cos(cameraPhi), 0.5, sin(cameraPhi)) * 2. + cameraLookAt;
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::PinholeCamera>(*scene, "pinholecamera", cameraPos, cameraLookAt, 35.f));
        
        constexpr int kNumPrims = 7;

        // Create some textures
        for (int primIdx = 0; primIdx < kNumPrims; ++primIdx)
        {
            scene->Emplace(AssetAllocator::CreateChildAsset<Host::SolidTexture>(*scene, tfm::format("hue%i", primIdx), Hue(float(primIdx) / float(kNumPrims))));
        }
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::TextureMap>(*scene, "floortexture", "C:\\projects\\enso\\data\\Texture1.exr"));
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::TextureMap>(*scene, "grace", "C:\\projects\\enso\\data\\Grace.exr"));

        // Create some materials
        for (int primIdx = 0; primIdx < kNumPrims; ++primIdx)
        {
            scene->Emplace(AssetAllocator::CreateChildAsset<Host::Lambertian>(*scene, tfm::format("lambert%i", primIdx), primIdx));
        }
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::Lambertian>(*scene, "floorlambert", kNumPrims));

        BidirectionalTransform transform;
        for (int primIdx = 0; primIdx < kNumPrims; ++primIdx)
        {
            float phi = kTwoPi * (0.75f + float(primIdx) / float(kNumPrims));
            transform = BidirectionalTransform(vec3(cos(phi), 0.f, sin(phi)) * 0.7f, kZero, 0.2f);
            
            switch (primIdx % 3)
            {
            case 0:
                scene->Emplace(AssetAllocator::CreateChildAsset<Host::Primitive<UnitSphereParams>>(*scene, tfm::format("ring%i", primIdx), transform, primIdx, UnitSphereParams()));
                break;
            case 1:
                scene->Emplace(AssetAllocator::CreateChildAsset<Host::Primitive<BoxParams>>(*scene, tfm::format("ring%i", primIdx), transform, primIdx, BoxParams(vec3(1.0f))));
                break;
            case 2:
                scene->Emplace(AssetAllocator::CreateChildAsset<Host::Primitive<CylinderParams>>(*scene, tfm::format("ring%i", primIdx), transform, primIdx, CylinderParams(1.f)));
                break;
            }
        }

        // Ground plane        
        transform = BidirectionalTransform(vec3(0.f, -0.2f, 0.f), vec3(-kHalfPi, 0.f, 0.f), 2.f);
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::Primitive<PlaneParams>>(*scene, "groundplane", transform, kNumPrims, PlaneParams(true)));

        // Emitter plane
        transform = BidirectionalTransform(kEmitterPos, kEmitterRot, kEmitterSca);
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::Primitive<PlaneParams>>(*scene, "emitterplane", transform, -1, PlaneParams(true)));

        // Light sampler
        auto emitterTracable = scene->Find<Host::Tracable>("emitterplane");
        scene->Emplace(AssetAllocator::CreateChildAsset<Host::QuadLight>(*scene, "emittersampler", kOne, emitterTracable));

        // Finalise the scene
        //scene->Finalise();         

        // Synchronise the newly created scene objects
        scene->Synchronise(kSyncObjects);
        return true;
    }   
}