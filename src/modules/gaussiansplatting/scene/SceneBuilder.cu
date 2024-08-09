#include "SceneBuilder.cuh"
#include "SceneContainer.cuh"

#include "cameras/PinholeCamera.cuh"
#include "materials/Material.cuh"
#include "lights/QuadLight.cuh"
#include "textures/Texture2D.cuh"
#include "tracables/Primitives.cuh"

#include "core/Vector.cuh"

#define kEmitterPos vec3(0., 0.5, 0.5)
#define kEmitterRot vec3(kHalfPi * 1.5, 0., 0.)
#define kEmitterSca 1.
#define kEmitterPower 2.
#define kEmitterRadiance (kOne * kEmitterPower / sqr(kEmitterSca))

namespace Enso
{
    __host__ bool Host::SceneBuilder::Rebuild(AssetHandle<Host::SceneContainer>& scene)
    {
        Log::Write("Rebuilding scene '%s'...", scene->GetAssetID());
        
        scene->DestroyManagedObjects();
        
        auto& cameras = scene->Cameras();
        auto& tracables = scene->Tracables();

        // Create the primary camera
        const float cameraPhi = -kPi;
        const vec3 cameraPos = vec3(cos(cameraPhi), 0.5, sin(cameraPhi)) * 2.;
        const vec3 cameraLookAt = vec3(0., -0., -0.);
        cameras.EmplaceBack(AssetAllocator::CreateChildAsset<Host::PinholeCamera>(*scene, "pinholecamera", cameraPos, cameraLookAt, 50.));

        constexpr int kNumSpheres = 7;
        BidirectionalTransform transform;
        for (int sphereIdx = 0; sphereIdx < kNumSpheres; ++sphereIdx)
        {
            float phi = kTwoPi * (0.75f + float(sphereIdx) / float(kNumSpheres));
            transform = BidirectionalTransform(vec3(cos(phi), 0.f, sin(phi)) * 0.7f, kZero, 0.2f);
            
            tracables.EmplaceBack(AssetAllocator::CreateChildAsset<Host::Primitive<UnitSphereParams>>(*scene, tfm::format("sphere%i", sphereIdx)));
        }

        // Ground plane        
        transform = BidirectionalTransform(vec3(0.f, -0.2f, 0.f), vec3(-kHalfPi, 0.f, 0.f), 2.f);
        tracables.EmplaceBack(AssetAllocator::CreateChildAsset<Host::Primitive<PlaneParams>>(*scene, "groundplane", transform, PlaneParams{false}));
       
        // Emitter plane
        transform = BidirectionalTransform(kEmitterPos, kEmitterRot, kEmitterSca);
        tracables.EmplaceBack(AssetAllocator::CreateChildAsset<Host::Primitive<PlaneParams>>(*scene, "emitterplane", transform, PlaneParams{ true }));       

        // Synchronise the scene
        scene->Synchronise(kSyncObjects);
        return true;
    }    
}