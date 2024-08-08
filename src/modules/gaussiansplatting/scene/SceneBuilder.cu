#include "SceneBuilder.cuh"
#include "SceneContainer.cuh"

namespace Enso
{
    __host__ Host::SceneBuilder::SceneBuilder(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene) :
        Host::GenericObject(initCtx),
        m_scene(scene)
    {
    }

    __host__ Host::SceneBuilder::~SceneBuilder() noexcept
    {
        m_scene.DestroyAsset();
    }   

    __host__ bool Host::SceneBuilder::Rebuild()
    {
        return true;
    }    
}