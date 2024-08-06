#include "SceneBuilder.cuh"
#include "SceneContainer.cuh"

namespace Enso
{
    __host__ Host::SceneBuilder::SceneBuilder(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx)
    {
        m_scene = AssetAllocator::CreateChildAsset<Host::SceneContainer>(*this, "scenecontainer");
    }

    __host__ Host::SceneBuilder::~SceneBuilder() noexcept
    {
        m_scene.DestroyAsset();
    }   

    __host__ AssetHandle<Host::SceneContainer> Host::SceneBuilder::Rebuild()
    {
        m_scene->DestroyManagedObjects();


        
        return m_scene;
    }    
}