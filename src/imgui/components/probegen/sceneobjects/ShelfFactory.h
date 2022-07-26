#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"
#include "manager/RenderManager.h"

#include "IMGUIAbstractShelf.h"

class IMGUIShelfFactory
{
public:
    IMGUIShelfFactory();

    IMGUIAbstractShelfMap Instantiate(const Json::Document& document, const Cuda::RenderObjectContainer& objectContainer);

private:
    std::map<std::string, std::function<std::shared_ptr<IMGUIAbstractShelf>(const ::Json::Node&)>>    m_instantiators;
};