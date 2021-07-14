#pragma once

#include <stdio.h>

namespace Cuda
{
	__host__ void VerifyTypeSizes();
	__host__ void TestScheduling();
}