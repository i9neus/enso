/*
* Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once

#include "CudaCommonIncludes.h"

void RunSineWaveKernel(unsigned int mesh_width, unsigned int mesh_height, Vertex *cudaDevVertptr, cudaStream_t streamToRun, float AnimTime);