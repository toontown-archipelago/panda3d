/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalTypes.h
 * @author Jace Mayer
 * @date 2026-10-14
 */

#ifndef METALTYPES_H
#define METALTYPES_H

#include "pandabase.h"

// Metal objects are Objective-C objects.  So that these headers can still be
// included from plain C++ code, we only pull in the Metal headers when
// compiling Objective-C++, and otherwise refer to them as generic objects.
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

typedef id<MTLDevice> MetalDevice;
typedef id<MTLCommandQueue> MetalCommandQueue;
typedef id<MTLCommandBuffer> MetalCommandBuffer;
typedef id<MTLRenderCommandEncoder> MetalRenderCommandEncoder;
typedef id<MTLComputeCommandEncoder> MetalComputeCommandEncoder;
typedef id<MTLBlitCommandEncoder> MetalBlitCommandEncoder;
typedef id<MTLBuffer> MetalBuffer;
typedef id<MTLTexture> MetalTexture;
typedef id<MTLSamplerState> MetalSamplerState;
typedef id<MTLLibrary> MetalLibrary;
typedef id<MTLFunction> MetalFunction;
typedef id<MTLRenderPipelineState> MetalRenderPipelineState;
typedef id<MTLComputePipelineState> MetalComputePipelineState;
typedef id<MTLDepthStencilState> MetalDepthStencilState;
typedef id<CAMetalDrawable> MetalDrawable;
typedef CAMetalLayer *MetalLayer;
#else
#include <objc/objc.h>

typedef id MetalDevice;
typedef id MetalCommandQueue;
typedef id MetalCommandBuffer;
typedef id MetalRenderCommandEncoder;
typedef id MetalComputeCommandEncoder;
typedef id MetalBlitCommandEncoder;
typedef id MetalBuffer;
typedef id MetalTexture;
typedef id MetalSamplerState;
typedef id MetalLibrary;
typedef id MetalFunction;
typedef id MetalRenderPipelineState;
typedef id MetalComputePipelineState;
typedef id MetalDepthStencilState;
typedef id MetalDrawable;
typedef id MetalLayer;
#endif

#endif
