/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsPipe.mm
 * @author Jace Mayer
 * @date 2026-10-14
 */

#include "metalGraphicsPipe.h"
#include "config_metaldisplay.h"
#include "graphicsOutput.h"
#include "metalGraphicsStateGuardian.h"
#include "metalGraphicsWindow.h"
#include "metalGraphicsBuffer.h"

#import <CoreGraphics/CGDirectDisplayMetal.h>

TypeHandle MetalGraphicsPipe::_type_handle;

/**
 * Takes a CoreGraphics display ID, which defaults to the main display.
 */
MetalGraphicsPipe::
MetalGraphicsPipe(CGDirectDisplayID display) : CocoaGraphicsPipe(display) {
  // Prefer the GPU that is driving the given display.  On Apple Silicon there
  // is only one GPU, but this is still the most correct thing to ask for.
  _device = CGDirectDisplayCopyCurrentMetalDevice(display);
  if (_device == nil) {
    _device = MTLCreateSystemDefaultDevice();
  }

  if (_device == nil) {
    metaldisplay_cat.error()
      << "No Metal device is available.\n";
    _is_valid = false;
    return;
  }

  if (![_device supportsFamily:MTLGPUFamilyApple7]) {
    metaldisplay_cat.error()
      << "Metal device " << [[_device name] UTF8String]
      << " is not supported; an Apple7 GPU family (M1) or newer is required.\n";
    [_device release];
    _device = nil;
    _is_valid = false;
    return;
  }

  if (metaldisplay_cat.is_debug()) {
    metaldisplay_cat.debug()
      << "Using Metal device: " << [[_device name] UTF8String] << "\n"
      << "  unified memory: " << (bool)[_device hasUnifiedMemory] << "\n"
      << "  argument buffers tier: " << (int)[_device argumentBuffersSupport] + 1 << "\n"
      << "  BC texture compression: " << (bool)[_device supportsBCTextureCompression] << "\n"
      << "  max buffer length: " << [_device maxBufferLength] << "\n"
      << "  recommended working set: " << [_device recommendedMaxWorkingSetSize] << "\n";
  }

  _supported_types = OT_window | OT_buffer | OT_texture_buffer;
  _is_valid = true;
}

/**
 *
 */
MetalGraphicsPipe::
~MetalGraphicsPipe() {
  if (_device != nil) {
    [_device release];
    _device = nil;
  }
}

/**
 * Returns the name of the rendering interface associated with this
 * GraphicsPipe.  This is used to present to the user to allow him/her to
 * choose between several possible GraphicsPipes available on a particular
 * platform, so the name should be meaningful and unique for a given platform.
 */
std::string MetalGraphicsPipe::
get_interface_name() const {
  return "Metal";
}

/**
 * This function is passed to the GraphicsPipeSelection object to allow the
 * user to make a default MetalGraphicsPipe.
 */
PT(GraphicsPipe) MetalGraphicsPipe::
pipe_constructor() {
  return new MetalGraphicsPipe;
}

/**
 * Creates a new window or buffer on the pipe, if possible.
 */
PT(GraphicsOutput) MetalGraphicsPipe::
make_output(std::string_view name,
            const FrameBufferProperties &fb_prop,
            const WindowProperties &win_prop,
            int flags,
            GraphicsEngine *engine,
            GraphicsStateGuardian *gsg,
            GraphicsOutput *host,
            int retry,
            bool &precertify) {

  if (!_is_valid) {
    return nullptr;
  }

  if (gsg != nullptr && !gsg->is_of_type(MetalGraphicsStateGuardian::get_class_type())) {
    return nullptr;
  }

  // First thing to try: a MetalGraphicsWindow

  if (retry == 0) {
    if ((flags & BF_require_parasite) != 0 ||
        (flags & BF_refuse_window) != 0 ||
        (flags & BF_resizeable) != 0 ||
        (flags & BF_size_track_host) != 0 ||
        (flags & BF_rtt_cumulative) != 0 ||
        (flags & BF_can_bind_color) != 0 ||
        (flags & BF_can_bind_every) != 0 ||
        (flags & BF_can_bind_layered) != 0) {
      return nullptr;
    }
    return new MetalGraphicsWindow(engine, this, std::string(name), fb_prop, win_prop,
                                   flags, gsg, host);
  }

  // Second thing to try: a MetalGraphicsBuffer

  if (retry == 1) {
    if ((flags & (BF_require_parasite | BF_require_window)) != 0) {
      return nullptr;
    }
    if ((flags & BF_fb_props_optional) == 0) {
      if (fb_prop.get_indexed_color() ||
          fb_prop.get_back_buffers() > 0 ||
          fb_prop.get_accum_bits() > 0 ||
          fb_prop.is_stereo()) {
        return nullptr;
      }
    }
    return new MetalGraphicsBuffer(engine, this, std::string(name), fb_prop, win_prop,
                                   flags, gsg, host);
  }

  // Nothing else left to try.
  return nullptr;
}
