/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file config_metaldisplay.mm
 * @author Jace Mayer
 * @date 2026-10-14
 */

#include "config_metaldisplay.h"
#include "config_cocoadisplay.h"
#include "metalGraphicsPipe.h"
#include "metalGraphicsStateGuardian.h"
#include "metalGraphicsWindow.h"
#include "metalGraphicsBuffer.h"
#include "metalShaderContext.h"
#include "metalTextureContext.h"
#include "metalSamplerContext.h"
#include "metalVertexBufferContext.h"
#include "metalIndexBufferContext.h"
#include "metalOcclusionQueryContext.h"
#include "metalGeomMunger.h"
#include "graphicsPipeSelection.h"
#include "dconfig.h"
#include "pandaSystem.h"

#if !defined(CPPPARSER) && !defined(LINK_ALL_STATIC) && !defined(BUILDING_METALDISPLAY)
  #error Buildsystem error: BUILDING_METALDISPLAY not defined
#endif

Configure(config_metaldisplay);
NotifyCategoryDef(metaldisplay, "display");

ConfigureFn(config_metaldisplay) {
  init_libmetaldisplay();
}

ConfigVariableInt metal_frames_in_flight
("metal-frames-in-flight", 3,
 PRC_DESC("The maximum number of frames that the CPU is allowed to get ahead "
          "of the GPU.  Higher values may improve throughput at the cost of "
          "latency and memory usage.  Must be at least 1."));

ConfigVariableBool metal_dump_msl
("metal-dump-msl", false,
 PRC_DESC("Set this true to write the Metal Shading Language code that is "
          "generated from each shader to the metaldisplay log, at the info "
          "level.  This is useful for diagnosing shader translation "
          "problems."));

ConfigVariableString metal_upscaler
("metal-upscaler", "metalfx",
 PRC_DESC("Chooses how a window is upscaled to its full size when it renders "
          "at a lower resolution because of pixel-zoom.  Set this to "
          "\"metalfx\" to use the MetalFX spatial upscaler where available, or "
          "\"linear\" to use plain bilinear filtering."));

ConfigVariableBool metal_debug_labels
("metal-debug-labels", false,
 PRC_DESC("Set this true to attach descriptive labels to Metal objects and "
          "command encoders, which makes GPU captures in Xcode easier to "
          "read.  This has a small CPU cost."));

/**
 * Initializes the library.  This must be called at least once before any of
 * the functions or classes in this library can be used.  Normally it will be
 * called by the static initializers and need not be called explicitly, but
 * special cases exist.
 */
void
init_libmetaldisplay() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;

  init_libcocoadisplay();

  MetalGraphicsPipe::init_type();
  MetalGraphicsStateGuardian::init_type();
  MetalGraphicsWindow::init_type();
  MetalGraphicsBuffer::init_type();
  MetalShaderContext::init_type();
  MetalTextureContext::init_type();
  MetalSamplerContext::init_type();
  MetalVertexBufferContext::init_type();
  MetalIndexBufferContext::init_type();
  MetalOcclusionQueryContext::init_type();
  MetalGeomMunger::init_type();

  GraphicsPipeSelection *selection = GraphicsPipeSelection::get_global_ptr();
  selection->add_pipe_type(MetalGraphicsPipe::get_class_type(),
                           MetalGraphicsPipe::pipe_constructor);

  PandaSystem *ps = PandaSystem::get_global_ptr();
  ps->add_system("Metal");
  ps->set_system_tag("Metal", "window_system", "Cocoa");
}

/**
 * Returns the TypeHandle index of the recommended graphics pipe type defined
 * by this module.
 */
int
get_pipe_type_p3metaldisplay() {
  init_libmetaldisplay();
  return MetalGraphicsPipe::get_class_type().get_index();
}
