/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsPipe.h
 * @author Jace Mayer
 * @date 2026-10-14
 */

#ifndef METALGRAPHICSPIPE_H
#define METALGRAPHICSPIPE_H

#include "pandabase.h"
#include "cocoaGraphicsPipe.h"
#include "metalTypes.h"

/**
 * This graphics pipe represents the interface for rendering with Apple's
 * Metal API.  It owns the MTLDevice that all GSGs created on it share.
 */
class EXPCL_METALDISPLAY MetalGraphicsPipe : public CocoaGraphicsPipe {
public:
  MetalGraphicsPipe(CGDirectDisplayID display = CGMainDisplayID());
  virtual ~MetalGraphicsPipe();

  virtual std::string get_interface_name() const;
  static PT(GraphicsPipe) pipe_constructor();

  INLINE MetalDevice get_device() const;

protected:
  virtual PT(GraphicsOutput) make_output(std::string_view name,
                                         const FrameBufferProperties &fb_prop,
                                         const WindowProperties &win_prop,
                                         int flags,
                                         GraphicsEngine *engine,
                                         GraphicsStateGuardian *gsg,
                                         GraphicsOutput *host,
                                         int retry,
                                         bool &precertify);

private:
  MetalDevice _device;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    CocoaGraphicsPipe::init_type();
    register_type(_type_handle, "MetalGraphicsPipe",
                  CocoaGraphicsPipe::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#include "metalGraphicsPipe.I"

#endif
