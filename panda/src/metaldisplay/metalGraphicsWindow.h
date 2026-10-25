/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsWindow.h
 * @author Jace Mayer
 * @date 2026-10-16
 */

#ifndef METALGRAPHICSWINDOW_H
#define METALGRAPHICSWINDOW_H

#include "pandabase.h"
#include "cocoaGraphicsWindow.h"
#include "metalTypes.h"
#include "patomic.h"

/**
 * A Cocoa window that presents its contents through a CAMetalLayer.
 */
class EXPCL_METALDISPLAY MetalGraphicsWindow : public CocoaGraphicsWindow {
public:
  MetalGraphicsWindow(GraphicsEngine *engine, GraphicsPipe *pipe,
                      std::string name,
                      const FrameBufferProperties &fb_prop,
                      const WindowProperties &win_prop,
                      int flags,
                      GraphicsStateGuardian *gsg,
                      GraphicsOutput *host);
  virtual ~MetalGraphicsWindow();

  virtual bool begin_frame(FrameMode mode, Thread *current_thread);
  virtual void end_frame(FrameMode mode, Thread *current_thread);

  virtual bool supports_pixel_zoom() const;

protected:
  virtual void close_window();
  virtual bool open_window();

private:
  bool choose_formats();
  bool update_drawable_size();
  void release_textures();
  void upscale(MetalTexture dest);

  MetalLayer _layer = nil;
  MetalDrawable _drawable = nil;
  MetalTexture _depth_stencil = nil;
  MetalTexture _msaa_color = nil;
  int _sample_count = 1;

  // When video sync is off, we render into our own texture and only present
  // when a drawable is available without waiting, since the window server
  // otherwise paces us to the display refresh rate.
  bool _decoupled = false;
  MetalTexture _offscreen_color = nil;

  // The texture of the last drawable we rendered to, for copying from.
  MetalTexture _last_frame = nil;
  patomic<int> _drawables_in_flight {0};
  LVecBase2i _drawable_size;

  // With a pixel zoom, the scene is rendered at a lower resolution, into
  // _scaled_color, and upscaled to the drawable size when presenting, using
  // MetalFX if available (in which case it upscales into _upscaled first).
  LVecBase2i _render_size;
  MetalTexture _scaled_color = nil;
  MetalTexture _upscaled = nil;
  id _spatial_scaler = nil;

  // These are MTLPixelFormat values.
  unsigned long _color_format = 0;
  unsigned long _depth_stencil_format = 0;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    CocoaGraphicsWindow::init_type();
    register_type(_type_handle, "MetalGraphicsWindow",
                  CocoaGraphicsWindow::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
