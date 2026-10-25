/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsBuffer.h
 * @author Jace Mayer
 * @date 2026-10-21
 */

#ifndef METALGRAPHICSBUFFER_H
#define METALGRAPHICSBUFFER_H

#include "pandabase.h"
#include "graphicsBuffer.h"
#include "metalTypes.h"
#include "texture.h"
#include "pvector.h"

/**
 * An offscreen buffer that renders into Metal textures.
 */
class EXPCL_METALDISPLAY MetalGraphicsBuffer : public GraphicsBuffer {
public:
  MetalGraphicsBuffer(GraphicsEngine *engine, GraphicsPipe *pipe,
                      std::string name,
                      const FrameBufferProperties &fb_prop,
                      const WindowProperties &win_prop,
                      int flags,
                      GraphicsStateGuardian *gsg,
                      GraphicsOutput *host);
  virtual ~MetalGraphicsBuffer();

  virtual bool begin_frame(FrameMode mode, Thread *current_thread);
  virtual void end_frame(FrameMode mode, Thread *current_thread);

  virtual bool get_supports_render_texture() const;
  virtual void select_target_tex_page(int page);

protected:
  virtual void close_buffer();
  virtual bool open_buffer();

private:
  bool choose_formats();
  bool rebuild_bitplanes();
  bool bind_texture(Texture *tex, unsigned long format);
  void release_attachments();
  void set_render_target();

  // A color, depth or stencil bitplane.  We render directly into a texture
  // bound with RTM_bind_or_copy, or else into our own.  With multisampling,
  // our own multisample texture is resolved into it.
  struct Bitplane {
    int _plane = 0;               // DrawableRegion::RenderTexturePlane
    unsigned long _format = 0;    // MTLPixelFormat
    PT(Texture) _bound;
    MetalTexture _texture = nil;  // our own texture, if any
    MetalTexture _resolve = nil;  // our own resolve texture, if any
  };
  pvector<Bitplane> _color_planes;
  Bitplane _depth_plane;
  Bitplane _stencil_plane;  // only if separate from the depth plane

  int _sample_count = 1;
  LVecBase2i _attachment_size;
  UpdateSeq _last_textures_seq;
  bool _needs_rebuild = true;
  int _page = 0;

  // These are MTLPixelFormat values.
  unsigned long _color_format = 0;
  unsigned long _depth_stencil_format = 0;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    GraphicsBuffer::init_type();
    register_type(_type_handle, "MetalGraphicsBuffer",
                  GraphicsBuffer::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
