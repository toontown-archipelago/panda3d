/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalTextureContext.h
 * @author Jace Mayer
 * @date 2026-10-18
 */

#ifndef METALTEXTURECONTEXT_H
#define METALTEXTURECONTEXT_H

#include "pandabase.h"
#include "textureContext.h"
#include "metalTypes.h"

/**
 * Holds the MTLTexture corresponding to a Panda Texture.
 */
class EXPCL_METALDISPLAY MetalTextureContext : public TextureContext {
public:
  MetalTextureContext(PreparedGraphicsObjects *pgo, Texture *tex);
  virtual ~MetalTextureContext();

  ALLOC_DELETED_CHAIN(MetalTextureContext);

  void release_resources();

  MetalTexture _texture = nil;

  // The properties the texture was created with, so that we can tell when it
  // needs to be recreated.
  unsigned long _pixel_format = 0;
  int _width = 0;
  int _height = 0;
  int _depth = 0;
  int _num_levels = 0;
  Texture::TextureType _texture_type = Texture::TT_2d_texture;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    TextureContext::init_type();
    register_type(_type_handle, "MetalTextureContext",
                  TextureContext::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
