/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalTextureContext.mm
 * @author Jace Mayer
 * @date 2026-10-18
 */

#include "metalTextureContext.h"

TypeHandle MetalTextureContext::_type_handle;

/**
 *
 */
MetalTextureContext::
MetalTextureContext(PreparedGraphicsObjects *pgo, Texture *tex) :
  TextureContext(pgo, tex) {
}

/**
 *
 */
MetalTextureContext::
~MetalTextureContext() {
  release_resources();
}

/**
 * Releases the Metal objects held by this context.
 */
void MetalTextureContext::
release_resources() {
  if (_texture != nil) {
    [_texture release];
    _texture = nil;
  }
}
