/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalVertexBufferContext.mm
 * @author Jace Mayer
 * @date 2026-10-17
 */

#include "metalVertexBufferContext.h"

TypeHandle MetalVertexBufferContext::_type_handle;

/**
 *
 */
MetalVertexBufferContext::
MetalVertexBufferContext(PreparedGraphicsObjects *pgo, GeomVertexArrayData *data) :
  VertexBufferContext(pgo, data) {
}

/**
 *
 */
MetalVertexBufferContext::
~MetalVertexBufferContext() {
  release_resources();
}

/**
 * Releases the Metal objects held by this context.
 */
void MetalVertexBufferContext::
release_resources() {
  if (_buffer != nil) {
    [_buffer release];
    _buffer = nil;
  }
}
