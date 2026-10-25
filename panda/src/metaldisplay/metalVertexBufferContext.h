/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalVertexBufferContext.h
 * @author Jace Mayer
 * @date 2026-10-17
 */

#ifndef METALVERTEXBUFFERCONTEXT_H
#define METALVERTEXBUFFERCONTEXT_H

#include "pandabase.h"
#include "vertexBufferContext.h"
#include "metalTypes.h"

/**
 * Holds the MTLBuffer containing a GeomVertexArrayData.
 */
class EXPCL_METALDISPLAY MetalVertexBufferContext : public VertexBufferContext {
public:
  MetalVertexBufferContext(PreparedGraphicsObjects *pgo, GeomVertexArrayData *data);
  virtual ~MetalVertexBufferContext();

  ALLOC_DELETED_CHAIN(MetalVertexBufferContext);

  void release_resources();

  MetalBuffer _buffer = nil;

  // The number of the last frame in which the GPU may read from the buffer.
  uint64_t _last_use_frame = 0;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    VertexBufferContext::init_type();
    register_type(_type_handle, "MetalVertexBufferContext",
                  VertexBufferContext::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
