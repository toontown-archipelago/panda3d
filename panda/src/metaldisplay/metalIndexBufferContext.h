/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalIndexBufferContext.h
 * @author Jace Mayer
 * @date 2026-10-17
 */

#ifndef METALINDEXBUFFERCONTEXT_H
#define METALINDEXBUFFERCONTEXT_H

#include "pandabase.h"
#include "indexBufferContext.h"
#include "metalTypes.h"

/**
 * Holds the MTLBuffer containing the indices of a GeomPrimitive.
 */
class EXPCL_METALDISPLAY MetalIndexBufferContext : public IndexBufferContext {
public:
  MetalIndexBufferContext(PreparedGraphicsObjects *pgo, GeomPrimitive *data);
  virtual ~MetalIndexBufferContext();

  ALLOC_DELETED_CHAIN(MetalIndexBufferContext);

  void release_resources();

  MetalBuffer _buffer = nil;

  // An MTLIndexType value.  Metal does not support 8-bit indices, so these
  // are widened to 16 bits.
  unsigned long _index_type = 0;
  bool _widened = false;

  // The number of the last frame in which the GPU may read from the buffer.
  uint64_t _last_use_frame = 0;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    IndexBufferContext::init_type();
    register_type(_type_handle, "MetalIndexBufferContext",
                  IndexBufferContext::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
