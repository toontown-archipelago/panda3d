/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalOcclusionQueryContext.h
 * @author Jace Mayer
 * @date 2026-10-24
 */

#ifndef METALOCCLUSIONQUERYCONTEXT_H
#define METALOCCLUSIONQUERYCONTEXT_H

#include "pandabase.h"
#include "occlusionQueryContext.h"
#include "metalTypes.h"
#include "pvector.h"

class MetalGraphicsStateGuardian;

/**
 * Counts the samples that passed the depth test, using Metal's visibility
 * result buffer.  Since the counting is done per render pass, a query that
 * spans multiple render passes uses one slot of the buffer for each of them.
 */
class EXPCL_METALDISPLAY MetalOcclusionQueryContext : public OcclusionQueryContext {
public:
  MetalOcclusionQueryContext(MetalGraphicsStateGuardian *gsg);
  virtual ~MetalOcclusionQueryContext();

  ALLOC_DELETED_CHAIN(MetalOcclusionQueryContext);

  virtual bool is_answer_ready() const;
  virtual void waiting_for_answer();
  virtual int get_num_fragments() const;

  void add_slot(MetalBuffer buffer, size_t offset, MetalCommandBuffer cmd);
  void mark_incomplete() { _incomplete = true; }

private:
  MetalGraphicsStateGuardian *_gsg;

  struct Slot {
    MetalBuffer _buffer;
    size_t _offset;
  };
  pvector<Slot> _slots;

  // The command buffer of the last render pass the query spans.  Command
  // buffers complete in order, so when this one is done, so are the others.
  MetalCommandBuffer _cmd = nil;

  // Set if some samples couldn't be counted, in which case we report the
  // object as visible.
  bool _incomplete = false;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    OcclusionQueryContext::init_type();
    register_type(_type_handle, "MetalOcclusionQueryContext",
                  OcclusionQueryContext::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
