/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGeomMunger.h
 * @author Jace Mayer
 * @date 2026-10-20
 */

#ifndef METALGEOMMUNGER_H
#define METALGEOMMUNGER_H

#include "pandabase.h"
#include "standardMunger.h"

class MetalGraphicsStateGuardian;

/**
 * This specialization on GeomMunger finesses vertices for Metal rendering.
 * Colors are packed into 32-bit BGRA values, which Metal can read directly.
 */
class EXPCL_METALDISPLAY MetalGeomMunger final : public StandardMunger {
public:
  MetalGeomMunger(MetalGraphicsStateGuardian *gsg, const RenderState *state);
  ALLOC_DELETED_CHAIN_DECL(MetalGeomMunger);

private:
  bool _force_hardware_skinning = false;

protected:
  virtual CPT(GeomVertexData) munge_data_impl(const GeomVertexData *data);
  virtual CPT(GeomVertexFormat) munge_format_impl(const GeomVertexFormat *orig,
                                                  const GeomVertexAnimationSpec &animation);

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    StandardMunger::init_type();
    register_type(_type_handle, "MetalGeomMunger",
                  StandardMunger::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
