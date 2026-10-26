/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalSamplerContext.h
 * @author Jace Mayer
 * @date 2026-10-18
 */

#ifndef METALSAMPLERCONTEXT_H
#define METALSAMPLERCONTEXT_H

#include "pandabase.h"
#include "samplerContext.h"
#include "metalTypes.h"

/**
 * Holds the MTLSamplerState corresponding to a Panda SamplerState.
 */
class EXPCL_METALDISPLAY MetalSamplerContext : public SamplerContext {
public:
  MetalSamplerContext(const SamplerState &sampler, MetalSamplerState state);
  virtual ~MetalSamplerContext();

  ALLOC_DELETED_CHAIN(MetalSamplerContext);

  void release_resources();

  MetalSamplerState _sampler_state = nil;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    SamplerContext::init_type();
    register_type(_type_handle, "MetalSamplerContext",
                  SamplerContext::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#endif
