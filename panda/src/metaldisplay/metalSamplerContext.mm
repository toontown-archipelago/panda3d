/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalSamplerContext.mm
 * @author Jace Mayer
 * @date 2026-10-18
 */

#include "metalSamplerContext.h"

TypeHandle MetalSamplerContext::_type_handle;

/**
 *
 */
MetalSamplerContext::
MetalSamplerContext(const SamplerState &sampler, MetalSamplerState state) :
  SamplerContext(sampler),
  _sampler_state(state) {
}

/**
 *
 */
MetalSamplerContext::
~MetalSamplerContext() {
  release_resources();
}

/**
 * Releases the Metal objects held by this context.
 */
void MetalSamplerContext::
release_resources() {
  if (_sampler_state != nil) {
    [_sampler_state release];
    _sampler_state = nil;
  }
}
