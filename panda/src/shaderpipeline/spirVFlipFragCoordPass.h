/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file spirVFlipFragCoordPass.h
 * @author Jace Mayer
 * @date 2026-10-22
 */

#ifndef SPIRVFLIPFRAGCOORDPASS_H
#define SPIRVFLIPFRAGCOORDPASS_H

#include "spirVTransformPass.h"

/**
 * Rewrites all reads of gl_FragCoord in fragment shaders so that the y
 * coordinate becomes y * flip.x + flip.y, where flip is the first member of a
 * uniform block injected at the given set and binding.  This is used by APIs
 * whose window origin is in the upper-left corner to present the lower-left
 * origin that GLSL shaders expect.
 */
class EXPCL_PANDA_SHADERPIPELINE SpirVFlipFragCoordPass final : public SpirVTransformPass {
public:
  SpirVFlipFragCoordPass(uint32_t set, uint32_t binding) :
    _set(set), _binding(binding) {}

  virtual void run(SpirVModule &module) override;
  virtual std::string_view get_name() const override { return "SpirVFlipFragCoordPass"; }

public:
  const uint32_t _set;
  const uint32_t _binding;

  // The injected uniform block, or 0 if the shader does not use gl_FragCoord.
  Id _flip_block_var_id;
};

#endif
