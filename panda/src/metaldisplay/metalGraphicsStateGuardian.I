/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsStateGuardian.I
 * @author Jace Mayer
 * @date 2026-10-15
 */

/**
 * Returns the Metal device used by this GSG.
 */
INLINE MetalDevice MetalGraphicsStateGuardian::
get_device() const {
  return _device;
}

/**
 * Returns the command buffer for the current frame, or nil if we are not
 * between begin_frame() and end_frame().
 */
INLINE MetalCommandBuffer MetalGraphicsStateGuardian::
get_command_buffer() const {
  return _cmd;
}

/**
 * Returns the ShaderAttrib that is about to be applied.  Used by the shader
 * context to determine whether the shader inputs have changed.
 */
INLINE const RenderAttrib *MetalGraphicsStateGuardian::
get_target_shader_attrib() const {
  return _target_shader.p();
}

/**
 * Returns a number that uniquely identifies the current render encoder.
 */
INLINE uint64_t MetalGraphicsStateGuardian::
get_encoder_id() const {
  return _encoder_id;
}

/**
 * Returns the shader context whose uniforms are currently bound.
 */
INLINE MetalShaderContext *MetalGraphicsStateGuardian::
get_bound_shader_context() const {
  return _bound_sc;
}

/**
 * Records which shader context's uniforms are currently bound.
 */
INLINE void MetalGraphicsStateGuardian::
set_bound_shader_context(MetalShaderContext *sc) {
  _bound_sc = sc;
}
