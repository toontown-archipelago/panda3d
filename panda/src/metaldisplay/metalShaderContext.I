/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalShaderContext.I
 * @author Jace Mayer
 * @date 2026-10-19
 */

/**
 * Returns the vertex buffer slot used for the given array of the vertex data.
 */
INLINE int MetalShaderContext::
get_vertex_buffer_slot(int array_index) {
  return max_buffer_slot - 1 - array_index;
}

/**
 * Returns the buffer slot used for the given vertex array by the vertex stage
 * of a tessellation pipeline, which runs as a compute kernel.
 */
INLINE int MetalShaderContext::
get_tess_vertex_buffer_slot(int array_index) {
  return 10 + array_index;
}

/**
 * Returns true if skinning can be injected into this shader's vertex stage.
 */
INLINE bool MetalShaderContext::
can_inject_animation() const {
  return !_self_skinning && _anim_locations != 0 && _vertex_module != nullptr;
}

/**
 * Returns true if instancing can be injected into this shader's vertex stage.
 */
INLINE bool MetalShaderContext::
can_inject_instancing() const {
  return _instance_mat_location >= 0 && _vertex_module != nullptr;
}
