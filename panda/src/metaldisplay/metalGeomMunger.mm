/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGeomMunger.mm
 * @author Jace Mayer
 * @date 2026-10-20
 */

#include "metalGeomMunger.h"
#include "metalGraphicsStateGuardian.h"
#include "config_metaldisplay.h"
#include "config_gobj.h"
#include "shaderAttrib.h"
#include "transformBlendTable.h"

TypeHandle MetalGeomMunger::_type_handle;

ALLOC_DELETED_CHAIN_DEF(MetalGeomMunger);

/**
 *
 */
MetalGeomMunger::
MetalGeomMunger(MetalGraphicsStateGuardian *gsg, const RenderState *state) :
  StandardMunger(gsg, state, 1, NT_packed_dabc, C_color) {

  // Unless a custom shader is applied (which may do its own skinning with a
  // limited number of matrices), hardware-animated-vertices means that we do
  // all the skinning on the GPU, since the Metal renderer can inject it into
  // the shader.
  const ShaderAttrib *shader_attrib;
  state->get_attrib_def(shader_attrib);
  _force_hardware_skinning = hardware_animated_vertices &&
                             shader_attrib->get_shader() == nullptr;
}

/**
 * Given a source GeomVertexData, converts it as necessary for rendering.
 */
CPT(GeomVertexData) MetalGeomMunger::
munge_data_impl(const GeomVertexData *data) {
  CPT(GeomVertexData) new_data = StandardMunger::munge_data_impl(data);
  if (!_force_hardware_skinning) {
    return new_data;
  }

  // If the standard logic left the animation to the CPU, do it on the GPU
  // anyway, if the transform table fits in what the shader can handle.
  const GeomVertexFormat *format = new_data->get_format();
  GeomVertexAnimationSpec animation = format->get_animation();
  if (animation.get_animation_type() != AT_panda ||
      new_data->get_slider_table() != nullptr) {
    return new_data;
  }

  const TransformBlendTable *table = new_data->get_transform_blend_table();
  if (table == nullptr || table->get_num_transforms() == 0 ||
      table->get_max_simultaneous_transforms() > 4 ||
      table->get_num_transforms() > 256) {
    return new_data;
  }

  animation.set_hardware(table->get_max_simultaneous_transforms(), true);
  CPT(GeomVertexFormat) new_format = munge_format(format, animation);
  if (new_format == format) {
    return new_data;
  }
  return new_data->convert_to(new_format);
}

/**
 * Given a source GeomVertexFormat, converts it if necessary to the
 * appropriate format for rendering.
 */
CPT(GeomVertexFormat) MetalGeomMunger::
munge_format_impl(const GeomVertexFormat *orig,
                  const GeomVertexAnimationSpec &animation) {
  if (animation.get_animation_type() != AT_hardware) {
    return orig;
  }

  // Skinning is done on the GPU, so we need to add columns for the blend
  // weights and the indices into the transform table.
  PT(GeomVertexFormat) new_format = new GeomVertexFormat(*orig);
  new_format->set_animation(animation);

  // Make sure the old weights and indices are removed, just in case.
  new_format->remove_column(InternalName::get_transform_weight());
  new_format->remove_column(InternalName::get_transform_index());

  // And we don't need the transform_blend table any more.
  new_format->remove_column(InternalName::get_transform_blend());

  PT(GeomVertexArrayFormat) new_array_format = new GeomVertexArrayFormat;
  new_array_format->add_column
    (InternalName::get_transform_weight(), 4, NT_float32, C_other);
  new_array_format->add_column
    (InternalName::get_transform_index(), 4, NT_uint16, C_index);
  new_format->add_array(new_array_format);

  return GeomVertexFormat::register_format(new_format);
}
