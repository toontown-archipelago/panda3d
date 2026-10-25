/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file spirVFlipFragCoordPass.cxx
 * @author Jace Mayer
 * @date 2026-10-22
 */

#include "spirVFlipFragCoordPass.h"
#include "spirVInstructionCursor.h"

/**
 *
 */
void SpirVFlipFragCoordPass::
run(SpirVModule &module) {
  Id frag_coord_id;
  for (uint32_t word = 0; word < module.get_id_bound(); ++word) {
    Id id(word);
    if (module.get_definition_type(id) == SpirVModule::DT_variable &&
        module.get_function_id(id) == 0 &&
        module.get_storage_class(id) == spv::StorageClassInput &&
        module.get_builtin(id) == spv::BuiltInFragCoord) {
      frag_coord_id = id;
      break;
    }
  }
  if (frag_coord_id == 0) {
    return;
  }

  Id entry_function_id;
  for (size_t i = 0; i < module.get_num_entry_points(); ++i) {
    const EntryPoint &ep = module.get_entry_point(i);
    if (ep.model == spv::ExecutionModelFragment) {
      entry_function_id = ep.function_id;
      break;
    }
  }
  if (entry_function_id == 0) {
    return;
  }

  const ShaderType *vec4_type = ShaderType::register_type(ShaderType::Vector(ShaderType::ST_float, 4));

  // The flipped coordinate is stored in a private variable, which takes the
  // place of gl_FragCoord in the rest of the shader.
  Id flipped_id = module.define_variable(vec4_type, spv::StorageClassPrivate);
  module.set_name(flipped_id, "p3d_FlippedFragCoord");

  for (Function &function : module.modify_functions()) {
    SpirVInstructionCursor cursor(module, function);
    while (cursor.next()) {
      Instruction &op = *cursor;
      switch (op.opcode) {
      case spv::OpLoad:
        if (Id(op.args[2]) == frag_coord_id) {
          op.args[2] = flipped_id;
        }
        break;

      case spv::OpAccessChain:
      case spv::OpInBoundsAccessChain:
        if (Id(op.args[2]) == frag_coord_id) {
          Id pointee_id = module.unwrap_pointer_type(Id(op.args[0]));
          op.args[0] = module.define_pointer_type(pointee_id, spv::StorageClassPrivate);
          op.args[2] = flipped_id;
          module.record_result(op, function.id);
        }
        break;

      default:
        break;
      }
    }
  }

  // Define the uniform block holding the scale and offset.
  {
    ShaderType::Struct struct_type;
    struct_type.add_member(vec4_type, "flip");
    const ShaderType *block_type = ShaderType::register_type(std::move(struct_type));

    Id block_type_id = module.define_type(block_type);
    _flip_block_var_id = module.define_variable(block_type, spv::StorageClassUniform);
    module.decorate(block_type_id, spv::DecorationBlock);
    module.decorate(_flip_block_var_id, spv::DecorationBinding, _binding);
    module.decorate(_flip_block_var_id, spv::DecorationDescriptorSet, _set);
    module.set_name(_flip_block_var_id, "p3d_FragCoordFlip");
  }

  // Compute the flipped coordinate at the start of the entry point.
  SpirVBuilder builder(module);
  builder.set_insertion_point_to_body_start(entry_function_id);

  Id coord = builder.op_load(frag_coord_id);
  Id flip = builder.op_load(builder.op_access_chain(_flip_block_var_id, {module.define_int_constant(0)}));
  Id y = builder.op_composite_extract(coord, {1});
  y = builder.op_multiply(y, builder.op_composite_extract(flip, {0}));
  y = builder.op_add(y, builder.op_composite_extract(flip, {1}));
  builder.op_store(flipped_id, builder.op_composite_insert(y, coord, {1}));
}
