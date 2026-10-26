/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalShaderContext.mm
 * @author Jace Mayer
 * @date 2026-10-19
 */

#include "metalShaderContext.h"
#include "metalGraphicsStateGuardian.h"
#include "metalTextureContext.h"
#include "config_metaldisplay.h"
#include "shaderModuleSpirV.h"
#include "spirVTransformer.h"
#include "spirVMakeBlockPass.h"
#include "spirVRemoveUnusedVariablesPass.h"
#include "spirVInjectAlphaTestPass.h"
#include "spirVReplaceVariableTypePass.h"
#include "spirVHoistStructResourcesPass.h"
#include "spirVInjectVertexTransformPass.h"
#include "spirVFlipFragCoordPass.h"
#include "spirVBuilder.h"
#include "geomVertexFormat.h"
#include "clockObject.h"
#include "colorAttrib.h"
#include "colorScaleAttrib.h"
#include "materialAttrib.h"
#include "fogAttrib.h"
#include "lightAttrib.h"
#include "clipPlaneAttrib.h"
#include "texMatrixAttrib.h"
#include "textureAttrib.h"
#include "textureStage.h"
#include "pset.h"
#include "texGenAttrib.h"
#include "renderModeAttrib.h"

// Note: this module is compiled with exceptions enabled, since SPIRV-Cross
// reports translation errors by throwing.
#include <spirv_cross/spirv_msl.hpp>
#include <functional>

#import <Foundation/Foundation.h>

TypeHandle MetalShaderContext::_type_handle;

namespace {
/**
 * Returns the newest Metal Shading Language version supported by the OS, as
 * a major * 100 + minor * 10 number, and the corresponding MTLLanguageVersion.
 */
int
get_msl_version(MTLLanguageVersion &language_version) {
#ifdef __MAC_14_0
  if (@available(macOS 14.0, *)) {
    language_version = MTLLanguageVersion3_1;
    return 310;
  }
#endif
#ifdef __MAC_13_0
  if (@available(macOS 13.0, *)) {
    language_version = MTLLanguageVersion3_0;
    return 300;
  }
#endif
  if (@available(macOS 12.0, *)) {
    language_version = MTLLanguageVersion2_4;
    return 240;
  }
  language_version = MTLLanguageVersion2_3;
  return 230;
}

/**
 * Adds a dummy input for each given location that a tessellation stage does
 * not use, so that SPIRV-Cross lays out its inputs the same way as the
 * outputs of the previous stage, which are read from a buffer.
 */
class SpirVKeepInputsPass final : public SpirVTransformPass {
public:
  struct Location {
    uint32_t _location;
    uint32_t _vecsize;
    bool _per_patch;
  };

  SpirVKeepInputsPass(pvector<Location> locations) : _locations(std::move(locations)) {}

  virtual void run(SpirVModule &module) override {
    pset<uint32_t> present;
    for (uint32_t word = 0; word < module.get_id_bound(); ++word) {
      Id id(word);
      if (module.get_definition_type(id) == SpirVModule::DT_variable &&
          module.get_function_id(id) == 0 &&
          module.get_storage_class(id) == spv::StorageClassInput &&
          module.get_location(id) >= 0) {
        // Mark all the locations taken up by the input.  Per-vertex inputs
        // are arrays over the patch, which doesn't count.
        const ShaderType *type = module.resolve_type(id);
        if (type != nullptr && !module.has_decoration(id, spv::DecorationPatch) &&
            type->as_array() != nullptr) {
          type = type->as_array()->get_element_type();
        }
        uint32_t count = (type != nullptr) ? std::max(type->get_num_interface_locations(), 1) : 1;
        for (uint32_t i = 0; i < count; ++i) {
          present.insert((uint32_t)module.get_location(id) + i);
        }
      }
    }
    if (module.get_num_entry_points() == 0) {
      return;
    }
    Id function_id = module.get_entry_point(0).function_id;

    // The previous stage also wrote gl_Position, so read that too.  It is in
    // the gl_PerVertex block, which we declare if this stage doesn't.
    Id block_var_id;
    uint32_t position_member = 0;
    for (uint32_t word = 0; word < module.get_id_bound() && block_var_id == 0; ++word) {
      Id id(word);
      if (module.get_definition_type(id) != SpirVModule::DT_variable ||
          module.get_function_id(id) != 0 ||
          module.get_storage_class(id) != spv::StorageClassInput) {
        continue;
      }
      Id type_id = module.unwrap_pointer_type(module.get_type_id(id));
      const ShaderType *type = module.resolve_type(id);
      if (type != nullptr && type->as_array() != nullptr) {
        type_id = module.get_composite_member_type_id(type_id, 0);
        type = type->as_array()->get_element_type();
      }
      if (type == nullptr || type->as_struct() == nullptr) {
        continue;
      }
      for (uint32_t m = 0; m < module.get_num_members(type_id); ++m) {
        if (module.get_member_builtin(type_id, m) == spv::BuiltInPosition) {
          block_var_id = id;
          position_member = m;
          break;
        }
      }
    }
    if (block_var_id == 0) {
      ShaderType::Struct block;
      block.add_member(ShaderType::register_type(ShaderType::Vector(ShaderType::ST_float, 4)), "gl_Position");
      const ShaderType *block_type = ShaderType::register_type(std::move(block));
      Id block_type_id = module.define_type(block_type);
      module.decorate(block_type_id, spv::DecorationBlock);
      module.decorate_member(block_type_id, 0, spv::DecorationBuiltIn, spv::BuiltInPosition);
      block_var_id = module.define_variable(ShaderType::register_type(ShaderType::Array(block_type, 32)),
                                            spv::StorageClassInput);
    }
    {
      SpirVBuilder builder(module);
      builder.set_insertion_point_to_body_start(function_id);
      builder.op_load(builder.op_access_chain(block_var_id,
        {module.define_int_constant(0), module.define_int_constant(position_member)}));
    }

    for (const Location &loc : _locations) {
      if (present.count(loc._location)) {
        continue;
      }
      const ShaderType *type = (loc._vecsize <= 1) ? (const ShaderType *)ShaderType::FLOAT
        : ShaderType::register_type(ShaderType::Vector(ShaderType::ST_float, loc._vecsize));
      if (!loc._per_patch) {
        // Per-vertex inputs are arrays over the input patch.
        type = ShaderType::register_type(ShaderType::Array(type, 32));
      }
      Id var_id = module.define_variable(type, spv::StorageClassInput);
      module.decorate(var_id, spv::DecorationLocation, loc._location);
      if (loc._per_patch) {
        module.decorate(var_id, spv::DecorationPatch);
      }

      SpirVBuilder builder(module);
      builder.set_insertion_point_to_body_start(function_id);
      if (loc._per_patch) {
        builder.op_load(var_id);
      } else {
        builder.op_load(builder.op_access_chain(var_id, {module.define_int_constant(0)}));
      }
      present.insert(loc._location);
    }
  }
  virtual std::string_view get_name() const override { return "SpirVKeepInputsPass"; }

private:
  pvector<Location> _locations;
};
}


/**
 * Orders pipeline keys so that they can be used in a map.
 */
bool MetalShaderContext::PipelineKey::
operator < (const PipelineKey &other) const {
  if (_format != other._format) {
    return _format < other._format;
  }
  int cmp = memcmp(_color_formats, other._color_formats, sizeof(_color_formats));
  if (cmp != 0) {
    return cmp < 0;
  }
  if (_depth_format != other._depth_format) {
    return _depth_format < other._depth_format;
  }
  if (_stencil_format != other._stencil_format) {
    return _stencil_format < other._stencil_format;
  }
  if (_sample_count != other._sample_count) {
    return _sample_count < other._sample_count;
  }
  if (_blend != other._blend) {
    return _blend < other._blend;
  }
  if (_color_write_mask != other._color_write_mask) {
    return _color_write_mask < other._color_write_mask;
  }
  if (_alpha_test_mode != other._alpha_test_mode) {
    return _alpha_test_mode < other._alpha_test_mode;
  }
  if (_alpha_test_ref != other._alpha_test_ref) {
    return _alpha_test_ref < other._alpha_test_ref;
  }
  if (_flat_color != other._flat_color) {
    return _flat_color < other._flat_color;
  }
  if (_animated != other._animated) {
    return _animated < other._animated;
  }
  if (_instanced != other._instanced) {
    return _instanced < other._instanced;
  }
  return memcmp(_uv_names, other._uv_names, sizeof(_uv_names)) < 0;
}

/**
 * Constructs a shader context.  Follow this up with a call to
 * create_functions().
 */
MetalShaderContext::
MetalShaderContext(Shader *shader, Shader *source) :
  ShaderContext(shader),
  _source(source != nullptr ? source : shader) {
  shader = _source;
  _subsumes_alpha_test = shader->_subsumes_alpha_test;
  _matrix_cache = pvector<LMatrix4>(shader->_matrix_cache_desc.size(), LMatrix4::ident_mat());
  _matrix_cache_deps = shader->_matrix_cache_deps;

  int next_unused_location = 0;
  for (const Shader::VertexInputBinding &bind : shader->_vertex_inputs) {
    VertexInput input;
    input._name = bind._name;
    input._append_uv = bind._append_uv;
    input._location = bind._id._location;
    if (input._append_uv >= 0) {
      _uses_uv_names = true;
    }
    input._scalar_type = ShaderType::ST_float;
    input._num_components = 4;

    ShaderType::ScalarType scalar_type;
    uint32_t num_elements, num_rows, num_columns;
    if (bind._id._type->as_scalar_type(scalar_type, num_elements, num_rows, num_columns)) {
      input._scalar_type = scalar_type;
      input._num_components = num_columns;
    }

    if (bind._name == InternalName::get_color()) {
      _uses_vertex_color = true;
    }

    // Determine which inputs need to be transformed by the skinning matrices.
    const InternalName *name = bind._name;
    if (name == InternalName::get_transform_index() ||
        name == InternalName::get_transform_weight()) {
      // The shader does its own skinning.
      _self_skinning = true;

      // The indices are stored as plain integers, which Metal can't convert
      // to floating-point while fetching them, so a shader that declares
      // them as floats has its input changed to an unsigned integer type.
      if (name == InternalName::get_transform_index() &&
          input._scalar_type == ShaderType::ST_float) {
        input._scalar_type = ShaderType::ST_uint;
        _retyped_index_location = input._location;
      }
    }
    else if (name == InternalName::get_vertex()) {
      _anim_locations |= (1u << input._location);
      _anim_point_locations |= (1u << input._location);
    }
    else if (name == InternalName::get_normal() ||
             name == InternalName::get_tangent() ||
             name == InternalName::get_binormal() ||
             name->get_parent() == InternalName::get_tangent() ||
             name->get_parent() == InternalName::get_binormal()) {
      _anim_locations |= (1u << input._location);
    }
    next_unused_location = std::max(next_unused_location,
      input._location + (int)bind._id._type->get_num_interface_locations());

    _vertex_inputs.push_back(std::move(input));
  }

  if (_anim_locations != 0 && !_self_skinning) {
    _transform_index_location = next_unused_location++;
    _transform_weight_location = next_unused_location++;
  }

  // Determine which inputs need to be multiplied by the instance matrix.
  for (const Shader::Parameter &param : shader->_parameters) {
    if (param._binding == nullptr) {
      continue;
    }
    bool inverse, transpose;
    if ((param._stage_mask & (1 << (int)Shader::Stage::VERTEX)) != 0 &&
        param._binding->has_modelview_matrix(inverse, transpose)) {
      _model_matrices[param._name] = std::make_pair(inverse, transpose);
    }
  }
  if (!_model_matrices.empty()) {
    // The instance matrix is a mat3x4, which takes up three locations.
    _instance_mat_location = next_unused_location;
    next_unused_location += 3;
  }
}

/**
 *
 */
MetalShaderContext::
~MetalShaderContext() {
  release_resources();
}

/**
 * Releases the Metal objects held by this context.
 */
void MetalShaderContext::
release_resources() {
  for (MetalFunction &function : _vertex_variants) {
    if (function != nil) {
      [function release];
      function = nil;
    }
  }
  for (auto &item : _alpha_test_functions) {
    [item.second release];
  }
  _alpha_test_functions.clear();

  for (auto &item : _alpha_test_libraries) {
    [item.second release];
  }
  _alpha_test_libraries.clear();
  for (auto &item : _pipelines) {
    [item.second release];
  }
  _pipelines.clear();

  if (_vertex_function != nil) {
    [_vertex_function release];
    _vertex_function = nil;
  }
  if (_fragment_function != nil) {
    [_fragment_function release];
    _fragment_function = nil;
  }
  if (_compute_function != nil) {
    [_compute_function release];
    _compute_function = nil;
  }

  for (auto &item : _tess_vertex_pipelines) {
    [item.second release];
  }
  _tess_vertex_pipelines.clear();
  id *objects[] = {&_tess_control_pipeline, &_tess_control_function, &_tess_eval_function};
  for (id *object : objects) {
    if (*object != nil) {
      [*object release];
      *object = nil;
    }
  }
}

/**
 * Translates all of the shader's modules to MSL and compiles them into Metal
 * functions.  Returns false if this failed.
 */
bool MetalShaderContext::
create_functions(MetalGraphicsStateGuardian *gsg) {
  // Sort the parameters into the uniform block and the texture slots.
  ShaderType::Struct block_struct;
  pvector<const InternalName *> block_names;

  for (size_t pi = 0; pi < _source->_parameters.size(); ++pi) {
    const Shader::Parameter &param = _source->_parameters[pi];
    if (param._binding == nullptr) {
      continue;
    }

    // Separate out the textures, which may be nested in structs or arrays.
    const ShaderType *type = param._type;
    if (type->contains_opaque_type()) {
      pmap<pvector<uint32_t>, pvector<TextureBinding> > groups;
      int resource_index = 0;
      bool supported = true;
      type = r_extract_resources(param, pvector<uint32_t>(), groups, type,
                                 resource_index, supported);
      if (!supported) {
        metaldisplay_cat.error()
          << "Shader input " << *param._name << " has type " << *param._type
          << ", which is not yet supported by the Metal renderer.\n";
        return false;
      }

      for (auto &item : groups) {
        ResourceGroup group;
        group._param_index = pi;
        group._chain = item.first;
        group._first_slot = (uint32_t)_textures.size();
        _resource_groups.push_back(std::move(group));
        for (TextureBinding &binding : item.second) {
          _textures.push_back(std::move(binding));
        }
      }

      if (type == nullptr) {
        // It contained only textures.
        continue;
      }
    }

    // Skinning matrix tables go into their own buffer.
    if (_table_binding == nullptr && param._name->get_name() == "p3d_TransformTable" &&
        (param._binding->get_state_dep() & ~(Shader::D_vertex_data | Shader::D_frame)) == 0) {
      ShaderType::Struct table_struct;
      table_struct.add_member(type, param._name->get_name());
      _table_block_type = ShaderType::register_type(std::move(table_struct))->as_struct();
      _table_binding = param._binding;
      _table_name = param._name.p();
      continue;
    }

    // Booleans have no defined memory layout, so we store them as integers.
    const ShaderType *block_member_type = type->replace_scalar_type(ShaderType::ST_bool, ShaderType::ST_int);
    if (block_member_type != type) {
      _bool_members.push_back({param._name.p(), block_member_type});
    }

    block_struct.add_member(block_member_type, param._name->get_name());
    size_t offset = block_struct.get_member(block_struct.get_num_members() - 1).offset;
    _uniform_bindings.push_back({param._binding, offset});
    _uniform_deps |= param._binding->get_state_dep();
    block_names.push_back(param._name.p());
  }

  if (block_struct.get_num_members() > 0) {
    _block_type = ShaderType::register_type(std::move(block_struct))->as_struct();
    _block_names = std::move(block_names);
    _uniform_data.resize(_block_type->get_size_bytes(), 0);
  }

  // Process the stages in pipeline order, since with tessellation, each stage
  // needs to know the memory layout of the outputs of the previous stage.
  pvector<CPT(ShaderModule)> modules;
  for (Shader::LinkedModule &linked_module : _source->_modules) {
    CPT(ShaderModule) module = linked_module._module.get_read_pointer();
    nassertr(module != nullptr, false);
    modules.push_back(module);
  }
  std::sort(modules.begin(), modules.end(),
    [](const CPT(ShaderModule) &a, const CPT(ShaderModule) &b) {
      return (int)a->get_stage() < (int)b->get_stage();
    });

  bool has_tess_control = false;
  for (const CPT(ShaderModule) &module : modules) {
    Shader::Stage stage = module->get_stage();
    if (stage == Shader::Stage::TESS_CONTROL) {
      has_tess_control = true;
    }
    else if (stage == Shader::Stage::TESS_EVALUATION) {
      _is_tessellated = true;
      if (!read_tess_eval_modes(module)) {
        return false;
      }
    }
  }
  if (_is_tessellated && !has_tess_control) {
    metaldisplay_cat.error()
      << "Shader " << _source->get_debug_name() << " has a tessellation evaluation "
         "stage without a tessellation control stage, which is not supported by "
         "the Metal renderer.\n";
    return false;
  }

  bool success = true;
  for (const CPT(ShaderModule) &module : modules) {
    Shader::Stage stage = module->get_stage();
    if (stage == Shader::Stage::GEOMETRY) {
      metaldisplay_cat.error()
        << "Shader " << _source->get_debug_name() << " uses a " << stage
        << " shader, which is not supported by the Metal renderer.\n";
      success = false;
      continue;
    }

    std::string entry_point;
    MetalLibrary library = compile_library(gsg, module, RenderAttrib::M_none, 0, entry_point);
    if (library == nil) {
      success = false;
      continue;
    }

    MetalFunction function = [library newFunctionWithName:[NSString stringWithUTF8String:entry_point.c_str()]];
    [library release];

    if (function == nil) {
      metaldisplay_cat.error()
        << "Could not find entry point " << entry_point << " in "
        << _source->get_debug_name() << "\n";
      success = false;
      continue;
    }

    if (metal_debug_labels) {
      function.label = [NSString stringWithUTF8String:_source->get_debug_name().c_str()];
    }

    switch (stage) {
    case Shader::Stage::VERTEX:
      _vertex_function = function;
      _vertex_module = module;
      break;
    case Shader::Stage::TESS_CONTROL:
      _tess_control_function = function;
      break;
    case Shader::Stage::TESS_EVALUATION:
      _tess_eval_function = function;
      break;
    case Shader::Stage::FRAGMENT:
      _fragment_function = function;
      _fragment_module = module;
      break;
    case Shader::Stage::COMPUTE:
      _compute_function = function;
      _is_compute = true;
      break;
    default:
      [function release];
      break;
    }
  }

  if (success && !_is_compute && _vertex_function == nil) {
    metaldisplay_cat.error()
      << "Shader " << _source->get_debug_name() << " has no vertex stage.\n";
    success = false;
  }
  return success;
}

/**
 * Removes the textures from the given type, adding them to the given map,
 * keyed by the chain of struct members leading to them.  Returns the type
 * without the textures, or nullptr if nothing is left.
 */
const ShaderType *MetalShaderContext::
r_extract_resources(const Shader::Parameter &param, const pvector<uint32_t> &chain,
                    pmap<pvector<uint32_t>, pvector<TextureBinding> > &groups,
                    const ShaderType *type, int &resource_index, bool &supported) {

  if (const ShaderType::Array *array_type = type->as_array()) {
    // All of the elements share the same chain, since they end up in the same
    // (hoisted) array.
    const ShaderType *element_type = array_type->get_element_type();
    const ShaderType *new_element_type = nullptr;
    uint32_t count = array_type->get_num_elements();
    for (uint32_t i = 0; i < count; ++i) {
      new_element_type = r_extract_resources(param, chain, groups, element_type,
                                             resource_index, supported);
    }
    if (new_element_type != nullptr) {
      return ShaderType::register_type(ShaderType::Array(new_element_type, count));
    }
    return nullptr;
  }

  if (const ShaderType::Struct *struct_type = type->as_struct()) {
    // Keep the original offsets, since that's how the data is laid out.
    ShaderType::Struct new_struct;
    for (uint32_t i = 0; i < struct_type->get_num_members(); ++i) {
      const ShaderType::Struct::Member &member = struct_type->get_member(i);
      pvector<uint32_t> member_chain(chain);
      member_chain.push_back(i);
      const ShaderType *new_member_type =
        r_extract_resources(param, member_chain, groups, member.type,
                            resource_index, supported);
      if (new_member_type != nullptr) {
        new_struct.add_member(new_member_type, member.name, member.offset);
      }
    }
    if (new_struct.get_num_members() > 0) {
      return ShaderType::register_type(std::move(new_struct));
    }
    return nullptr;
  }

  if (type->as_resource() == nullptr) {
    return type;
  }

  if (const ShaderType::SampledImage *sampled_image = type->as_sampled_image()) {
    TextureBinding binding;
    binding._binding = param._binding;
    binding._resource_id = param._binding->get_resource_id(resource_index++);
    binding._texture_type = sampled_image->get_texture_type();
    groups[chain].push_back(std::move(binding));
  } else {
    // Images, storage buffers and separate samplers are not yet supported.
    supported = false;
  }
  return nullptr;
}

/**
 * Reads the execution modes of the given tessellation evaluation module that
 * determine how the patches are tessellated.  Returns false if they aren't
 * supported by Metal.
 */
bool MetalShaderContext::
read_tess_eval_modes(const ShaderModule *module) {
  if (!module->is_of_type(ShaderModuleSpirV::get_class_type())) {
    return false;
  }
  const ShaderModuleSpirV *spv_module = (const ShaderModuleSpirV *)module;
  try {
    spirv_cross::Compiler compiler(spv_module->get_data(), spv_module->get_data_size());
    const spirv_cross::Bitset &modes = compiler.get_execution_mode_bitset();

    if (modes.get(spv::ExecutionModeIsolines)) {
      metaldisplay_cat.error()
        << "Shader " << _source->get_debug_name() << " tessellates isolines, "
           "which is not supported by the Metal renderer.\n";
      return false;
    }
    if (modes.get(spv::ExecutionModePointMode)) {
      metaldisplay_cat.warning()
        << "Shader " << _source->get_debug_name() << " uses point mode "
           "tessellation, which is not supported by the Metal renderer.\n";
    }

    _tess_quads = modes.get(spv::ExecutionModeQuads);
    _tess_eval_modes.push_back(_tess_quads ? spv::ExecutionModeQuads : spv::ExecutionModeTriangles);

    if (modes.get(spv::ExecutionModeSpacingFractionalEven)) {
      _tess_partition_mode = MTLTessellationPartitionModeFractionalEven;
      _tess_eval_modes.push_back(spv::ExecutionModeSpacingFractionalEven);
    } else if (modes.get(spv::ExecutionModeSpacingFractionalOdd)) {
      _tess_partition_mode = MTLTessellationPartitionModeFractionalOdd;
      _tess_eval_modes.push_back(spv::ExecutionModeSpacingFractionalOdd);
    } else {
      _tess_partition_mode = MTLTessellationPartitionModeInteger;
      _tess_eval_modes.push_back(spv::ExecutionModeSpacingEqual);
    }

    _tess_cw = modes.get(spv::ExecutionModeVertexOrderCw);
    _tess_eval_modes.push_back(_tess_cw ? spv::ExecutionModeVertexOrderCw : spv::ExecutionModeVertexOrderCcw);
  }
  catch (const spirv_cross::CompilerError &error) {
    metaldisplay_cat.error()
      << "Failed to reflect tessellation evaluation stage of "
      << _source->get_debug_name() << ": " << error.what() << "\n";
    return false;
  }
  return true;
}

/**
 * Remembers the outputs of the given stage, which are written to a buffer, so
 * that the next stage can be made to read them with the same layout.  Also
 * computes an upper bound on the size of each record in that buffer.
 */
void MetalShaderContext::
record_stage_outputs(spirv_cross::CompilerMSL &compiler, int model) {
  _prev_stage_outputs.clear();
  size_t vertex_size = 0;
  size_t patch_size = 0;

  // Adds an entry for each location taken up by a value of the given type,
  // descending into structs (interface blocks), arrays and matrix columns.
  std::function<void(const spirv_cross::SPIRType &, const pvector<uint32_t> &, uint32_t &, bool)> add_type;
  add_type = [&](const spirv_cross::SPIRType &type, const pvector<uint32_t> &dims,
                 uint32_t &location, bool per_patch) {
    uint32_t num_elements = 1;
    for (uint32_t dim : dims) {
      num_elements *= std::max(dim, 1u);
    }
    for (uint32_t e = 0; e < num_elements; ++e) {
      if (type.basetype == spirv_cross::SPIRType::Struct) {
        for (uint32_t m = 0; m < (uint32_t)type.member_types.size(); ++m) {
          if (compiler.has_member_decoration(type.self, m, spv::DecorationLocation)) {
            location = compiler.get_member_decoration(type.self, m, spv::DecorationLocation);
          }
          const spirv_cross::SPIRType &member_type = compiler.get_type(type.member_types[m]);
          pvector<uint32_t> member_dims(member_type.array.begin(), member_type.array.end());
          add_type(member_type, member_dims, location, per_patch);
        }
        continue;
      }
      for (uint32_t c = 0; c < std::max(type.columns, 1u); ++c) {
        InterfaceVariable var;
        var._location = location++;
        var._component = 0;
        var._vecsize = type.vecsize;
        var._builtin = -1;
        var._per_patch = per_patch;
        _prev_stage_outputs.push_back(var);
        // Each location takes up at most 16 bytes (or 32 for doubles).
        (per_patch ? patch_size : vertex_size) += (type.width > 32) ? 32 : 16;
      }
    }
  };

  spirv_cross::ShaderResources resources = compiler.get_shader_resources();
  for (const spirv_cross::Resource &res : resources.stage_outputs) {
    const spirv_cross::SPIRType &type = compiler.get_type(res.type_id);
    bool per_patch = compiler.has_decoration(res.id, spv::DecorationPatch);
    uint32_t location = compiler.get_decoration(res.id, spv::DecorationLocation);

    // The per-vertex outputs of a tessellation control stage are arrays with
    // an element for each control point; SPIRV-Cross stores the outermost
    // dimension last.
    pvector<uint32_t> dims(type.array.begin(), type.array.end());
    if (model == spv::ExecutionModelTessellationControl && !per_patch && !dims.empty()) {
      dims.pop_back();
    }
    size_t first = _prev_stage_outputs.size();
    add_type(type, dims, location, per_patch);
    uint32_t component = compiler.get_decoration(res.id, spv::DecorationComponent);
    if (first < _prev_stage_outputs.size()) {
      _prev_stage_outputs[first]._component = component;
    }
  }

  // gl_Position is written as well.  (The other built-ins aren't passed on,
  // and the tessellation levels go into the tessellation factor buffer.)
  for (const spirv_cross::BuiltInResource &res : resources.builtin_outputs) {
    if (res.builtin != spv::BuiltInPosition) {
      continue;
    }
    InterfaceVariable var;
    var._location = 0;
    var._component = 0;
    var._vecsize = 4;
    var._builtin = (int)res.builtin;
    var._per_patch = false;
    _prev_stage_outputs.push_back(var);
    vertex_size += 16;
  }

  // Leave some room for padding.
  vertex_size += 16;
  patch_size += 16;
  if (model == spv::ExecutionModelVertex) {
    _tess_vertex_stride = vertex_size;
  } else {
    _tess_control_stride = vertex_size;
    _tess_patch_stride = patch_size;
  }
}

/**
 * Transforms the given SPIR-V module to suit Metal's resource model and
 * translates it to MSL.
 */
bool MetalShaderContext::
translate_module(MetalGraphicsStateGuardian *gsg, const ShaderModule *module,
                 int alpha_test_mode, int vertex_variant,
                 std::string &msl, std::string &entry_point) {

  if (!module->is_of_type(ShaderModuleSpirV::get_class_type())) {
    metaldisplay_cat.error()
      << "Shader modules of type " << module->get_type()
      << " are not supported by the Metal renderer.\n";
    return false;
  }
  const ShaderModuleSpirV *spv_module = (const ShaderModuleSpirV *)module;

  spv::ExecutionModel model;
  switch (module->get_stage()) {
  case Shader::Stage::VERTEX:
    model = spv::ExecutionModelVertex;
    break;
  case Shader::Stage::TESS_CONTROL:
    model = spv::ExecutionModelTessellationControl;
    break;
  case Shader::Stage::TESS_EVALUATION:
    model = spv::ExecutionModelTessellationEvaluation;
    break;
  case Shader::Stage::FRAGMENT:
    model = spv::ExecutionModelFragment;
    break;
  case Shader::Stage::COMPUTE:
    model = spv::ExecutionModelGLCompute;
    break;
  default:
    return false;
  }

  // Gather all of the loose uniforms into a block, since Metal does not have
  // a notion of loose uniforms.
  SpirVTransformer transformer(spv_module->_instructions);

  // Move textures out of structs, so that they can be bound individually.
  pmap<uint32_t, uint32_t> id_to_slot;
  {
    bool needs_hoist = false;
    for (const ResourceGroup &group : _resource_groups) {
      if (!group._chain.empty()) {
        needs_hoist = true;
      }
    }

    SpirVHoistStructResourcesPass hoist_pass(true);
    if (needs_hoist) {
      transformer.run(hoist_pass);
    }

    for (const ResourceGroup &group : _resource_groups) {
      const Shader::Parameter &param = _source->_parameters[group._param_index];
      int index = spv_module->find_parameter(param._name);
      if (index < 0) {
        continue;
      }
      uint32_t var_id = spv_module->get_parameter(index).id;
      if (group._chain.empty()) {
        id_to_slot[var_id] = group._first_slot;
      } else {
        SpirVTransformPass::AccessChain chain{SpirVId(var_id)};
        for (uint32_t member : group._chain) {
          chain.append(member);
        }
        auto it = hoist_pass._hoisted_vars.find(chain);
        if (it != hoist_pass._hoisted_vars.end()) {
          id_to_slot[(uint32_t)it->second] = group._first_slot;
        }
      }
    }
  }

  // Feed float transform indices as uints; loads are converted back to float.
  if (model == spv::ExecutionModelVertex && _retyped_index_location >= 0) {
    for (size_t i = 0; i < spv_module->get_num_inputs(); ++i) {
      const ShaderModule::Variable &var = spv_module->get_input(i);
      if (var._location == _retyped_index_location) {
        uint32_t num_components = 1;
        const ShaderType::Vector *vector = var.type->as_vector();
        if (vector != nullptr) {
          num_components = vector->get_num_components();
        }
        const ShaderType *new_type = (num_components == 1) ? (const ShaderType *)ShaderType::UINT
          : ShaderType::register_type(ShaderType::Vector(ShaderType::ST_uint, num_components));
        transformer.run(SpirVReplaceVariableTypePass(SpirVId(var.id), new_type, spv::StorageClassInput));
        break;
      }
    }
  }

  // Change bool uniforms into ints before moving them into the block.  Loads
  // of them are converted back to bools.
  for (const auto &item : _bool_members) {
    int index = spv_module->find_parameter(item.first);
    if (index >= 0) {
      transformer.run(SpirVReplaceVariableTypePass(SpirVId(spv_module->get_parameter(index).id),
                                                   item.second, spv::StorageClassUniformConstant));
    }
  }

  // Inject skinning and/or instancing.  This has to happen before the
  // uniforms are moved into the block, since instancing replaces the model
  // matrix uniforms.  The skinning matrices are stored in a uniform block,
  // which goes into buffer slot 2.
  uint32_t transform_block_id = 0;
  pmap<uint32_t, uint32_t> matrix_remap;
  if (vertex_variant != 0 && model == spv::ExecutionModelVertex) {
    SpirVInjectVertexTransformPass pass(false, false, 0, 2);
    if (vertex_variant & VV_animated) {
      pass.setup_animation(_anim_locations, _anim_point_locations,
                           _transform_index_location, _transform_weight_location);
    }
    if (vertex_variant & VV_instanced) {
      pass.setup_instancing_attrib(_instance_mat_location);
      for (size_t i = 0; i < spv_module->get_num_parameters(); ++i) {
        const ShaderModule::Variable &var = spv_module->get_parameter(i);
        auto it = _model_matrices.find(var.name);
        if (it != _model_matrices.end()) {
          pass.mark_model_matrix(SpirVId(var.id), it->second.first, it->second.second);
        }
      }
    }
    transformer.run(pass);
    transform_block_id = pass._transform_block_var_id;

    // The original uniforms have been replaced by new variables.
    for (const auto &item : pass._matrix_vars) {
      matrix_remap[(uint32_t)item.first] = (uint32_t)item.second._id;
    }
  }

  if (_block_type != nullptr) {
    pvector<uint32_t> ids = spv_module->get_parameter_ids_from_names(_block_names);
    pvector<SpirVId> member_ids;
    member_ids.reserve(ids.size());
    for (uint32_t id : ids) {
      auto it = matrix_remap.find(id);
      member_ids.push_back(SpirVId(it != matrix_remap.end() ? it->second : id));
    }
    transformer.run(SpirVMakeBlockPass(_block_type, member_ids, spv::StorageClassUniform, 0, 0));
  }

  // The transform table goes into its own block, in buffer slot 1.
  uint32_t table_block_id = 0;
  if (_table_block_type != nullptr) {
    int index = spv_module->find_parameter(_table_name);
    if (index >= 0) {
      pvector<SpirVId> table_ids;
      table_ids.push_back(SpirVId(spv_module->get_parameter(index).id));
      SpirVMakeBlockPass table_pass(_table_block_type, table_ids, spv::StorageClassUniform, 0, 1);
      transformer.run(table_pass);
      table_block_id = table_pass._block_var_id;
    }
  }

  // The alpha test reference value becomes function constant 0.
  if (model == spv::ExecutionModelFragment &&
      alpha_test_mode != RenderAttrib::M_none &&
      alpha_test_mode != RenderAttrib::M_always) {
    transformer.run(SpirVInjectAlphaTestPass((SpirVInjectAlphaTestPass::Mode)alpha_test_mode, 0, true));
  }

  // Metal's window origin is in the upper-left corner, so gl_FragCoord needs
  // to be flipped by a scale and offset given in buffer slot 4.
  uint32_t flip_block_id = 0;
  if (getenv("NOFLIP") == nullptr && model == spv::ExecutionModelFragment) {
    SpirVFlipFragCoordPass flip_pass(4, 0);
    transformer.run(flip_pass);
    flip_block_id = flip_pass._flip_block_var_id;
  }
  transformer.run(SpirVRemoveUnusedVariablesPass());

  if (_is_tessellated && (model == spv::ExecutionModelTessellationControl ||
                          model == spv::ExecutionModelTessellationEvaluation)) {
    pvector<SpirVKeepInputsPass::Location> locations;
    for (const InterfaceVariable &var : _prev_stage_outputs) {
      if (var._builtin < 0) {
        locations.push_back({var._location, var._vecsize, var._per_patch});
      }
    }
    transformer.run(SpirVKeepInputsPass(std::move(locations)));
  }

  ShaderModuleSpirV::InstructionStream stream = transformer.get_result();
  if (!stream.validate()) {
    metaldisplay_cat.error()
      << "SPIR-V " << module->get_stage() << " module of "
      << _source->get_debug_name() << " did not pass validation after transforms.\n";
    return false;
  }

  try {
    spirv_cross::CompilerMSL compiler(std::move(stream._words));

    spirv_cross::CompilerMSL::Options options;
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    MTLLanguageVersion language_version;
    int msl_version = get_msl_version(language_version);
    options.set_msl_version(msl_version / 100, (msl_version / 10) % 10);
    options.enable_point_size_builtin = true;
    options.pad_fragment_output_components = true;
    options.texture_1D_as_2D = true;
    options.texture_buffer_native = true;
    if (_is_tessellated) {
      // See MetalGraphicsStateGuardian::draw_patches() for how these are run.
      options.multi_patch_workgroup = true;
      options.tess_domain_origin_lower_left = true;
      if (model == spv::ExecutionModelVertex) {
        options.vertex_for_tessellation = true;
        options.capture_output_to_buffer = true;
      }
      else if (model == spv::ExecutionModelTessellationControl) {
        options.capture_output_to_buffer = true;
      }
      else if (model == spv::ExecutionModelTessellationEvaluation) {
        options.raw_buffer_tese_input = true;
      }
    }
    compiler.set_msl_options(options);

    // Give the entry point a predictable name.
    auto entry_points = compiler.get_entry_points_and_stages();
    nassertr(!entry_points.empty(), false);
    entry_point = (model == spv::ExecutionModelVertex) ? "panda_vertex"
                : (model == spv::ExecutionModelTessellationControl) ? "panda_tess_control"
                : (model == spv::ExecutionModelTessellationEvaluation) ? "panda_tess_eval"
                : (model == spv::ExecutionModelFragment) ? "panda_fragment"
                : "panda_compute";
    compiler.rename_entry_point(entry_points[0].name, entry_point, entry_points[0].execution_model);

    if (model == spv::ExecutionModelTessellationControl) {
      // In GLSL, the domain and spacing are only declared in the evaluation
      // stage, but SPIRV-Cross needs to know them here.
      for (uint32_t mode : _tess_eval_modes) {
        compiler.set_execution_mode((spv::ExecutionMode)mode);
      }
      _tess_output_vertices = compiler.get_execution_mode_argument(spv::ExecutionModeOutputVertices);
    }
    else if (model == spv::ExecutionModelTessellationEvaluation) {
      // And vice versa for the number of control points.
      compiler.set_execution_mode(spv::ExecutionModeOutputVertices, _tess_output_vertices);
    }

    // The uniform block always goes into buffer slot 0.
    spirv_cross::ShaderResources resources = compiler.get_shader_resources();
    for (const spirv_cross::Resource &res : resources.uniform_buffers) {
      uint32_t slot = ((uint32_t)res.id == transform_block_id) ? 2
                    : ((uint32_t)res.id == table_block_id) ? 1
                    : ((uint32_t)res.id == flip_block_id) ? 4 : 0;
      compiler.set_decoration(res.id, spv::DecorationDescriptorSet, slot);
      compiler.set_decoration(res.id, spv::DecorationBinding, 0);

      spirv_cross::MSLResourceBinding binding;
      binding.stage = model;
      binding.desc_set = slot;
      binding.binding = 0;
      binding.msl_buffer = slot;
      compiler.add_msl_resource_binding(binding);
    }

    // Assign the texture and sampler slots.  Names are not always preserved
    // in the SPIR-V, so we match the resources up by the variable id instead.
    for (const spirv_cross::Resource &res : resources.sampled_images) {
      auto it = id_to_slot.find((uint32_t)res.id);
      if (it == id_to_slot.end()) {
        metaldisplay_cat.error()
          << "Could not find shader input for texture " << compiler.get_name(res.id)
          << " (id " << (uint32_t)res.id << ") in " << _source->get_debug_name() << "\n";
        return false;
      }
      uint32_t slot = it->second;

      compiler.set_decoration(res.id, spv::DecorationDescriptorSet, 1);
      compiler.set_decoration(res.id, spv::DecorationBinding, slot);

      spirv_cross::MSLResourceBinding binding;
      binding.stage = model;
      binding.desc_set = 1;
      binding.binding = slot;
      binding.msl_texture = slot;
      binding.msl_sampler = slot;
      compiler.add_msl_resource_binding(binding);
    }

    msl = compiler.compile();

    if (_is_tessellated && (model == spv::ExecutionModelVertex ||
                            model == spv::ExecutionModelTessellationControl)) {
      record_stage_outputs(compiler, model);
    }

    // SPIRV-Cross sometimes wraps array elements in this helper even when the
    // element already has the size of the array stride, which results in a
    // zero-length array.  Make the helper handle that case.
    static const std::string padded_decl =
      "template <typename T, int stride>\n"
      "struct spvPaddedArrayElement { T data; char padding[stride - sizeof(T)]; };\n";
    size_t pos = msl.find(padded_decl);
    if (pos != std::string::npos) {
      msl.replace(pos, padded_decl.size(),
        "template <typename T, int stride, bool = (stride > (int)sizeof(T))>\n"
        "struct spvPaddedArrayElement { T data; char padding[stride - sizeof(T)]; };\n"
        "template <typename T, int stride>\n"
        "struct spvPaddedArrayElement<T, stride, false> { T data; };\n");
    }
  }
  catch (const spirv_cross::CompilerError &error) {
    metaldisplay_cat.error()
      << "Failed to translate " << module->get_stage() << " module of "
      << _source->get_debug_name() << " to MSL: " << error.what() << "\n";
    return false;
  }

  return true;
}

/**
 * Translates the given module to MSL and compiles it into a Metal library.
 * Returns nil on failure.
 */
MetalLibrary MetalShaderContext::
compile_library(MetalGraphicsStateGuardian *gsg, const ShaderModule *module,
                int alpha_test_mode, int vertex_variant, std::string &entry_point) {
  Shader::Stage stage = module->get_stage();

  std::string msl;
  if (!translate_module(gsg, module, alpha_test_mode, vertex_variant, msl, entry_point)) {
    return nil;
  }

  if (metal_dump_msl) {
    metaldisplay_cat.info()
      << "MSL for " << stage << " stage of " << _source->get_debug_name()
      << ":\n" << msl << "\n";
  }

  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  MTLLanguageVersion language_version;
  get_msl_version(language_version);
  options.languageVersion = language_version;

  NSError *error = nil;
  MetalLibrary library = [gsg->get_device() newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()]
                                                          options:options
                                                            error:&error];
  [options release];

  if (library == nil) {
    metaldisplay_cat.error()
      << "Failed to compile MSL for " << stage << " stage of "
      << _source->get_debug_name() << ":\n"
      << [[error localizedDescription] UTF8String] << "\n";
    if (!metal_dump_msl) {
      metaldisplay_cat.error(false) << msl << "\n";
    }
  }
  return library;
}

/**
 * Returns the vertex function, or the variant with skinning injected.
 */
MetalFunction MetalShaderContext::
get_vertex_function(MetalGraphicsStateGuardian *gsg, int vertex_variant) {
  if (vertex_variant == 0) {
    return _vertex_function;
  }
  nassertr(vertex_variant > 0 && vertex_variant < 4, nil);
  MetalFunction &function = _vertex_variants[vertex_variant];
  if (function != nil) {
    return function;
  }
  if (((vertex_variant & VV_animated) && !can_inject_animation()) ||
      ((vertex_variant & VV_instanced) && !can_inject_instancing())) {
    return nil;
  }

  std::string entry_point;
  MetalLibrary library = compile_library(gsg, _vertex_module, RenderAttrib::M_none, vertex_variant, entry_point);
  if (library != nil) {
    function = [library newFunctionWithName:[NSString stringWithUTF8String:entry_point.c_str()]];
    [library release];
  }
  return function;
}

/**
 * Returns the fragment function to use for the given alpha test settings.
 */
MetalFunction MetalShaderContext::
get_fragment_function(MetalGraphicsStateGuardian *gsg,
                      int alpha_test_mode, uint32_t alpha_test_ref) {
  if (alpha_test_mode == RenderAttrib::M_none ||
      alpha_test_mode == RenderAttrib::M_always ||
      _fragment_module == nullptr || _subsumes_alpha_test) {
    return _fragment_function;
  }

  auto key = std::make_pair(alpha_test_mode, alpha_test_ref);
  auto fit = _alpha_test_functions.find(key);
  if (fit != _alpha_test_functions.end()) {
    return fit->second;
  }

  MetalLibrary library;
  auto lit = _alpha_test_libraries.find(alpha_test_mode);
  if (lit != _alpha_test_libraries.end()) {
    library = lit->second;
  } else {
    std::string entry_point;
    library = compile_library(gsg, _fragment_module, alpha_test_mode, 0, entry_point);
    _alpha_test_libraries[alpha_test_mode] = library;
  }

  MetalFunction function = nil;
  if (library != nil) {
    NSError *error = nil;
    if (alpha_test_mode == RenderAttrib::M_never) {
      function = [library newFunctionWithName:@"panda_fragment"];
    } else {
      float ref;
      memcpy(&ref, &alpha_test_ref, sizeof(float));
      MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];
      [values setConstantValue:&ref type:MTLDataTypeFloat atIndex:0];
      function = [library newFunctionWithName:@"panda_fragment" constantValues:values error:&error];
      [values release];
    }
    if (function == nil) {
      metaldisplay_cat.error()
        << "Failed to create alpha test variant of " << _source->get_debug_name();
      if (error != nil) {
        metaldisplay_cat.error(false) << ": " << [[error localizedDescription] UTF8String];
      }
      metaldisplay_cat.error(false) << "\n";
    }
  }

  _alpha_test_functions[key] = function;
  return function;
}

/**
 * Determines which of the shader's inputs have changed since the last time
 * this was called, and fetches their new values.
 */
void MetalShaderContext::
update_uniforms(MetalGraphicsStateGuardian *gsg, const RenderState *target_rs,
                const TransformState *modelview_transform,
                const TransformState *camera_transform,
                const TransformState *projection_transform) {
  int altered = 0;

  if (_modelview_transform != modelview_transform) {
    _modelview_transform = modelview_transform;
    altered |= Shader::D_transform;
  }
  if (_camera_transform != camera_transform) {
    _camera_transform = camera_transform;
    altered |= Shader::D_view_transform;
  }
  if (_projection_transform != projection_transform) {
    _projection_transform = projection_transform;
    altered |= Shader::D_projection;
  }

  CPT(RenderState) state_rs = _state_rs.lock();
  if (state_rs == nullptr) {
    // Reset all of the state bits.
    altered |= Shader::D_state;
    _state_rs = target_rs;
    _color_attrib = target_rs->get_attrib_def(ColorAttrib::get_class_slot());
  }
  else if (state_rs != target_rs) {
    // The state has changed since last time.
    if (_color_attrib != target_rs->get_attrib_def(ColorAttrib::get_class_slot())) {
      altered |= Shader::D_color;
      _color_attrib = target_rs->get_attrib_def(ColorAttrib::get_class_slot());
    }
    if (state_rs->get_attrib(ColorScaleAttrib::get_class_slot()) !=
        target_rs->get_attrib(ColorScaleAttrib::get_class_slot())) {
      altered |= Shader::D_colorscale;
    }
    if (state_rs->get_attrib(MaterialAttrib::get_class_slot()) !=
        target_rs->get_attrib(MaterialAttrib::get_class_slot())) {
      altered |= Shader::D_material;
    }
    if (state_rs->get_attrib(FogAttrib::get_class_slot()) !=
        target_rs->get_attrib(FogAttrib::get_class_slot())) {
      altered |= Shader::D_fog;
    }
    if (state_rs->get_attrib(LightAttrib::get_class_slot()) !=
        target_rs->get_attrib(LightAttrib::get_class_slot())) {
      altered |= Shader::D_light;
    }
    if (state_rs->get_attrib(ClipPlaneAttrib::get_class_slot()) !=
        target_rs->get_attrib(ClipPlaneAttrib::get_class_slot())) {
      altered |= Shader::D_clip_planes;
    }
    if (state_rs->get_attrib(TexMatrixAttrib::get_class_slot()) !=
        target_rs->get_attrib(TexMatrixAttrib::get_class_slot())) {
      altered |= Shader::D_tex_matrix;
    }
    if (state_rs->get_attrib(TextureAttrib::get_class_slot()) !=
        target_rs->get_attrib(TextureAttrib::get_class_slot())) {
      altered |= Shader::D_texture;
    }
    if (state_rs->get_attrib(TexGenAttrib::get_class_slot()) !=
        target_rs->get_attrib(TexGenAttrib::get_class_slot())) {
      altered |= Shader::D_tex_gen;
    }
    if (state_rs->get_attrib(RenderModeAttrib::get_class_slot()) !=
        target_rs->get_attrib(RenderModeAttrib::get_class_slot())) {
      altered |= Shader::D_render_mode;
    }
    _state_rs = target_rs;
  }

  // We only compare the pointer, so we don't need to lock the weak pointer,
  // but we do need to make sure that it wasn't a recycled pointer from a
  // deleted attribute.
  const RenderAttrib *target_shader = gsg->get_target_shader_attrib();
  if (_shader_attrib.get_orig() != target_shader ||
      _shader_attrib.was_deleted()) {
    altered |= Shader::D_shader_inputs;
    _shader_attrib = target_shader;
  }

  // Is this the first time this shader is used this frame?
  int frame_number = ClockObject::get_global_clock()->get_frame_count();
  if (frame_number != _frame_number) {
    altered |= Shader::D_frame;
    _frame_number = frame_number;
  }

  if (_uniform_deps & Shader::D_vertex_data) {
    altered |= Shader::D_vertex_data;
  }

  // Textures may have changed from last frame, and they can also be bound as
  // shader inputs, so we re-resolve them in those cases.
  if (altered & (Shader::D_texture | Shader::D_frame | Shader::D_shader_inputs)) {
    _textures_dirty = true;
  }

  if ((altered & _uniform_deps) == 0) {
    return;
  }
  _uniforms_dirty = true;

  if (altered & _matrix_cache_deps) {
    gsg->update_shader_matrix_cache(_source, _matrix_cache.data(), altered);
  }

  ShaderInputBinding::State state;
  state.gsg = gsg;
  state.matrix_cache = _matrix_cache.data();

  unsigned char *data = _uniform_data.data();
  for (const UniformBinding &binding : _uniform_bindings) {
    if (altered & binding._binding->get_state_dep()) {
      binding._binding->fetch_data(state, data + binding._offset, false);
    }
  }
}

/**
 * Fetches the textures and samplers of the texture inputs, if they may have
 * changed since the last time.
 */
void MetalShaderContext::
resolve_textures(MetalGraphicsStateGuardian *gsg) {
  if (_textures_dirty) {
    ShaderInputBinding::State state;
    state.gsg = gsg;
    state.matrix_cache = _matrix_cache.data();

    _resolved_textures.resize(_textures.size());
    _resolved_samplers.resize(_textures.size());

    for (size_t i = 0; i < _textures.size(); ++i) {
      const TextureBinding &binding = _textures[i];

      int view = gsg->get_current_tex_view_offset();
      SamplerState sampler;
      PT(Texture) tex = binding._binding->fetch_texture(state, binding._resource_id, sampler, view);

      MetalTexture texture = nil;
      if (tex != nullptr) {
        if (Texture::is_integer(tex->get_format())) {
          sampler.set_minfilter(sampler.uses_mipmaps() ? SamplerState::FT_nearest_mipmap_nearest : SamplerState::FT_nearest);
          sampler.set_magfilter(SamplerState::FT_nearest);
        }
        texture = gsg->use_texture(tex, view);
      }
      if (texture == nil) {
        // Apply a white texture in order to make it easier to use a shader
        // that takes a texture on a model that doesn't have a texture applied.
        texture = gsg->get_white_texture(binding._texture_type);
      }

      _resolved_textures[i] = texture;
      _resolved_samplers[i] = gsg->get_sampler_state(sampler);
    }
    _textures_dirty = false;
  }
}

/**
 * Binds the uniform block and textures to the given render encoder, skipping
 * whatever is already bound.
 */
void MetalShaderContext::
bind_resources(MetalGraphicsStateGuardian *gsg, MetalRenderCommandEncoder encoder) {
  uint64_t encoder_id = gsg->get_encoder_id();
  bool rebind = (_bound_encoder != encoder_id ||
                 gsg->get_bound_shader_context() != this);

  size_t size = _uniform_data.size();
  if (size > 0 && (rebind || _uniforms_dirty)) {
    if (size <= 4096) {
      [encoder setVertexBytes:_uniform_data.data() length:size atIndex:0];
      [encoder setFragmentBytes:_uniform_data.data() length:size atIndex:0];
    } else {
      MetalBuffer buffer;
      size_t offset;
      void *ptr = gsg->alloc_transient_buffer(size, buffer, offset);
      nassertv(ptr != nullptr);
      memcpy(ptr, _uniform_data.data(), size);
      [encoder setVertexBuffer:buffer offset:offset atIndex:0];
      [encoder setFragmentBuffer:buffer offset:offset atIndex:0];
    }
    _uniforms_dirty = false;
  }
  _bound_encoder = encoder_id;
  gsg->set_bound_shader_context(this);

  if (_textures.empty()) {
    return;
  }

  resolve_textures(gsg);
  for (size_t i = 0; i < _textures.size(); ++i) {
    gsg->bind_texture(encoder, (int)i, _resolved_textures[i], _resolved_samplers[i]);
  }
}

/**
 * Binds the uniform block, textures and samplers to the given compute encoder,
 * for running the vertex or tessellation control stage of a tessellation
 * pipeline.  bind_resources() must have been called first for this draw.
 */
void MetalShaderContext::
bind_compute_resources(MetalGraphicsStateGuardian *gsg, MetalComputeCommandEncoder encoder) {
  size_t size = _uniform_data.size();
  if (size > 0) {
    if (size <= 4096) {
      [encoder setBytes:_uniform_data.data() length:size atIndex:0];
    } else {
      MetalBuffer buffer;
      size_t offset;
      void *ptr = gsg->alloc_transient_buffer(size, buffer, offset);
      nassertv(ptr != nullptr);
      memcpy(ptr, _uniform_data.data(), size);
      [encoder setBuffer:buffer offset:offset atIndex:0];
    }
  }
  for (size_t i = 0; i < _resolved_textures.size(); ++i) {
    [encoder setTexture:_resolved_textures[i] atIndex:i];
    [encoder setSamplerState:_resolved_samplers[i] atIndex:i];
  }
}

/**
 * Returns the compute pipeline that runs the vertex stage of a tessellation
 * pipeline for vertex data of the format in the given key.  The vertex
 * attributes are fetched by Metal, as described by a stage input descriptor.
 * index_type is 0 for non-indexed drawing, 1 for 16-bit, 2 for 32-bit indices.
 */
MetalComputePipelineState MetalShaderContext::
get_tess_vertex_pipeline(MetalGraphicsStateGuardian *gsg, const PipelineKey &key,
                         int index_type) {
  auto pkey = std::make_pair(key, index_type);
  auto it = _tess_vertex_pipelines.find(pkey);
  if (it != _tess_vertex_pipelines.end()) {
    return it->second;
  }

  // Convert the vertex descriptor that we would use for a regular pipeline.
  // The buffer slots are moved out of the way of the ones used by
  // SPIRV-Cross for the kernel's other buffers.
  MTLVertexDescriptor *vdesc = make_vertex_descriptor(key);
  auto remap_slot = [](NSUInteger slot) -> NSUInteger {
    if (slot == (NSUInteger)defaults_buffer_slot) {
      return tess_defaults_buffer_slot;
    }
    return get_tess_vertex_buffer_slot(max_buffer_slot - 1 - (int)slot);
  };

  MTLStageInputOutputDescriptor *sdesc = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
  for (NSUInteger i = 0; i < 31; ++i) {
    MTLVertexAttributeDescriptor *attrib = vdesc.attributes[i];
    if (attrib.format == MTLVertexFormatInvalid) {
      continue;
    }
    MTLAttributeDescriptor *sattrib = sdesc.attributes[i];
    sattrib.format = (MTLAttributeFormat)attrib.format;
    sattrib.offset = attrib.offset;
    sattrib.bufferIndex = remap_slot(attrib.bufferIndex);
  }
  for (NSUInteger slot = 0; slot <= (NSUInteger)defaults_buffer_slot; ++slot) {
    MTLVertexBufferLayoutDescriptor *layout = vdesc.layouts[slot];
    if (layout.stride == 0 || slot == (NSUInteger)instance_buffer_slot) {
      continue;
    }
    MTLBufferLayoutDescriptor *slayout = sdesc.layouts[remap_slot(slot)];
    slayout.stride = layout.stride;
    switch (layout.stepFunction) {
    case MTLVertexStepFunctionPerVertex:
      slayout.stepFunction = (index_type != 0) ? MTLStepFunctionThreadPositionInGridXIndexed
                                               : MTLStepFunctionThreadPositionInGridX;
      slayout.stepRate = 1;
      break;
    case MTLVertexStepFunctionPerInstance:
      slayout.stepFunction = MTLStepFunctionThreadPositionInGridY;
      slayout.stepRate = layout.stepRate;
      break;
    default:
      slayout.stepFunction = MTLStepFunctionConstant;
      slayout.stepRate = 0;
      break;
    }
  }
  if (index_type != 0) {
    sdesc.indexType = (index_type == 1) ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    sdesc.indexBufferIndex = tess_index_buffer_slot;
  }

  MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
  desc.computeFunction = _vertex_function;
  desc.stageInputDescriptor = sdesc;
  if (metal_debug_labels) {
    desc.label = [NSString stringWithUTF8String:_source->get_debug_name().c_str()];
  }

  NSError *error = nil;
  MetalComputePipelineState pipeline =
    [gsg->get_device() newComputePipelineStateWithDescriptor:desc
                                                     options:MTLPipelineOptionNone
                                                  reflection:nil
                                                       error:&error];
  [desc release];
  if (pipeline == nil) {
    metaldisplay_cat.error()
      << "Failed to create vertex stage pipeline for " << _source->get_debug_name()
      << ": " << [[error localizedDescription] UTF8String] << "\n";
  }
  _tess_vertex_pipelines[pkey] = pipeline;
  return pipeline;
}

/**
 * Returns the compute pipeline that runs the tessellation control stage.
 */
MetalComputePipelineState MetalShaderContext::
get_tess_control_pipeline(MetalGraphicsStateGuardian *gsg) {
  if (_tess_control_pipeline == nil && _tess_control_function != nil) {
    NSError *error = nil;
    _tess_control_pipeline = [gsg->get_device() newComputePipelineStateWithFunction:_tess_control_function
                                                                               error:&error];
    if (_tess_control_pipeline == nil) {
      metaldisplay_cat.error()
        << "Failed to create tessellation control pipeline for "
        << _source->get_debug_name() << ": "
        << [[error localizedDescription] UTF8String] << "\n";
    }
  }
  return _tess_control_pipeline;
}

/**
 * Determines the actual names of the vertex columns that are fed to inputs
 * whose name depends on the texture stages, the same way that the OpenGL
 * renderer does, and stores them in the pipeline key.
 */
void MetalShaderContext::
resolve_uv_names(const TextureAttrib *texture, PipelineKey &key) {
  memset(key._uv_names, 0, sizeof(key._uv_names));
  if (!_uses_uv_names) {
    return;
  }

  int uv_index = 0;
  for (const VertexInput &input : _vertex_inputs) {
    if (input._append_uv < 0) {
      continue;
    }
    if (uv_index >= PipelineKey::max_uv_inputs) {
      break;
    }

    const InternalName *name = input._name;
    int texslot = input._append_uv;
    if (texture != nullptr && texslot < texture->get_num_on_stages()) {
      TextureStage *stage = texture->get_on_stage(texslot);
      InternalName *texname = stage->get_texcoord_name();

      if (input._name == InternalName::get_texcoord()) {
        name = texname;
      } else if (texname != InternalName::get_texcoord()) {
        // Keep the name alive, since the key only stores a pointer.
        PT(InternalName) appended = ((InternalName *)input._name.p())->append(texname->get_basename());
        _appended_uv_names.insert(appended);
        name = appended;
      }
    }
    key._uv_names[uv_index++] = name;
  }
}

/**
 * Returns the offset in the defaults buffer of the value to use for an input
 * whose vertex column is missing.  These match the defaults used by OpenGL:
 * the buffer contains the flat color, (0, 0, 0, 1), full weight on the first
 * transform (1, 0, 0, 0), and the transform indices (0, 1, 2, 3) as uints.
 */
static uint32_t
get_default_offset(const InternalName *name, bool integer) {
  if (name == InternalName::get_transform_index()) {
    return 48;
  }
  if (name == InternalName::get_transform_weight() && !integer) {
    return 32;
  }
  if (name == InternalName::get_color() && !integer) {
    return 0;
  }
  return 16;
}

/**
 * Returns the MTLVertexFormat suitable for feeding the given column to an
 * input of the given scalar type, or MTLVertexFormatInvalid.
 */
static MTLVertexFormat
get_vertex_format(const GeomVertexColumn *column, ShaderType::ScalarType scalar_type) {
  int n = column->get_num_values();
  bool normalized = (column->get_contents() == GeomEnums::C_color);
  bool integer = (scalar_type == ShaderType::ST_int ||
                  scalar_type == ShaderType::ST_uint ||
                  scalar_type == ShaderType::ST_bool);

  switch (column->get_numeric_type()) {
  case GeomEnums::NT_float32:
    switch (n) {
    case 1: return MTLVertexFormatFloat;
    case 2: return MTLVertexFormatFloat2;
    case 3: return MTLVertexFormatFloat3;
    case 4: return MTLVertexFormatFloat4;
    }
    break;

  case GeomEnums::NT_packed_dabc:
    if (n == 1 || n == 4) {
      return MTLVertexFormatUChar4Normalized_BGRA;
    }
    break;

  case GeomEnums::NT_uint8:
    if (integer) {
      switch (n) {
      case 1: return MTLVertexFormatUChar;
      case 2: return MTLVertexFormatUChar2;
      case 3: return MTLVertexFormatUChar3;
      case 4: return MTLVertexFormatUChar4;
      }
    } else {
      switch (n) {
      case 1: return MTLVertexFormatUCharNormalized;
      case 2: return MTLVertexFormatUChar2Normalized;
      case 3: return MTLVertexFormatUChar3Normalized;
      case 4: return MTLVertexFormatUChar4Normalized;
      }
    }
    break;

  case GeomEnums::NT_int8:
    if (integer) {
      switch (n) {
      case 1: return MTLVertexFormatChar;
      case 2: return MTLVertexFormatChar2;
      case 3: return MTLVertexFormatChar3;
      case 4: return MTLVertexFormatChar4;
      }
    } else {
      switch (n) {
      case 1: return MTLVertexFormatCharNormalized;
      case 2: return MTLVertexFormatChar2Normalized;
      case 3: return MTLVertexFormatChar3Normalized;
      case 4: return MTLVertexFormatChar4Normalized;
      }
    }
    break;

  case GeomEnums::NT_uint16:
    if (integer) {
      switch (n) {
      case 1: return MTLVertexFormatUShort;
      case 2: return MTLVertexFormatUShort2;
      case 3: return MTLVertexFormatUShort3;
      case 4: return MTLVertexFormatUShort4;
      }
    } else if (normalized) {
      switch (n) {
      case 1: return MTLVertexFormatUShortNormalized;
      case 2: return MTLVertexFormatUShort2Normalized;
      case 3: return MTLVertexFormatUShort3Normalized;
      case 4: return MTLVertexFormatUShort4Normalized;
      }
    }
    break;

  case GeomEnums::NT_int16:
    if (integer) {
      switch (n) {
      case 1: return MTLVertexFormatShort;
      case 2: return MTLVertexFormatShort2;
      case 3: return MTLVertexFormatShort3;
      case 4: return MTLVertexFormatShort4;
      }
    } else if (normalized) {
      switch (n) {
      case 1: return MTLVertexFormatShortNormalized;
      case 2: return MTLVertexFormatShort2Normalized;
      case 3: return MTLVertexFormatShort3Normalized;
      case 4: return MTLVertexFormatShort4Normalized;
      }
    }
    break;

  case GeomEnums::NT_uint32:
    switch (n) {
    case 1: return MTLVertexFormatUInt;
    case 2: return MTLVertexFormatUInt2;
    case 3: return MTLVertexFormatUInt3;
    case 4: return MTLVertexFormatUInt4;
    }
    break;

  case GeomEnums::NT_int32:
    switch (n) {
    case 1: return MTLVertexFormatInt;
    case 2: return MTLVertexFormatInt2;
    case 3: return MTLVertexFormatInt3;
    case 4: return MTLVertexFormatInt4;
    }
    break;

  case GeomEnums::NT_packed_ufloat:
    if (n == 3) {
      if (@available(macOS 14.0, *)) {
        return MTLVertexFormatFloatRG11B10;
      }
    }
    break;

  default:
    break;
  }

  return MTLVertexFormatInvalid;
}

/**
 * Describes where each vertex input of the vertex stage comes from, for the
 * given pipeline key.  Returns an autoreleased MTLVertexDescriptor.
 */
id MetalShaderContext::
make_vertex_descriptor(const PipelineKey &key) {
  const GeomVertexFormat *format = key._format;

  MTLVertexDescriptor *vdesc = [MTLVertexDescriptor vertexDescriptor];
  bool uses_defaults = false;
  uint32_t used_arrays = 0;

  int uv_index = 0;
  for (const VertexInput &input : _vertex_inputs) {
    MTLVertexAttributeDescriptor *attrib = vdesc.attributes[input._location];

    // Some inputs (eg. texture coordinates) take the name of the column from
    // the texture stage they are associated with.
    const InternalName *name = input._name;
    if (input._append_uv >= 0 && uv_index < PipelineKey::max_uv_inputs) {
      if (key._uv_names[uv_index] != nullptr) {
        name = key._uv_names[uv_index];
      }
      ++uv_index;
    }

    int array_index = format->get_array_with(name);
    const GeomVertexColumn *column = (array_index >= 0) ? format->get_column(name) : nullptr;
    bool use_default = (column == nullptr) ||
                       (key._flat_color && input._name == InternalName::get_color());

    if (!use_default) {
      MTLVertexFormat vformat = get_vertex_format(column, input._scalar_type);
      if (vformat == MTLVertexFormatInvalid) {
        metaldisplay_cat.error()
          << "Vertex column " << *column << " cannot be used for shader input "
          << *input._name << "\n";
        use_default = true;
      } else {
        attrib.format = vformat;
        attrib.offset = column->get_start();
        attrib.bufferIndex = get_vertex_buffer_slot(array_index);
        used_arrays |= (1u << array_index);
      }
    }

    if (metaldisplay_cat.is_debug()) {
      metaldisplay_cat.debug()
        << "Vertex input " << *name << " at location " << input._location
        << " of " << _source->get_debug_name() << ": "
        << (use_default ? "using default value" : "from vertex array") << "\n";
    }

    if (use_default) {
      // Feed a constant value instead; see get_default_offset().
      bool integer = (input._scalar_type == ShaderType::ST_int ||
                      input._scalar_type == ShaderType::ST_uint ||
                      input._scalar_type == ShaderType::ST_bool);
      if (input._scalar_type == ShaderType::ST_int) {
        attrib.format = MTLVertexFormatInt4;
      } else if (integer) {
        attrib.format = MTLVertexFormatUInt4;
      } else {
        attrib.format = MTLVertexFormatFloat4;
      }
      attrib.offset = get_default_offset(name, integer);
      attrib.bufferIndex = defaults_buffer_slot;
      uses_defaults = true;
    }
  }

  // The injected skinning code reads the blend weights and indices.
  if (key._animated) {
    const InternalName *names[2] = {InternalName::get_transform_index(), InternalName::get_transform_weight()};
    int locations[2] = {_transform_index_location, _transform_weight_location};
    MTLVertexFormat formats[2] = {MTLVertexFormatUShort4, MTLVertexFormatFloat4};
    for (int i = 0; i < 2; ++i) {
      int array_index = format->get_array_with(names[i]);
      const GeomVertexColumn *column = (array_index >= 0) ? format->get_column(names[i]) : nullptr;
      MTLVertexAttributeDescriptor *attrib = vdesc.attributes[locations[i]];
      if (column != nullptr) {
        attrib.format = (i == 0) ? get_vertex_format(column, ShaderType::ST_uint) : formats[i];
        attrib.offset = column->get_start();
        attrib.bufferIndex = get_vertex_buffer_slot(array_index);
        used_arrays |= (1u << array_index);
      } else {
        attrib.format = (i == 0) ? MTLVertexFormatUInt4 : MTLVertexFormatFloat4;
        attrib.offset = get_default_offset(names[i], i == 0);
        attrib.bufferIndex = defaults_buffer_slot;
        uses_defaults = true;
      }
    }
  }

  // The injected instancing code reads a per-instance 3x4 matrix.
  if (key._instanced) {
    for (int i = 0; i < 3; ++i) {
      MTLVertexAttributeDescriptor *attrib = vdesc.attributes[_instance_mat_location + i];
      attrib.format = MTLVertexFormatFloat4;
      attrib.offset = i * 16;
      attrib.bufferIndex = instance_buffer_slot;
    }
    MTLVertexBufferLayoutDescriptor *layout = vdesc.layouts[instance_buffer_slot];
    layout.stride = 48;
    layout.stepFunction = MTLVertexStepFunctionPerInstance;
    layout.stepRate = 1;
  }

  for (int ai = 0; ai < (int)format->get_num_arrays(); ++ai) {
    if ((used_arrays & (1u << ai)) == 0) {
      continue;
    }
    const GeomVertexArrayFormat *array_format = format->get_array(ai);
    MTLVertexBufferLayoutDescriptor *layout = vdesc.layouts[get_vertex_buffer_slot(ai)];
    layout.stride = array_format->get_stride();
    if (array_format->get_divisor() > 0) {
      layout.stepFunction = MTLVertexStepFunctionPerInstance;
      layout.stepRate = array_format->get_divisor();
    } else {
      layout.stepFunction = MTLVertexStepFunctionPerVertex;
      layout.stepRate = 1;
    }
  }

  if (uses_defaults) {
    MTLVertexBufferLayoutDescriptor *layout = vdesc.layouts[defaults_buffer_slot];
    layout.stride = 64;
    layout.stepFunction = MTLVertexStepFunctionConstant;
    layout.stepRate = 0;
  }
  return vdesc;
}

/**
 * Returns a render pipeline state for this shader, creating it if necessary.
 */
MetalRenderPipelineState MetalShaderContext::
get_pipeline(MetalGraphicsStateGuardian *gsg, const PipelineKey &key) {
  auto it = _pipelines.find(key);
  if (it != _pipelines.end()) {
    return it->second;
  }

  nassertr(_vertex_function != nil, nil);

  MetalFunction vertex_function = _is_tessellated ? _tess_eval_function : get_vertex_function(gsg,
    (key._animated ? VV_animated : 0) | (key._instanced ? VV_instanced : 0));
  if (vertex_function == nil) {
    _pipelines[key] = nil;
    return nil;
  }

  MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
  if (metal_debug_labels) {
    desc.label = [NSString stringWithUTF8String:_source->get_debug_name().c_str()];
  }
  desc.vertexFunction = vertex_function;
  desc.fragmentFunction = get_fragment_function(gsg, key._alpha_test_mode, key._alpha_test_ref);

  if (_is_tessellated) {
    // The vertex stage runs as a compute kernel; the evaluation stage reads
    // the control points straight from the buffer that the control stage
    // wrote them to.
    desc.tessellationPartitionMode = (MTLTessellationPartitionMode)_tess_partition_mode;
    desc.tessellationOutputWindingOrder = _tess_cw ? MTLWindingClockwise : MTLWindingCounterClockwise;
    desc.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
    desc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
    desc.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
    desc.maxTessellationFactor = 64;
  } else {
    desc.vertexDescriptor = make_vertex_descriptor(key);
  }

  // Describe the render target and blending.
  // The same blending and write mask apply to all color attachments, as in
  // OpenGL.
  for (int i = 0; i < 8 && key._color_formats[i] != MTLPixelFormatInvalid; ++i) {
    MTLRenderPipelineColorAttachmentDescriptor *color = desc.colorAttachments[i];
    color.pixelFormat = (MTLPixelFormat)key._color_formats[i];
    color.writeMask = (MTLColorWriteMask)key._color_write_mask;
    gsg->apply_blend_key(color, key._blend);
  }

  desc.depthAttachmentPixelFormat = (MTLPixelFormat)key._depth_format;
  desc.stencilAttachmentPixelFormat = (MTLPixelFormat)key._stencil_format;
  desc.rasterSampleCount = key._sample_count;
  desc.alphaToCoverageEnabled = ((key._blend >> 27) & 1) != 0 && key._sample_count > 1;

  NSError *error = nil;
  MetalRenderPipelineState pipeline =
    [gsg->get_device() newRenderPipelineStateWithDescriptor:desc error:&error];
  [desc release];

  if (pipeline == nil) {
    metaldisplay_cat.error()
      << "Failed to create pipeline state for " << _source->get_debug_name()
      << ": " << [[error localizedDescription] UTF8String] << "\n";
  }

  // Cache failures too, so that we don't keep trying.
  _pipelines[key] = pipeline;
  return pipeline;
}
