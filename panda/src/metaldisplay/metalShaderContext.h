/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalShaderContext.h
 * @author Jace Mayer
 * @date 2026-10-19
 */

#ifndef METALSHADERCONTEXT_H
#define METALSHADERCONTEXT_H

#include "pandabase.h"
#include "shaderContext.h"
#include "shaderInputBinding.h"
#include "renderState.h"
#include "transformState.h"
#include "weakPointerTo.h"
#include "metalTypes.h"
#include "pmap.h"
#include "pset.h"

class MetalGraphicsStateGuardian;
class GeomVertexFormat;
class TextureAttrib;

namespace spirv_cross {
  class CompilerMSL;
}

/**
 * Holds the Metal functions that were translated from a Shader's SPIR-V
 * modules, along with the information needed to feed them their inputs, and
 * the pipeline states that have been created for them.
 */
class EXPCL_METALDISPLAY MetalShaderContext : public ShaderContext {
public:
  MetalShaderContext(Shader *shader, Shader *source = nullptr);
  virtual ~MetalShaderContext();

  ALLOC_DELETED_CHAIN(MetalShaderContext);

  bool create_functions(MetalGraphicsStateGuardian *gsg);
  void release_resources();

  void update_uniforms(MetalGraphicsStateGuardian *gsg,
                       const RenderState *target_rs,
                       const TransformState *modelview_transform,
                       const TransformState *camera_transform,
                       const TransformState *projection_transform);
  void resolve_textures(MetalGraphicsStateGuardian *gsg);
  void bind_resources(MetalGraphicsStateGuardian *gsg,
                      MetalRenderCommandEncoder encoder);

  /**
   * Everything that affects which MTLRenderPipelineState we need.
   */
  struct PipelineKey {
    const GeomVertexFormat *_format;
    // MTLPixelFormat of each color attachment, including aux bitplanes.
    uint16_t _color_formats[8];
    unsigned long _depth_format;
    unsigned long _stencil_format;
    unsigned long _sample_count;
    uint32_t _blend;
    uint32_t _color_write_mask;
    bool _flat_color;
    bool _animated;  // skinning is injected
    bool _instanced; // instancing is injected
    int _alpha_test_mode;
    uint32_t _alpha_test_ref;  // bits of a float

    // The actual vertex column names used by the inputs that take their name
    // from a texture stage (eg. p3d_MultiTexCoord0), in order.
    static const int max_uv_inputs = 8;
    const InternalName *_uv_names[max_uv_inputs];

    bool operator < (const PipelineKey &other) const;
  };
  MetalRenderPipelineState get_pipeline(MetalGraphicsStateGuardian *gsg,
                                        const PipelineKey &key);

  // Tessellation.  Metal runs the vertex and tessellation control stages as
  // compute kernels, which write their outputs to buffers, before the patches
  // are drawn with the tessellation evaluation stage as vertex function.
  bool _is_tessellated = false;
  MetalComputePipelineState get_tess_vertex_pipeline(MetalGraphicsStateGuardian *gsg,
                                                     const PipelineKey &key,
                                                     int index_type);
  MetalComputePipelineState get_tess_control_pipeline(MetalGraphicsStateGuardian *gsg);
  void bind_compute_resources(MetalGraphicsStateGuardian *gsg,
                              MetalComputeCommandEncoder encoder);
  uint32_t _tess_output_vertices = 0;
  bool _tess_quads = false;
  bool _tess_cw = false;
  unsigned long _tess_partition_mode = 0;  // MTLTessellationPartitionMode
  // Upper bounds of the sizes of the output records of each stage.
  size_t _tess_vertex_stride = 0;
  size_t _tess_control_stride = 0;
  size_t _tess_patch_stride = 0;
  // Buffer slots used by the vertex kernel for the vertex arrays, the
  // defaults and the index buffer (the others are used by SPIRV-Cross).
  static const int tess_index_buffer_slot = 8;
  static const int tess_defaults_buffer_slot = 9;
  static INLINE int get_tess_vertex_buffer_slot(int array_index);

  // Vertex buffer slots are allocated from the top of the range, so they
  // don't clash with the slots used by the shader for its own resources.
  static const int max_buffer_slot = 30;
  static const int defaults_buffer_slot = 30;
  static const int instance_buffer_slot = 3;
  static INLINE int get_vertex_buffer_slot(int array_index);

  struct VertexInput {
    CPT_InternalName _name;
    int _append_uv;
    int _location;
    ShaderType::ScalarType _scalar_type;
    uint32_t _num_components;
  };
  pvector<VertexInput> _vertex_inputs;
  bool _uses_vertex_color = false;
  bool _uses_uv_names = false;

  void resolve_uv_names(const TextureAttrib *texture, PipelineKey &key);

  bool _is_compute = false;

  // A p3d_TransformTable input is stored in a separate buffer (slot 1), so
  // that it can be shared by all the Geoms that use the same table, rather
  // than being copied into the uniform block for every draw.
  PT(ShaderInputBinding) _table_binding;
  const ShaderType::Struct *_table_block_type = nullptr;
  const InternalName *_table_name = nullptr;

  // Skinning on the GPU.  If the shader does not handle the animation itself,
  // a variant of the vertex function is made with skinning injected into it.
  bool _self_skinning = false;
  uint32_t _anim_locations = 0;
  uint32_t _anim_point_locations = 0;
  int _transform_index_location = -1;
  int _transform_weight_location = -1;
  INLINE bool can_inject_animation() const;

  // Instancing on the GPU.  The model matrix inputs are multiplied with a
  // per-instance matrix read from a vertex attribute.
  pmap<CPT_InternalName, std::pair<bool, bool> > _model_matrices;
  int _instance_mat_location = -1;

  // Location of a float transform_index input that is fed as uints, or -1.
  int _retyped_index_location = -1;
  INLINE bool can_inject_instancing() const;

  enum VertexVariant {
    VV_animated = 1,
    VV_instanced = 2,
  };

  // The shader whose modules and parameters we actually use.  This differs
  // from _shader if the shader had to be recompiled, eg. because it was
  // written in an old GLSL version that is passed directly to the driver.
  PT(Shader) _source;

  // Shaders that implement alpha testing themselves.
  bool _subsumes_alpha_test = false;

private:
  struct TextureBinding;
  const ShaderType *r_extract_resources(const Shader::Parameter &param,
                                        const pvector<uint32_t> &chain,
                                        pmap<pvector<uint32_t>, pvector<TextureBinding> > &groups,
                                        const ShaderType *type, int &resource_index,
                                        bool &supported);
  bool translate_module(MetalGraphicsStateGuardian *gsg, const ShaderModule *module,
                        int alpha_test_mode, int vertex_variant,
                        std::string &msl, std::string &entry_point);
  MetalLibrary compile_library(MetalGraphicsStateGuardian *gsg, const ShaderModule *module,
                               int alpha_test_mode, int vertex_variant,
                               std::string &entry_point);
  MetalFunction get_vertex_function(MetalGraphicsStateGuardian *gsg, int vertex_variant);
  MetalFunction get_fragment_function(MetalGraphicsStateGuardian *gsg,
                                      int alpha_test_mode, uint32_t alpha_test_ref);

private:
  id make_vertex_descriptor(const PipelineKey &key);  // MTLVertexDescriptor *
  bool read_tess_eval_modes(const ShaderModule *module);
  void record_stage_outputs(spirv_cross::CompilerMSL &compiler, int model);

  // Describes an output of one stage that is fed to the next stage, so that
  // the latter can be made to use the same memory layout.
  struct InterfaceVariable {
    uint32_t _location;
    uint32_t _component;
    uint32_t _vecsize;
    int _builtin;  // spv::BuiltIn, or -1
    bool _per_patch;
  };
  pvector<InterfaceVariable> _prev_stage_outputs;

  // Execution modes of the tessellation evaluation stage that the control
  // stage needs to know about, and vice versa.
  pvector<uint32_t> _tess_eval_modes;

  MetalFunction _tess_control_function = nil;
  MetalFunction _tess_eval_function = nil;
  MetalComputePipelineState _tess_control_pipeline = nil;
  pmap<std::pair<PipelineKey, int>, MetalComputePipelineState> _tess_vertex_pipelines;

private:
  MetalFunction _vertex_function = nil;
  MetalFunction _fragment_function = nil;
  MetalFunction _compute_function = nil;
  // Indexed by VertexVariant bits; index 0 is unused (see _vertex_function).
  MetalFunction _vertex_variants[4] = {nil, nil, nil, nil};
  CPT(ShaderModule) _vertex_module;

  // The fragment stage is translated again for every alpha test mode, and the
  // reference value is passed in as a function constant.
  CPT(ShaderModule) _fragment_module;
  pmap<int, MetalLibrary> _alpha_test_libraries;
  pmap<std::pair<int, uint32_t>, MetalFunction> _alpha_test_functions;

  const ShaderType::Struct *_block_type = nullptr;
  pvector<const InternalName *> _block_names;

  // All non-opaque inputs are gathered into a single uniform block, which is
  // bound to buffer slot 0 in every stage.
  struct UniformBinding {
    PT(ShaderInputBinding) _binding;
    size_t _offset;
  };
  pvector<UniformBinding> _uniform_bindings;
  pvector<unsigned char> _uniform_data;
  int _uniform_deps = 0;

  // Used to avoid redundant binding of resources.
  bool _uniforms_dirty = true;
  bool _textures_dirty = true;
  uint64_t _bound_encoder = 0;
  pvector<MetalTexture> _resolved_textures;
  pvector<MetalSamplerState> _resolved_samplers;

  // Bool uniforms, and the integer type they are converted to.
  pvector<std::pair<const InternalName *, const ShaderType *> > _bool_members;

  // Each texture input is assigned a texture and sampler slot, which are
  // the same for all stages.
  struct TextureBinding {
    PT(ShaderInputBinding) _binding;
    ShaderInputBinding::ResourceId _resource_id;
    Texture::TextureType _texture_type;
  };
  pvector<TextureBinding> _textures;

  // A group of consecutive texture slots corresponding to one resource in the
  // shader, which may be an array, or be part of a struct (identified by the
  // chain of member indices within the parameter).
  struct ResourceGroup {
    size_t _param_index;
    pvector<uint32_t> _chain;
    uint32_t _first_slot;
  };
  pvector<ResourceGroup> _resource_groups;

  pvector<LMatrix4> _matrix_cache;
  int _matrix_cache_deps = 0;

  // Used to determine which inputs have changed since the last draw.
  WCPT(RenderState) _state_rs;
  CPT(TransformState) _modelview_transform;
  CPT(TransformState) _camera_transform;
  CPT(TransformState) _projection_transform;
  const RenderAttrib *_color_attrib = nullptr;
  WCPT(RenderAttrib) _shader_attrib;
  int _frame_number = -1;

  pmap<PipelineKey, MetalRenderPipelineState> _pipelines;
  pset<PT(InternalName)> _appended_uv_names;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    ShaderContext::init_type();
    register_type(_type_handle, "MetalShaderContext",
                  ShaderContext::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#include "metalShaderContext.I"

#endif
