/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsStateGuardian.h
 * @author Jace Mayer
 * @date 2026-10-15
 */

#ifndef METALGRAPHICSSTATEGUARDIAN_H
#define METALGRAPHICSSTATEGUARDIAN_H

#include "pandabase.h"
#include "graphicsStateGuardian.h"
#include "metalTypes.h"
#include "pmap.h"
#include "circularAllocator.h"
#include "samplerState.h"
#include "colorBlendAttrib.h"
#include "patomic.h"
#include "lightMutex.h"
#include "pdeque.h"

#include <dispatch/dispatch.h>

class MetalGraphicsPipe;
class MetalShaderContext;
class MetalTextureContext;
class MetalVertexBufferContext;
class MetalIndexBufferContext;
class StencilAttrib;
class TransformTable;
class InstanceList;

/**
 * Manages a Metal command queue, and the command buffer and encoders that are
 * used to record the rendering commands for each frame.
 */
class EXPCL_METALDISPLAY MetalGraphicsStateGuardian : public GraphicsStateGuardian {
public:
  MetalGraphicsStateGuardian(GraphicsEngine *engine, MetalGraphicsPipe *pipe,
                             MetalGraphicsStateGuardian *share_with);
  virtual ~MetalGraphicsStateGuardian();

  virtual void reset();
  virtual void close_gsg();

  virtual std::string get_driver_vendor();
  virtual std::string get_driver_renderer();
  virtual std::string get_driver_version();

  virtual void clear(DrawableRegion *clearable);
  virtual void prepare_display_region(DisplayRegionPipelineReader *dr);
  virtual CPT(TransformState) calc_projection_mat(const Lens *lens);
  virtual bool prepare_lens();

  virtual bool begin_frame(Thread *current_thread);
  virtual void end_frame(Thread *current_thread);

  virtual void begin_occlusion_query();
  virtual PT(OcclusionQueryContext) end_occlusion_query();

  virtual bool framebuffer_copy_to_texture(Texture *tex, int view, int z,
                                           const DisplayRegion *dr,
                                           const RenderBuffer &rb);
  virtual bool framebuffer_copy_to_ram(Texture *tex, int view, int z,
                                       const DisplayRegion *dr,
                                       const RenderBuffer &rb,
                                       ScreenshotRequest *request = nullptr);

  virtual TextureContext *prepare_texture(Texture *tex);
  virtual bool update_texture(TextureContext *tc, bool force);
  virtual bool update_texture(TextureContext *tc, bool force, CompletionToken token);
  virtual void release_texture(TextureContext *tc);
  virtual bool extract_texture_data(Texture *tex);

  virtual SamplerContext *prepare_sampler(const SamplerState &sampler);
  virtual void release_sampler(SamplerContext *sc);

  virtual ShaderContext *prepare_shader(Shader *shader);
  virtual void release_shader(ShaderContext *sc);

  virtual VertexBufferContext *prepare_vertex_buffer(GeomVertexArrayData *data);
  virtual void release_vertex_buffer(VertexBufferContext *vbc);

  virtual IndexBufferContext *prepare_index_buffer(GeomPrimitive *data);
  virtual void release_index_buffer(IndexBufferContext *ibc);

  virtual PT(GeomMunger) make_geom_munger(const RenderState *state,
                                          Thread *current_thread);

  virtual void set_state_and_transform(const RenderState *state,
                                       const TransformState *transform);

  virtual void end_scene();

  virtual bool begin_draw_primitives(const GeomPipelineReader *geom_reader,
                                     const GeomVertexDataPipelineReader *data_reader,
                                     const InstanceList *instances, bool force);
  virtual bool draw_triangles(const GeomPrimitivePipelineReader *reader,
                              bool force);
  virtual bool draw_tristrips(const GeomPrimitivePipelineReader *reader,
                              bool force);
  virtual bool draw_lines(const GeomPrimitivePipelineReader *reader,
                          bool force);
  virtual bool draw_linestrips(const GeomPrimitivePipelineReader *reader,
                               bool force);
  virtual bool draw_patches(const GeomPrimitivePipelineReader *reader,
                            bool force);
  virtual bool draw_points(const GeomPrimitivePipelineReader *reader,
                           bool force);
  virtual void end_draw_primitives();

  // These are used by MetalShaderContext.
  INLINE const RenderAttrib *get_target_shader_attrib() const;
  MetalTexture use_texture(Texture *tex, int view);
  MetalTexture get_white_texture(Texture::TextureType type);
  MetalSamplerState get_sampler_state(const SamplerState &sampler);
  void *alloc_transient_buffer(size_t size, MetalBuffer &buffer, size_t &offset);
  void apply_blend_key(id color_attachment, uint32_t key) const;

  // Used to avoid binding the same resources repeatedly.
  INLINE uint64_t get_encoder_id() const;
  INLINE MetalShaderContext *get_bound_shader_context() const;
  INLINE void set_bound_shader_context(MetalShaderContext *sc);
  void bind_texture(MetalRenderCommandEncoder encoder, int slot,
                    MetalTexture texture, MetalSamplerState sampler);
  void bind_vertex_buffer(MetalRenderCommandEncoder encoder, int slot,
                          MetalBuffer buffer, size_t offset);

  INLINE MetalDevice get_device() const;
  INLINE MetalCommandBuffer get_command_buffer() const;

  /**
   * Describes the textures that a render pass draws into.  Color attachment 0
   * is the main color buffer, the others are aux bitplanes.  With
   * multisampling, the samples are resolved into the resolve textures.
   */
  struct RenderTarget {
    static const int max_color_attachments = 8;
    MetalTexture _color[max_color_attachments] = {};
    MetalTexture _resolve[max_color_attachments] = {};
    // The DrawableRegion::RenderTexturePlane of each color attachment.
    int _planes[max_color_attachments] = {};
    int _num_color = 0;
    MetalTexture _depth = nil;
    MetalTexture _stencil = nil;
    MetalTexture _depth_resolve = nil;
    // The slice of array or cube map textures to render into.
    int _slice = 0;
    int _width = 0;
    int _height = 0;
    // True if the rows are stored bottom-up, as in OpenGL, which means the
    // projection is flipped.  _output_inverted is the GraphicsOutput's own
    // inverted flag, which determines the origin of its display regions.
    bool _inverted = false;
    bool _output_inverted = false;
  };
  void set_render_target(const RenderTarget &target);
  void set_render_target(MetalTexture color, MetalTexture depth_stencil,
                         int width, int height, bool inverted,
                         MetalTexture resolve = nil);
  int choose_sample_count(int requested) const;
  bool copy_scaled(MetalTexture source, MetalTexture dest);
  MetalRenderCommandEncoder get_render_encoder();
  void end_render_pass();
  void finish();
  void flush_and_wait();

private:
  PT(Shader) upgrade_legacy_glsl_shader(Shader *shader);
  bool copy_flipped(MetalTexture source, MetalTexture dest, int slice,
                    int x, int y, int w, int h);
  bool upload_texture(MetalTextureContext *tc);
  bool upload_buffer_texture(MetalTextureContext *tc, unsigned long format,
                             int num_components, bool swap_red_blue);
  bool texture_matches(MetalTextureContext *tc) const;
  bool update_vertex_buffer(MetalVertexBufferContext *vbc,
                            const GeomVertexArrayDataHandle *reader, bool force);
  bool update_index_buffer(MetalIndexBufferContext *ibc,
                           const GeomPrimitivePipelineReader *reader, bool force);
  bool do_draw_primitive(const GeomPrimitivePipelineReader *reader, bool force,
                         unsigned long primitive_type);

  void bind_transform_table(MetalRenderCommandEncoder encoder, const TransformTable *table);
  void bind_table_input(MetalRenderCommandEncoder encoder, MetalShaderContext *sc);
  struct TransformTableEntry;
  const TransformTableEntry *get_table_input(MetalShaderContext *sc);
  MetalBuffer use_vertex_array(const GeomVertexArrayDataHandle *handle, bool force);
  MetalIndexBufferContext *use_index_buffer(const GeomPrimitivePipelineReader *reader, bool force);
  void get_vertex_defaults(float defaults[16]) const;
  void bind_instances(MetalRenderCommandEncoder encoder, const InstanceList *instances);
  void check_table_caches();
  uint32_t get_blend_key(const RenderState *state) const;
  MetalDepthStencilState get_depth_stencil_state(const RenderState *state);
  void apply_dynamic_state(MetalRenderCommandEncoder encoder, const RenderState *state);

  void apply_viewport();
  void draw_clear_quad(DrawableRegion *clearable, unsigned int color_mask,
                       bool clear_depth, bool clear_stencil);
  MetalTexture get_copy_source(const RenderBuffer &rb, bool is_depth) const;

  MetalRenderPipelineState get_clear_pipeline(unsigned int color_mask);
  MetalDepthStencilState get_clear_depth_stencil_state(bool write_depth,
                                                       bool write_stencil);

private:
  MetalDevice _device = nil;
  MetalCommandQueue _queue = nil;
  MetalLibrary _builtin_library = nil;

  // Limits the number of frames the CPU may get ahead of the GPU.
  dispatch_semaphore_t _frame_semaphore = nullptr;
  int _frames_in_flight = 0;

  // Counts the frames that were started, and the last frame that the GPU
  // finished rendering.
  uint64_t _frame_number = 0;
  patomic<uint64_t> _completed_frame_number {0};

  // A ring buffer used for transient data such as large uniform blocks.
  MetalBuffer _transient_buffer = nil;
  CircularAllocator _transient_allocator;
  patomic<uint64_t> _transient_generation {0};

  // The shader used for states that don't specify one.
  MetalShaderContext *_default_sc = nullptr;
  MetalShaderContext *_current_sc = nullptr;

  // Caches the auto-shader version of the last shaderless state.
  CPT(RenderState) _last_ff_state;
  CPT(RenderState) _last_ff_auto_state;
  Shader *_current_shader = nullptr;
  bool _instanced_warning_shown = false;
  unsigned long _instance_count = 1;

  // Skinning matrices uploaded this frame, per transform table.
  struct TransformTableEntry {
    MetalBuffer _buffer = nil;
    size_t _offset = 0;
    UpdateSeq _modified;
  };
  pmap<const TransformTable *, TransformTableEntry> _transform_table_cache;
  pmap<std::pair<const TransformTable *, const void *>, TransformTableEntry> _table_input_cache;
  pmap<const InstanceList *, TransformTableEntry> _instance_cache;
  std::pair<const TransformTable *, const void *> _last_table_input_key;
  TransformTableEntry *_last_table_input_entry = nullptr;
  uint64_t _transform_table_frame = 0;

  // State bound to the current render encoder, so that we can skip
  // redundant calls.  Reset whenever a new encoder is started.
  uint64_t _encoder_id = 0;
  MetalShaderContext *_bound_sc = nullptr;
  static const int max_bound_slots = 31;
  MetalTexture _bound_textures[max_bound_slots];
  MetalSamplerState _bound_samplers[max_bound_slots];
  MetalBuffer _bound_vertex_buffers[max_bound_slots];
  size_t _bound_vertex_offsets[max_bound_slots];
  int _bound_cull_mode = -1;
  int _bound_fill_mode = -1;
  float _bound_depth_bias[3] = {0, 0, 0};
  bool _bound_depth_bias_valid = false;
  uint32_t _bound_stencil_ref = 0;
  bool _bound_stencil_ref_valid = false;
  LColor _bound_blend_color;
  bool _bound_blend_color_valid = false;
  float _bound_defaults[16];
  bool _bound_defaults_valid = false;

  // The last pipeline that was looked up, to avoid a map lookup if the same
  // pipeline is used repeatedly.
  MetalShaderContext *_last_pipeline_sc = nullptr;

  // The Geom being drawn with a tessellation shader; see draw_patches().
  const GeomPipelineReader *_tess_geom_reader = nullptr;
  bool _tess_drawing = false;
  bool _tess_instancing_warning_shown = false;
  unsigned char _last_pipeline_key[192];
  MetalRenderPipelineState _last_pipeline = nil;

  void reset_bound_state();
  void release_later(id object);

  MetalRenderPipelineState _current_pipeline = nil;
  MetalDepthStencilState _current_depth_stencil_state = nil;

  // Built-in white textures, per texture type.
  pmap<int, MetalTexture> _white_textures;

  // Depth/stencil states, keyed by the relevant attributes.
  struct DepthStencilKey {
    int _depth_func;
    bool _depth_write;
    const StencilAttrib *_stencil;
    bool operator < (const DepthStencilKey &other) const;
  };
  pmap<DepthStencilKey, MetalDepthStencilState> _depth_stencil_states;

  // These are only valid between begin_frame() and end_frame().
  id _autorelease_pool = nil;
  MetalCommandBuffer _cmd = nil;
  MetalRenderCommandEncoder _render_encoder = nil;

  // GPU timing for PStats.  Apple GPUs can only take timestamps between
  // render passes, so we report when each frame started and finished
  // executing on the GPU, using the command buffer timestamps.
  void send_gpu_timings();
  struct GPUFrameTiming {
    int _frame_number;
    double _start;
    double _end;
  };
  LightMutex _gpu_timings_lock;
  pdeque<GPUFrameTiming> _gpu_timings;
  double _host_to_pstats_time = 0.0;

  // Occlusion queries count samples into slots of this buffer, which is
  // replaced every frame in which queries are used.
  void begin_visibility_slot();
  MetalBuffer _visibility_buffer = nil;
  size_t _visibility_next_offset = 0;

  // Describes the render target that the next render pass will draw to.
  // The load actions are modified by clear() as long as the render pass has
  // not yet begun.
  id _pass_desc = nil;
  int _target_width = 0;
  int _target_height = 0;
  bool _target_inverted = false;
  bool _target_output_inverted = false;
  bool _target_has_depth = false;
  bool _target_has_stencil = false;
  int _target_num_color = 0;
  int _target_planes[RenderTarget::max_color_attachments] = {};
  bool _resolved_this_target = false;

  // The current viewport and scissor rectangle, in Metal's coordinates.
  // The depth range that is actually applied may be narrowed further by a
  // DepthOffsetAttrib.
  double _viewport[6] = {0, 0, 0, 0, 0, 1};
  double _applied_depth_range[2] = {0, 1};
  unsigned long _scissor[4] = {0, 0, 0, 0};

  // Built-in pipelines used to clear a part of the render target.
  // These are zero-initialized and compared bytewise.
  struct ClearPipelineKey {
    uint16_t _color_formats[RenderTarget::max_color_attachments];
    uint16_t _depth_format;
    uint16_t _stencil_format;
    uint8_t _sample_count;
    uint8_t _color_mask;  // which color attachments are written

    bool operator < (const ClearPipelineKey &other) const;
  };
  pmap<ClearPipelineKey, MetalRenderPipelineState> _clear_pipelines;
  MetalDepthStencilState _clear_depth_stencil_states[4] = {nil, nil, nil, nil};
  pmap<unsigned long, MetalRenderPipelineState> _copy_flip_pipelines;
  pmap<unsigned long, MetalRenderPipelineState> _scale_pipelines;

public:
  static TypeHandle get_class_type() {
    return _type_handle;
  }
  static void init_type() {
    GraphicsStateGuardian::init_type();
    register_type(_type_handle, "MetalGraphicsStateGuardian",
                  GraphicsStateGuardian::get_class_type());
  }
  virtual TypeHandle get_type() const {
    return get_class_type();
  }
  virtual TypeHandle force_init_type() {init_type(); return get_class_type();}

private:
  static TypeHandle _type_handle;
};

#include "metalGraphicsStateGuardian.I"

#endif
