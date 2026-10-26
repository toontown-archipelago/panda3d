/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsStateGuardian.mm
 * @author Jace Mayer
 * @date 2026-10-15
 */

#include "metalGraphicsStateGuardian.h"
#include "pStatTimer.h"
#include "metalGraphicsPipe.h"
#include "config_metaldisplay.h"
#include "displayRegion.h"
#include "drawableRegion.h"
#include "lens.h"
#include "sceneSetup.h"
#include "renderBuffer.h"
#include "texture.h"
#include "metalShaderContext.h"
#include "metalTextureContext.h"
#include "metalSamplerContext.h"
#include "metalVertexBufferContext.h"
#include "metalOcclusionQueryContext.h"
#include "geomPatches.h"
#include "lightMutexHolder.h"
#include "pStatClient.h"
#include "pStatFrameData.h"
#include "config_pstatclient.h"

#import <QuartzCore/QuartzCore.h>
#include "metalIndexBufferContext.h"
#include "metalGeomMunger.h"
#include "config_display.h"
#include "shader.h"
#include "shaderModuleGlsl.h"
#include "shaderAttrib.h"
#include "colorAttrib.h"
#include "colorWriteAttrib.h"
#include "colorBlendAttrib.h"
#include "transparencyAttrib.h"
#include "depthTestAttrib.h"
#include "depthWriteAttrib.h"
#include "depthOffsetAttrib.h"
#include "depthBiasAttrib.h"
#include "stencilAttrib.h"
#include "alphaTestAttrib.h"
#include "textureAttrib.h"
#include "transformTable.h"
#include "instanceList.h"
#include "vertexTransform.h"
#include "cullFaceAttrib.h"
#include "renderModeAttrib.h"
#include "geomVertexArrayData.h"
#include "geomPrimitive.h"
#include "completionToken.h"
#include "graphicsEngine.h"

#import <Foundation/Foundation.h>

TypeHandle MetalGraphicsStateGuardian::_type_handle;

// Time spent waiting for the GPU to finish an earlier frame.
static PStatCollector _wait_gpu_pcollector("Wait:GPU");

// Used when a state does not specify a shader.
static const std::string default_vshader =
  "#version 330\n"
  "in vec4 p3d_Vertex;\n"
  "in vec4 p3d_Color;\n"
  "in vec2 p3d_MultiTexCoord0;\n"
  "out vec2 texcoord;\n"
  "out vec4 color;\n"
  "uniform mat4 p3d_ModelViewProjectionMatrix;\n"
  "uniform vec4 p3d_ColorScale;\n"
  "void main(void) {\n"
  "  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;\n"
  "  texcoord = p3d_MultiTexCoord0;\n"
  "  color = p3d_Color * p3d_ColorScale;\n"
  "}\n";

static const std::string default_fshader =
  "#version 330\n"
  "in vec2 texcoord;\n"
  "in vec4 color;\n"
  "out vec4 p3d_FragColor;\n"
  "uniform sampler2D p3d_Texture0;\n"
  "void main(void) {\n"
  "  p3d_FragColor = texture(p3d_Texture0, texcoord);\n"
  "  p3d_FragColor *= color;\n"
  "}\n";

// Shaders used internally by the GSG, eg. for clearing part of the screen.
static const char *const builtin_msl = R"(
#include <metal_stdlib>
using namespace metal;

struct ClearVertexOut {
  float4 position [[position]];
};

vertex ClearVertexOut clear_vertex(uint vid [[vertex_id]],
                                   constant float &depth [[buffer(0)]]) {
  // Generates a triangle that covers the entire viewport.
  float2 p = float2((vid << 1) & 2, vid & 2);
  ClearVertexOut out;
  out.position = float4(p * 2.0 - 1.0, depth, 1.0);
  return out;
}

// The number of color attachments is given as function constant 0.
constant int clear_num_colors [[function_constant(0)]];
constant bool clear_has_c1 = clear_num_colors > 1;
constant bool clear_has_c2 = clear_num_colors > 2;
constant bool clear_has_c3 = clear_num_colors > 3;
constant bool clear_has_c4 = clear_num_colors > 4;
constant bool clear_has_c5 = clear_num_colors > 5;
constant bool clear_has_c6 = clear_num_colors > 6;
constant bool clear_has_c7 = clear_num_colors > 7;

struct ClearColors {
  float4 c0 [[color(0)]];
  float4 c1 [[color(1), function_constant(clear_has_c1)]];
  float4 c2 [[color(2), function_constant(clear_has_c2)]];
  float4 c3 [[color(3), function_constant(clear_has_c3)]];
  float4 c4 [[color(4), function_constant(clear_has_c4)]];
  float4 c5 [[color(5), function_constant(clear_has_c5)]];
  float4 c6 [[color(6), function_constant(clear_has_c6)]];
  float4 c7 [[color(7), function_constant(clear_has_c7)]];
};

fragment ClearColors clear_fragment(constant float4 *colors [[buffer(0)]]) {
  ClearColors out;
  out.c0 = colors[0];
  if (clear_has_c1) out.c1 = colors[1];
  if (clear_has_c2) out.c2 = colors[2];
  if (clear_has_c3) out.c3 = colors[3];
  if (clear_has_c4) out.c4 = colors[4];
  if (clear_has_c5) out.c5 = colors[5];
  if (clear_has_c6) out.c6 = colors[6];
  if (clear_has_c7) out.c7 = colors[7];
  return out;
}

// Used to copy a region of a texture while flipping it vertically.
// params = (source x, source y, height, unused)
fragment float4 copy_flip_fragment(ClearVertexOut in [[stage_in]],
                                   texture2d<float> src [[texture(0)]],
                                   constant int4 &params [[buffer(0)]]) {
  uint2 c = uint2(in.position.xy);
  return src.read(uint2(c.x + params.x, params.y + params.z - 1 - c.y));
}

// Used to stretch a texture over the whole render target with bilinear
// filtering.  inv_size is the reciprocal of the render target size.
fragment float4 scale_fragment(ClearVertexOut in [[stage_in]],
                               texture2d<float> src [[texture(0)]],
                               constant float2 &inv_size [[buffer(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  return src.sample(s, in.position.xy * inv_size);
}

struct DepthOut {
  float depth [[depth(any)]];
};

fragment DepthOut copy_flip_depth_fragment(ClearVertexOut in [[stage_in]],
                                           depth2d<float> src [[texture(0)]],
                                           constant int4 &params [[buffer(0)]]) {
  uint2 c = uint2(in.position.xy);
  DepthOut out;
  out.depth = src.read(uint2(c.x + params.x, params.y + params.z - 1 - c.y));
  return out;
}
)";

/**
 * Orders clear pipeline keys so that they can be used in a map.
 */
bool MetalGraphicsStateGuardian::ClearPipelineKey::
operator < (const ClearPipelineKey &other) const {
  return memcmp(this, &other, sizeof(ClearPipelineKey)) < 0;
}

/**
 *
 */
MetalGraphicsStateGuardian::
MetalGraphicsStateGuardian(GraphicsEngine *engine, MetalGraphicsPipe *pipe,
                           MetalGraphicsStateGuardian *share_with) :
  GraphicsStateGuardian(CS_default, engine, pipe)
{
  // All GSGs on the same device can use each other's resources.
  if (share_with != nullptr) {
    _prepared_objects = share_with->get_prepared_objects();
  }

  _device = [pipe->get_device() retain];
  _queue = [_device newCommandQueue];

  _frames_in_flight = std::max((int)metal_frames_in_flight, 1);
  _frame_semaphore = dispatch_semaphore_create(_frames_in_flight);
}

/**
 *
 */
MetalGraphicsStateGuardian::
~MetalGraphicsStateGuardian() {
  close_gsg();

  if (_frame_semaphore != nullptr) {
    dispatch_release(_frame_semaphore);
    _frame_semaphore = nullptr;
  }
  if (_queue != nil) {
    [_queue release];
    _queue = nil;
  }
  if (_device != nil) {
    [_device release];
    _device = nil;
  }
}

/**
 * Resets all internal state as if the gsg were newly created.
 */
void MetalGraphicsStateGuardian::
reset() {
  GraphicsStateGuardian::reset();

  if (_device == nil || _queue == nil) {
    _is_valid = false;
    return;
  }

  if (_builtin_library == nil) {
    NSError *error = nil;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    _builtin_library = [_device newLibraryWithSource:[NSString stringWithUTF8String:builtin_msl]
                                             options:options
                                               error:&error];
    [options release];

    if (_builtin_library == nil) {
      metaldisplay_cat.error()
        << "Failed to compile built-in Metal shaders: "
        << [[error localizedDescription] UTF8String] << "\n";
      _is_valid = false;
      return;
    }
  }

  _is_hardware = true;

  // Metal renders upside down relative to OpenGL's conventions.  Offscreen
  // buffers compensate for this themselves (see MetalGraphicsBuffer), rather
  // than through the inverted flag, since applications written for OpenGL
  // may reset that flag.
  _copy_texture_inverted = false;

  _max_texture_dimension = 16384;
  _max_3d_texture_dimension = 2048;
  _max_2d_texture_array_layers = 2048;
  _max_cube_map_dimension = 16384;

  _max_texture_stages = 16;
  _max_color_targets = 8;

  _supports_3d_texture = true;
  _supports_2d_texture_array = true;
  _supports_cube_map = true;
  _supports_cube_map_array = true;
  _supports_texture_srgb = true;
  _supports_depth_texture = true;
  _supports_shadow_filter = true;
  _supports_sampler_objects = true;
  _supports_generate_mipmap = true;
  _supports_tex_non_pow2 = true;
  _supports_depth_stencil = true;
  _supports_stencil = true;
  _supports_stencil_wrap = true;
  _supports_two_sided_stencil = true;

  // Metal takes the value of flat-shaded attributes from the first vertex.
  // Triangle fans are not supported, so these will be decomposed by the
  // munger.
  _supported_geom_rendering =
    Geom::GR_indexed_point |
    Geom::GR_point |
    Geom::GR_indexed_other |
    Geom::GR_triangle_strip |
    Geom::GR_line_strip |
    Geom::GR_flat_first_vertex |
    Geom::GR_strip_cut_index |
    Geom::GR_shader_vertex_blend |
    Geom::GR_instancing;

  // Instancing is done by injecting the instance matrix into the shader.
  _supports_geometry_instancing = true;
  _supports_buffer_texture = true;
  _supports_occlusion_query = true;
#ifdef DO_PSTATS
  _supports_timer_query = true;
#endif
  // Apple GPUs support texture buffers of up to 2^28 texels.
  _max_buffer_texture_size = 1 << 28;

  // Vertex animation is always done on the GPU, by injecting the skinning
  // code into the vertex shader, which reads up to 4 indices into a table of
  // up to 256 matrices.
  _max_vertex_transforms = 4;
  _max_vertex_transform_indices = 256;
  _supports_fixed_function_vertex_blending = true;

  // Geometry and double-precision shaders are not available.
  _supported_shader_caps =
    Shader::C_basic_shader |
    Shader::C_vertex_texture |
    Shader::C_point_coord |
    Shader::C_standard_derivatives |
    Shader::C_shadow_samplers |
    Shader::C_non_square_matrices |
    Shader::C_texture_lod |
    Shader::C_unified_model |
    Shader::C_noperspective_interpolation |
    Shader::C_texture_array |
    Shader::C_texture_integer |
    Shader::C_texture_query_size |
    Shader::C_sampler_cube_shadow |
    Shader::C_draw_buffers |
    Shader::C_clip_distance |
    Shader::C_instance_id |
    Shader::C_bit_encoding |
    Shader::C_texture_query_lod |
    Shader::C_texture_gather_red |
    Shader::C_texture_gather_any |
    Shader::C_extended_arithmetic |
    Shader::C_cube_map_array |
    Shader::C_sample_variables |
    Shader::C_multisample_interpolation |
    Shader::C_dynamic_indexing |
    Shader::C_texture_query_levels |
    Shader::C_texture_buffer |
    Shader::C_tessellation_shader;

  if (_transient_buffer == nil) {
    // This is the initial size; it grows when it fills up.
    size_t size = 4 * 1024 * 1024;
    _transient_buffer = [_device newBufferWithLength:size options:MTLResourceStorageModeShared];
    _transient_allocator = CircularAllocator(size, 256);
  }

  _is_valid = true;

  if (_default_sc == nullptr) {
    static PT(Shader) default_shader;
    if (default_shader.is_null()) {
      default_shader = Shader::make(Shader::SL_GLSL, default_vshader, default_fshader);
    }
    if (default_shader != nullptr) {
      ShaderContext *sc = default_shader->prepare_now(get_prepared_objects(), this);
      if (sc != nullptr) {
        _default_sc = DCAST(MetalShaderContext, sc);
      }
    }
    if (_default_sc == nullptr) {
      metaldisplay_cat.error()
        << "Failed to compile the default shader.\n";
    }
  }
}

/**
 * This is called by the associated GraphicsWindow when close_window() is
 * called.  It should null out the _win pointer and possibly free any open
 * resources associated with the GSG.
 */
void MetalGraphicsStateGuardian::
close_gsg() {
  if (_queue != nil) {
    finish();
  }

  for (auto &item : _clear_pipelines) {
    [item.second release];
  }
  _clear_pipelines.clear();

  for (auto &item : _copy_flip_pipelines) {
    [item.second release];
  }
  _copy_flip_pipelines.clear();

  for (auto &item : _scale_pipelines) {
    [item.second release];
  }
  _scale_pipelines.clear();

  for (MetalDepthStencilState &state : _clear_depth_stencil_states) {
    if (state != nil) {
      [state release];
      state = nil;
    }
  }

  if (_pass_desc != nil) {
    [_pass_desc release];
    _pass_desc = nil;
  }

  if (_builtin_library != nil) {
    [_builtin_library release];
    _builtin_library = nil;
  }

  for (auto &item : _white_textures) {
    [item.second release];
  }
  _white_textures.clear();

  for (auto &item : _depth_stencil_states) {
    [item.second release];
  }
  _depth_stencil_states.clear();

  if (_transient_buffer != nil) {
    [_transient_buffer release];
    _transient_buffer = nil;
  }

  _default_sc = nullptr;
  _current_sc = nullptr;
  _current_shader = nullptr;

  GraphicsStateGuardian::close_gsg();
}

/**
 * Returns the vendor of the video card driver
 */
std::string MetalGraphicsStateGuardian::
get_driver_vendor() {
  return "Apple";
}

/**
 * Returns GL_Renderer
 */
std::string MetalGraphicsStateGuardian::
get_driver_renderer() {
  if (_device == nil) {
    return std::string();
  }
  return [[_device name] UTF8String];
}

/**
 * Returns driver version.
 */
std::string MetalGraphicsStateGuardian::
get_driver_version() {
  return [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String];
}

/**
 * Clears the framebuffer within the current DisplayRegion, according to the
 * flags indicated by the given DrawableRegion object.
 */
void MetalGraphicsStateGuardian::
clear(DrawableRegion *clearable) {
  nassertv(_pass_desc != nil);

  // Which of the color attachments (including aux bitplanes) to clear.
  unsigned int color_mask = 0;
  for (int i = 0; i < _target_num_color; ++i) {
    if (clearable->get_clear_active(_target_planes[i])) {
      color_mask |= (1u << i);
    }
  }
  bool clear_depth = clearable->get_clear_depth_active() && _target_has_depth;
  bool clear_stencil = clearable->get_clear_stencil_active() && _target_has_stencil;
  if (color_mask == 0 && !clear_depth && !clear_stencil) {
    return;
  }

  bool covers_target =
    _scissor[0] == 0 && _scissor[1] == 0 &&
    _scissor[2] >= (unsigned long)_target_width &&
    _scissor[3] >= (unsigned long)_target_height;

  if (_render_encoder != nil || !covers_target) {
    // The render pass has already begun, or we only need to clear a part of
    // the target, so we have to draw a quad.
    draw_clear_quad(clearable, color_mask, clear_depth, clear_stencil);
    return;
  }

  // We haven't started rendering yet, so we can do this very cheaply by
  // setting the load actions of the render pass.
  MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)_pass_desc;
  for (int i = 0; i < _target_num_color; ++i) {
    if (color_mask & (1u << i)) {
      LColor color = clearable->get_clear_value(_target_planes[i]);
      desc.colorAttachments[i].loadAction = MTLLoadActionClear;
      desc.colorAttachments[i].clearColor = MTLClearColorMake(color[0], color[1], color[2], color[3]);
    }
  }
  if (clear_depth) {
    desc.depthAttachment.loadAction = MTLLoadActionClear;
    desc.depthAttachment.clearDepth = clearable->get_clear_depth();
  }
  if (clear_stencil) {
    desc.stencilAttachment.loadAction = MTLLoadActionClear;
    desc.stencilAttachment.clearStencil = clearable->get_clear_stencil();
  }
}

/**
 * Prepare a display region for rendering (set up scissor region and viewport)
 */
void MetalGraphicsStateGuardian::
prepare_display_region(DisplayRegionPipelineReader *dr) {
  nassertv(dr != nullptr);
  GraphicsStateGuardian::prepare_display_region(dr);

  int x, y, w, h;
  dr->get_region_pixels(0, x, y, w, h);

  // Panda's pixel coordinates have the origin in the lower-left corner,
  // whereas Metal puts it in the upper left.  This is already accounted for
  // if the target's rows are flipped, or if the output is inverted.
  if (_target_inverted == _target_output_inverted) {
    y = _target_height - (y + h);
  }

  PN_stdfloat min_depth, max_depth;
  dr->get_depth_range(min_depth, max_depth);

  _viewport[0] = x;
  _viewport[1] = y;
  _viewport[2] = w;
  _viewport[3] = h;
  _viewport[4] = min_depth;
  _viewport[5] = max_depth;

  // Metal requires the scissor rectangle to lie within the render target.
  int x0 = std::max(x, 0);
  int y0 = std::max(y, 0);
  int x1 = std::min(x + w, _target_width);
  int y1 = std::min(y + h, _target_height);
  _scissor[0] = x0;
  _scissor[1] = y0;
  _scissor[2] = std::max(x1 - x0, 0);
  _scissor[3] = std::max(y1 - y0, 0);

  if (_render_encoder != nil) {
    apply_viewport();
  }
}

/**
 * Given a lens, calculates the appropriate projection matrix for use with
 * this gsg.  Note that the projection matrix depends a lot upon the
 * coordinate system of the rendering API.
 *
 * The return value is a TransformState if the lens is acceptable, NULL if it
 * is not.
 */
CPT(TransformState) MetalGraphicsStateGuardian::
calc_projection_mat(const Lens *lens) {
  if (lens == nullptr) {
    return nullptr;
  }

  if (!lens->is_linear()) {
    return nullptr;
  }

  // Metal uses a Z range of 0 to 1, whereas the Panda convention is for the
  // projection matrix to produce a Z range of -1 to 1.  We have to rescale
  // to compensate.
  static const LMatrix4 rescale_mat
    (1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, 0.5, 0,
     0, 0, 0.5, 1);

  LMatrix4 result = lens->get_projection_mat(_current_stereo_channel) * rescale_mat;

  // Flip the projection if the target stores its rows bottom-up, and also if
  // the scene is supposed to be inverted, unless the target already took
  // care of that.
  if (_target_inverted != (_scene_setup->get_inverted() != _target_output_inverted)) {
    result *= LMatrix4::scale_mat(1.0f, -1.0f, 1.0f);
  }

  return TransformState::make_mat(result);
}

/**
 * Makes the current lens (whichever lens was most recently specified with
 * set_scene()) active, so that it will transform future rendered geometry.
 * Normally this is only called from the draw process, and usually it is
 * called by set_scene().
 *
 * The return value is true if the lens is acceptable, false if it is not.
 */
bool MetalGraphicsStateGuardian::
prepare_lens() {
  return true;
}

/**
 * Called before each frame is rendered, to allow the GSG a chance to do any
 * internal cleanup before beginning the frame.
 *
 * The return value is true if successful (in which case the frame will be
 * drawn and end_frame() will be called later), or false if unsuccessful (in
 * which case nothing will be drawn and end_frame() will not be called).
 */
bool MetalGraphicsStateGuardian::
begin_frame(Thread *current_thread) {
  nassertr(_cmd == nil, false);

  // Objects that Metal hands us are autoreleased, and the draw thread does
  // not otherwise have an autorelease pool, so we create one per frame.
  _autorelease_pool = [[NSAutoreleasePool alloc] init];

#ifdef DO_PSTATS
  _timer_queries_active = pstats_gpu_timing && PStatClient::is_connected();
  if (_timer_queries_active) {
    // Relate the GPU timestamps, which use the host clock, to the PStats clock.
    _host_to_pstats_time = PStatClient::get_global_pstats()->get_real_time() - CACurrentMediaTime();
    send_gpu_timings();
  }
#endif

  // Don't get too far ahead of the GPU.
  {
    PStatTimer timer(_wait_gpu_pcollector, current_thread);
    dispatch_semaphore_wait(_frame_semaphore, DISPATCH_TIME_FOREVER);
  }

  _cmd = [[_queue commandBuffer] retain];
  if (metal_debug_labels) {
    _cmd.label = @"Panda3D frame";
  }
  ++_frame_number;

  if (!GraphicsStateGuardian::begin_frame(current_thread)) {
    // Nothing was recorded; just give back the frame slot.
    [_cmd release];
    _cmd = nil;
    dispatch_semaphore_signal(_frame_semaphore);
    [_autorelease_pool drain];
    _autorelease_pool = nil;
    return false;
  }
  return true;
}

/**
 * Sends the GPU timings of finished frames to PStats, combining the command
 * buffers of each frame.  A frame is only sent once a later one has finished.
 */
void MetalGraphicsStateGuardian::
send_gpu_timings() {
#ifdef DO_PSTATS
  pdeque<GPUFrameTiming> timings;
  {
    LightMutexHolder holder(_gpu_timings_lock);
    if (_gpu_timings.empty()) {
      return;
    }
    int last_frame = _gpu_timings.back()._frame_number;
    while (!_gpu_timings.empty() && _gpu_timings.front()._frame_number < last_frame) {
      timings.push_back(_gpu_timings.front());
      _gpu_timings.pop_front();
    }
  }

  PStatThread gpu_thread = get_pstats_thread();
  while (!timings.empty()) {
    GPUFrameTiming frame = timings.front();
    timings.pop_front();
    while (!timings.empty() && timings.front()._frame_number == frame._frame_number) {
      frame._start = std::min(frame._start, timings.front()._start);
      frame._end = std::max(frame._end, timings.front()._end);
      timings.pop_front();
    }
    PStatFrameData frame_data;
    frame_data.add_start(0, frame._start);
    frame_data.add_stop(0, frame._end);
    gpu_thread.add_frame(frame._frame_number, std::move(frame_data));
  }
#endif
}

/**
 * Begins a new occlusion query.  After this call, you may call
 * begin_draw_primitives() and draw_triangles()/draw_whatever() repeatedly.
 * Eventually, you should call end_occlusion_query() before the end of the
 * frame; that will return a new OcclusionQueryContext object that will tell
 * you how many pixels represented by the bracketed geometry passed the depth
 * test.
 */
void MetalGraphicsStateGuardian::
begin_occlusion_query() {
  nassertv(_current_occlusion_query == nullptr);
  nassertv(_cmd != nil && _pass_desc != nil);

  MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)_pass_desc;
  if (_visibility_buffer == nil) {
    // Room for many thousands of queries per frame.
    static const size_t size = 256 * 1024;
    _visibility_buffer = [_device newBufferWithLength:size options:MTLResourceStorageModeShared];
    memset(_visibility_buffer.contents, 0, size);
    _visibility_next_offset = 0;
  }
  if (desc.visibilityResultBuffer != _visibility_buffer) {
    // The buffer can only be set when a render pass begins.
    end_render_pass();
    desc.visibilityResultBuffer = _visibility_buffer;
  }

  _current_occlusion_query = new MetalOcclusionQueryContext(this);
  if (_render_encoder != nil) {
    begin_visibility_slot();
  }
}

/**
 * Ends a previous call to begin_occlusion_query().  This call returns the
 * OcclusionQueryContext object that will (eventually) report the number of
 * pixels that passed the depth test between the call to
 * begin_occlusion_query() and end_occlusion_query().
 */
PT(OcclusionQueryContext) MetalGraphicsStateGuardian::
end_occlusion_query() {
  nassertr(_current_occlusion_query != nullptr, nullptr);
  if (_render_encoder != nil) {
    [_render_encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
  }
  PT(OcclusionQueryContext) result = _current_occlusion_query;
  _current_occlusion_query = nullptr;
  return result;
}

/**
 * Starts counting the samples of the current render pass into a new slot of
 * the visibility buffer, for the current occlusion query.
 */
void MetalGraphicsStateGuardian::
begin_visibility_slot() {
  nassertv(_render_encoder != nil && _visibility_buffer != nil);
  MetalOcclusionQueryContext *query;
  DCAST_INTO_V(query, _current_occlusion_query);
  if (_visibility_next_offset + 8 > _visibility_buffer.length) {
    metaldisplay_cat.warning()
      << "Too many occlusion queries in one frame.\n";
    query->mark_incomplete();
    return;
  }
  size_t offset = _visibility_next_offset;
  _visibility_next_offset += 8;
  [_render_encoder setVisibilityResultMode:MTLVisibilityResultModeCounting offset:offset];
  query->add_slot(_visibility_buffer, offset, _cmd);
}

/**
 * Called after each frame is rendered, to allow the GSG a chance to do any
 * internal cleanup after rendering the frame, and before the window flips.
 */
void MetalGraphicsStateGuardian::
end_frame(Thread *current_thread) {
  nassertv(_cmd != nil);

  end_render_pass();

  GraphicsStateGuardian::end_frame(current_thread);

  // The occlusion queries keep the visibility buffer alive for themselves.
  if (_visibility_buffer != nil) {
    [_visibility_buffer release];
    _visibility_buffer = nil;
    _visibility_next_offset = 0;
  }

  // When the GPU is done with this frame, we can reuse the space it used in
  // the transient buffer, and buffers it used may be written to again.
  dispatch_semaphore_t semaphore = _frame_semaphore;
  uint64_t frame_number = _frame_number;
  size_t transient_head = _transient_allocator.get_head();
  CircularAllocator *allocator = &_transient_allocator;
  patomic<uint64_t> *completed = &_completed_frame_number;
  patomic<uint64_t> *generation = &_transient_generation;
  uint64_t this_generation = _transient_generation.load();
#ifdef DO_PSTATS
  if (_timer_queries_active) {
    int pstats_frame = ClockObject::get_global_clock()->get_frame_count(current_thread);
    double offset = _host_to_pstats_time;
    LightMutex *lock = &_gpu_timings_lock;
    pdeque<GPUFrameTiming> *timings = &_gpu_timings;
    [_cmd addCompletedHandler:^(id<MTLCommandBuffer> cmd) {
      if (cmd.GPUEndTime > 0.0) {
        LightMutexHolder holder(*lock);
        timings->push_back({pstats_frame, cmd.GPUStartTime + offset, cmd.GPUEndTime + offset});
      }
    }];
  }
#endif

  [_cmd addCompletedHandler:^(id<MTLCommandBuffer> cmd) {
    if (generation->load() == this_generation) {
      allocator->set_tail(transient_head);
    }
    completed->store(frame_number);
    dispatch_semaphore_signal(semaphore);
  }];
  [_cmd commit];
  [_cmd release];
  _cmd = nil;

  if (_pass_desc != nil) {
    // Don't hold on to the render target textures beyond this frame.
    [_pass_desc release];
    _pass_desc = nil;
  }

  [_autorelease_pool drain];
  _autorelease_pool = nil;
}

/**
 * Copy the pixels within the indicated display region from the framebuffer
 * into texture memory.
 *
 * If z > -1, it is the cube map index into which to copy.
 */
bool MetalGraphicsStateGuardian::
framebuffer_copy_to_texture(Texture *tex, int view, int z,
                            const DisplayRegion *dr, const RenderBuffer &rb) {
  nassertr(tex != nullptr && dr != nullptr, false);

  if (_cmd == nil || _pass_desc == nil) {
    return false;
  }

  Texture::Format format = tex->get_format();
  bool is_depth = (format == Texture::F_depth_stencil ||
                   format == Texture::F_depth_component ||
                   format == Texture::F_depth_component16 ||
                   format == Texture::F_depth_component24 ||
                   format == Texture::F_depth_component32 ||
                   (rb._buffer_type & (RenderBuffer::T_color | RenderBuffer::T_aux)) == 0);

  // With multisampling, this is the resolved image, which is up-to-date once
  // the render pass has ended (which we do below).
  MetalTexture source = get_copy_source(rb, is_depth);
  if (source == nil || source.sampleCount > 1) {
    return false;
  }

  // Choose a texture format that matches the source exactly, so that we can
  // copy it with a blit.
  Texture::ComponentType component_type;
  switch (source.pixelFormat) {
  case MTLPixelFormatBGRA8Unorm:
    format = Texture::F_rgba;
    component_type = Texture::T_unsigned_byte;
    break;
  case MTLPixelFormatBGRA8Unorm_sRGB:
    format = Texture::F_srgb_alpha;
    component_type = Texture::T_unsigned_byte;
    break;
  case MTLPixelFormatRGBA16Float:
    format = Texture::F_rgba16;
    component_type = Texture::T_half_float;
    break;
  case MTLPixelFormatRGBA32Float:
    format = Texture::F_rgba32;
    component_type = Texture::T_float;
    break;
  case MTLPixelFormatDepth32Float:
    format = Texture::F_depth_component32;
    component_type = Texture::T_float;
    break;
  case MTLPixelFormatDepth32Float_Stencil8:
    format = Texture::F_depth_stencil;
    component_type = Texture::T_float;
    break;
  default:
    return false;
  }

  int xo, yo, w, h;
  dr->get_region_pixels(xo, yo, w, h);
  if (_target_inverted == _target_output_inverted) {
    yo = _target_height - (yo + h);
  }
  if (xo < 0 || yo < 0 || w <= 0 || h <= 0 ||
      xo + w > (int)source.width || yo + h > (int)source.height) {
    return false;
  }

  Texture::TextureType texture_type = (z >= 0) ? Texture::TT_cube_map : Texture::TT_2d_texture;
  if (tex->get_x_size() != w || tex->get_y_size() != h ||
      tex->get_component_type() != component_type ||
      tex->get_format() != format ||
      tex->get_texture_type() != texture_type) {
    if (texture_type == Texture::TT_cube_map) {
      tex->setup_cube_map(w, component_type, format);
    } else {
      tex->setup_2d_texture(w, h, component_type, format);
    }
  }
  if (tex->has_ram_image()) {
    // The contents will be on the GPU only.
    tex->clear_ram_image();
  }

  TextureContext *tc = tex->prepare_now(_prepared_objects, this);
  if (tc == nullptr) {
    return false;
  }
  MetalTextureContext *mtc = DCAST(MetalTextureContext, tc);
  if (mtc->_texture == nil || mtc->was_modified()) {
    if (!upload_texture(mtc)) {
      return false;
    }
  }
  if (mtc->_texture.pixelFormat != source.pixelFormat) {
    return false;
  }

  end_render_pass();

  if (_target_inverted) {
    // The rows are already in the right order, so we can do a straight copy.
    MetalBlitCommandEncoder blit = [_cmd blitCommandEncoder];
    [blit copyFromTexture:source
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(xo, yo, 0)
               sourceSize:MTLSizeMake(w, h, 1)
                toTexture:mtc->_texture
         destinationSlice:(z >= 0) ? z : 0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
  } else {
    // A window is stored with its rows in the opposite order, which a blit
    // can't fix, so we use a render pass that flips the image.
    if (!copy_flipped(source, mtc->_texture, (z >= 0) ? z : 0, xo, yo, w, h)) {
      return false;
    }
  }

  if (mtc->_texture.mipmapLevelCount > 1) {
    MetalBlitCommandEncoder blit = [_cmd blitCommandEncoder];
    [blit generateMipmapsForTexture:mtc->_texture];
    [blit endEncoding];
  }

  mtc->mark_loaded();
  if (!mtc->get_active()) {
    mtc->set_active(true);
  }
  return true;
}

/**
 * Returns the texture of the current render target that a copy of the given
 * buffer should read from, or nil if there is no such texture.  Resolved
 * textures are preferred over multisample textures.
 */
MetalTexture MetalGraphicsStateGuardian::
get_copy_source(const RenderBuffer &rb, bool is_depth) const {
  MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)_pass_desc;
  if (is_depth) {
    MTLRenderPassDepthAttachmentDescriptor *attach = desc.depthAttachment;
    return (attach.resolveTexture != nil) ? attach.resolveTexture : attach.texture;
  }

  // Find the attachment for the requested aux bitplane, if any.
  int index = 0;
  if ((rb._buffer_type & RenderBuffer::T_color) == 0 &&
      (rb._buffer_type & RenderBuffer::T_aux) != 0) {
    index = -1;
    for (int i = 0; i < _target_num_color; ++i) {
      if (DrawableRegion::get_renderbuffer_type(_target_planes[i]) & rb._buffer_type) {
        index = i;
        break;
      }
    }
    if (index < 0) {
      return nil;
    }
  }
  MTLRenderPassColorAttachmentDescriptor *attach = desc.colorAttachments[index];
  return (attach.resolveTexture != nil) ? attach.resolveTexture : attach.texture;
}

/**
 * Copies a region of the source texture to the top-left corner of the given
 * slice of the destination texture, flipping it vertically.  Both textures
 * must have the same pixel format, which may be a color or depth format.
 */
bool MetalGraphicsStateGuardian::
copy_flipped(MetalTexture source, MetalTexture dest, int slice,
             int x, int y, int w, int h) {
  MTLPixelFormat format = dest.pixelFormat;
  bool is_depth = (format == MTLPixelFormatDepth32Float ||
                   format == MTLPixelFormatDepth16Unorm ||
                   format == MTLPixelFormatDepth32Float_Stencil8);

  if ((dest.usage & MTLTextureUsageRenderTarget) == 0 ||
      (source.usage & MTLTextureUsageShaderRead) == 0) {
    metaldisplay_cat.error()
      << "Cannot copy window to texture; texture is not renderable.\n";
    return false;
  }

  MetalRenderPipelineState &pipeline = _copy_flip_pipelines[(unsigned long)format];
  if (pipeline == nil) {
    MTLRenderPipelineDescriptor *pdesc = [[MTLRenderPipelineDescriptor alloc] init];
    pdesc.label = @"Panda3D flipped copy";
    pdesc.vertexFunction = [[_builtin_library newFunctionWithName:@"clear_vertex"] autorelease];
    if (is_depth) {
      pdesc.fragmentFunction = [[_builtin_library newFunctionWithName:@"copy_flip_depth_fragment"] autorelease];
      pdesc.depthAttachmentPixelFormat = format;
      if (format == MTLPixelFormatDepth32Float_Stencil8) {
        pdesc.stencilAttachmentPixelFormat = format;
      }
    } else {
      pdesc.fragmentFunction = [[_builtin_library newFunctionWithName:@"copy_flip_fragment"] autorelease];
      pdesc.colorAttachments[0].pixelFormat = format;
    }
    NSError *error = nil;
    pipeline = [_device newRenderPipelineStateWithDescriptor:pdesc error:&error];
    [pdesc release];
    if (pipeline == nil) {
      metaldisplay_cat.error()
        << "Failed to create copy pipeline: "
        << [[error localizedDescription] UTF8String] << "\n";
      return false;
    }
  }

  MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  if (is_depth) {
    pass.depthAttachment.texture = dest;
    pass.depthAttachment.slice = slice;
    pass.depthAttachment.loadAction = MTLLoadActionLoad;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
  } else {
    pass.colorAttachments[0].texture = dest;
    pass.colorAttachments[0].slice = slice;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  }

  MetalRenderCommandEncoder encoder = [_cmd renderCommandEncoderWithDescriptor:pass];
  [encoder setRenderPipelineState:pipeline];
  if (is_depth) {
    [encoder setDepthStencilState:get_clear_depth_stencil_state(true, false)];
  }
  MTLViewport viewport = {0, 0, (double)w, (double)h, 0, 1};
  [encoder setViewport:viewport];
  float zero = 0.0f;
  [encoder setVertexBytes:&zero length:sizeof(zero) atIndex:0];
  [encoder setFragmentTexture:source atIndex:0];
  int params[4] = {x, y, h, 0};
  [encoder setFragmentBytes:params length:sizeof(params) atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [encoder endEncoding];
  return true;
}

/**
 * Stretches the source texture over the whole of the destination texture with
 * bilinear filtering.  The destination must be renderable.
 */
bool MetalGraphicsStateGuardian::
copy_scaled(MetalTexture source, MetalTexture dest) {
  nassertr(_cmd != nil, false);
  MTLPixelFormat format = dest.pixelFormat;

  MetalRenderPipelineState &pipeline = _scale_pipelines[(unsigned long)format];
  if (pipeline == nil) {
    MTLRenderPipelineDescriptor *pdesc = [[MTLRenderPipelineDescriptor alloc] init];
    pdesc.label = @"Panda3D scaled copy";
    pdesc.vertexFunction = [[_builtin_library newFunctionWithName:@"clear_vertex"] autorelease];
    pdesc.fragmentFunction = [[_builtin_library newFunctionWithName:@"scale_fragment"] autorelease];
    pdesc.colorAttachments[0].pixelFormat = format;
    NSError *error = nil;
    pipeline = [_device newRenderPipelineStateWithDescriptor:pdesc error:&error];
    [pdesc release];
    if (pipeline == nil) {
      metaldisplay_cat.error()
        << "Failed to create scaling pipeline: "
        << [[error localizedDescription] UTF8String] << "\n";
      return false;
    }
  }

  MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = dest;
  pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;

  MetalRenderCommandEncoder encoder = [_cmd renderCommandEncoderWithDescriptor:pass];
  if (metal_debug_labels) {
    encoder.label = @"Panda3D upscale";
  }
  [encoder setRenderPipelineState:pipeline];
  float zero = 0.0f;
  [encoder setVertexBytes:&zero length:sizeof(zero) atIndex:0];
  [encoder setFragmentTexture:source atIndex:0];
  float inv_size[2] = {1.0f / dest.width, 1.0f / dest.height};
  [encoder setFragmentBytes:inv_size length:sizeof(inv_size) atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [encoder endEncoding];
  return true;
}

/**
 * Copy the pixels within the indicated display region from the framebuffer
 * into system memory, not texture memory.  Returns true on success, false on
 * failure.
 *
 * This completely redefines the ram image of the indicated texture.
 */
bool MetalGraphicsStateGuardian::
framebuffer_copy_to_ram(Texture *tex, int view, int z,
                        const DisplayRegion *dr, const RenderBuffer &rb,
                        ScreenshotRequest *request) {
  nassertr(tex != nullptr && dr != nullptr, false);

  if (_cmd == nil || _pass_desc == nil) {
    // We can only copy from the render target of the current frame.
    return false;
  }

  Texture::Format format = tex->get_format();
  bool is_depth = (format == Texture::F_depth_stencil ||
                   format == Texture::F_depth_component ||
                   format == Texture::F_depth_component16 ||
                   format == Texture::F_depth_component24 ||
                   format == Texture::F_depth_component32 ||
                   (rb._buffer_type & (RenderBuffer::T_color | RenderBuffer::T_aux)) == 0);

  // With multisampling, this is the resolved image, which is up-to-date once
  // the render pass has ended (which we do below).
  MetalTexture source = get_copy_source(rb, is_depth);
  if (source == nil || source.sampleCount > 1) {
    return false;
  }

  Texture::ComponentType component_type;
  size_t pixel_size;
  MTLBlitOption options = MTLBlitOptionNone;
  bool half_to_float = false;
  bool swap_float = false;

  switch (source.pixelFormat) {
  case MTLPixelFormatBGRA8Unorm:
    format = Texture::F_rgba;
    component_type = Texture::T_unsigned_byte;
    pixel_size = 4;
    break;

  case MTLPixelFormatBGRA8Unorm_sRGB:
    format = Texture::F_srgb_alpha;
    component_type = Texture::T_unsigned_byte;
    pixel_size = 4;
    break;

  case MTLPixelFormatRGBA16Float:
    format = Texture::F_rgba16;
    component_type = Texture::T_float;
    pixel_size = 8;
    half_to_float = true;
    break;

  case MTLPixelFormatRGBA32Float:
    format = Texture::F_rgba32;
    component_type = Texture::T_float;
    pixel_size = 16;
    swap_float = true;
    break;

  case MTLPixelFormatDepth32Float_Stencil8:
    options = MTLBlitOptionDepthFromDepthStencil;
    // fall through
  case MTLPixelFormatDepth32Float:
    format = Texture::F_depth_component32;
    component_type = Texture::T_float;
    pixel_size = 4;
    break;

  default:
    metaldisplay_cat.error()
      << "Cannot copy framebuffer with pixel format "
      << (int)source.pixelFormat << " to RAM\n";
    return false;
  }

  int xo, yo, w, h;
  dr->get_region_pixels(xo, yo, w, h);
  if (_target_inverted == _target_output_inverted) {
    yo = _target_height - (yo + h);
  }
  if (xo < 0 || yo < 0 || w <= 0 || h <= 0 ||
      xo + w > (int)source.width || yo + h > (int)source.height) {
    return false;
  }

  Texture::TextureType texture_type;
  int z_size;
  if (z >= 0) {
    texture_type = Texture::TT_cube_map;
    z_size = 6;
  } else {
    texture_type = Texture::TT_2d_texture;
    z_size = 1;
  }

  int num_views = tex->get_num_views();
  if (tex->get_x_size() != w || tex->get_y_size() != h ||
      tex->get_z_size() != z_size ||
      tex->get_component_type() != component_type ||
      tex->get_format() != format ||
      tex->get_texture_type() != texture_type ||
      view >= num_views) {

    tex->setup_texture(texture_type, w, h, z_size, component_type, format);

    num_views = std::max(view + 1, num_views);
    if (num_views > 1) {
      tex->set_num_views(num_views);
    }
  }

  nassertr(z < tex->get_z_size(), false);

  // Copy the region into a buffer that we can read from the CPU.
  NSUInteger row_size = w * pixel_size;
  MetalBuffer buffer = [_device newBufferWithLength:row_size * h
                                            options:MTLResourceStorageModeShared];
  nassertr(buffer != nil, false);

  end_render_pass();

  MetalBlitCommandEncoder blit = [_cmd blitCommandEncoder];
  [blit copyFromTexture:source
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(xo, yo, 0)
             sourceSize:MTLSizeMake(w, h, 1)
               toBuffer:buffer
      destinationOffset:0
 destinationBytesPerRow:row_size
destinationBytesPerImage:row_size * h
                options:options];
  [blit endEncoding];

  // Panda stores images bottom-up, whereas Metal stores them top-down.  In an
  // inverted target, the rows are already in the right order.
  size_t dest_row_size = w * tex->get_num_components() * tex->get_component_width();
  bool inverted = _target_inverted;
  auto convert = [=](const unsigned char *src, unsigned char *image_ptr) {
    for (int y = 0; y < h; ++y) {
      int dest_y = inverted ? y : (h - 1 - y);
      const unsigned char *src_row = src + y * row_size;
      unsigned char *dest_row = image_ptr + dest_y * dest_row_size;

      if (half_to_float) {
        // RGBA half floats to BGRA floats.
        const __fp16 *in = (const __fp16 *)src_row;
        float *out = (float *)dest_row;
        for (int x = 0; x < w; ++x) {
          out[x * 4 + 0] = in[x * 4 + 2];
          out[x * 4 + 1] = in[x * 4 + 1];
          out[x * 4 + 2] = in[x * 4 + 0];
          out[x * 4 + 3] = in[x * 4 + 3];
        }
      } else if (swap_float) {
        // RGBA floats to BGRA floats.
        const float *in = (const float *)src_row;
        float *out = (float *)dest_row;
        for (int x = 0; x < w; ++x) {
          out[x * 4 + 0] = in[x * 4 + 2];
          out[x * 4 + 1] = in[x * 4 + 1];
          out[x * 4 + 2] = in[x * 4 + 0];
          out[x * 4 + 3] = in[x * 4 + 3];
        }
      } else {
        memcpy(dest_row, src_row, row_size);
      }
    }
  };

  if (request != nullptr) {
    // Asynchronous: convert the image on a background queue once the GPU is
    // done with this frame, rather than waiting for it here.
    nassertr(dest_row_size * h <= tex->get_expected_ram_page_size(), false);
    PT(ScreenshotRequest) req = request;
    size_t page_size = tex->get_expected_ram_page_size();
    [_cmd addCompletedHandler:^(id<MTLCommandBuffer> cmd) {
      dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        pvector<unsigned char> image(page_size, 0);
        convert((const unsigned char *)[buffer contents], image.data());
        [buffer release];
        req->set_view_data(view, image.data());
        req->finish();
      });
    }];
    return true;
  }

  flush_and_wait();

  size_t image_size = tex->get_expected_ram_page_size();
  unsigned char *image_ptr = tex->modify_ram_image();
  if (z >= 0) {
    image_ptr += z * image_size;
  } else {
    image_size = tex->get_ram_image_size();
  }
  if (view > 0) {
    image_ptr += (view * tex->get_z_size()) * image_size;
    nassertr(view < tex->get_num_views(), false);
  }
  nassertr(dest_row_size * h <= image_size, false);

  convert((const unsigned char *)[buffer contents], image_ptr);
  [buffer release];
  return true;
}

/**
 * Submits the commands recorded so far in this frame and waits for the GPU to
 * finish executing them.  Recording continues in a new command buffer.
 */
void MetalGraphicsStateGuardian::
flush_and_wait() {
  nassertv(_cmd != nil);
  end_render_pass();

  [_cmd commit];
  [_cmd waitUntilCompleted];
  [_cmd release];

  _cmd = [[_queue commandBuffer] retain];
  if (metal_debug_labels) {
    _cmd.label = @"Panda3D frame (continued)";
  }
}

/**
 * Returns the smallest sample count supported by the device that is at least
 * the requested number (so that the framebuffer properties are satisfied), or
 * the largest supported count if the request exceeds what is supported, or 1
 * for no multisampling.
 */
int MetalGraphicsStateGuardian::
choose_sample_count(int requested) const {
  if (requested <= 1) {
    return 1;
  }
  static const int counts[] = {2, 4, 8};
  int best = 1;
  for (int count : counts) {
    if ([_device supportsTextureSampleCount:count]) {
      best = count;
      if (count >= requested) {
        break;
      }
    }
  }
  return best;
}

/**
 * Sets the textures that the following rendering commands will draw into.
 * This ends the current render pass, if any.
 */
void MetalGraphicsStateGuardian::
set_render_target(const RenderTarget &target) {
  end_render_pass();

  if (_pass_desc != nil) {
    [_pass_desc release];
  }

  MTLRenderPassDescriptor *desc = [[MTLRenderPassDescriptor alloc] init];
  _target_num_color = 0;
  for (int i = 0; i < target._num_color && i < RenderTarget::max_color_attachments; ++i) {
    MTLRenderPassColorAttachmentDescriptor *attach = desc.colorAttachments[i];
    attach.texture = target._color[i];
    attach.slice = (target._color[i].textureType == MTLTextureType3D) ? 0 : target._slice;
    attach.depthPlane = (target._color[i].textureType == MTLTextureType3D) ? target._slice : 0;
    attach.loadAction = MTLLoadActionLoad;
    if (target._resolve[i] != nil) {
      // Multisampled; we keep the samples, since the render pass may be
      // restarted later in the frame, and resolve them every time.
      attach.resolveTexture = target._resolve[i];
      attach.resolveSlice = target._slice;
      attach.storeAction = MTLStoreActionStoreAndMultisampleResolve;
    } else {
      attach.storeAction = MTLStoreActionStore;
    }
    _target_planes[i] = target._planes[i];
    ++_target_num_color;
  }

  _target_has_depth = (target._depth != nil);
  if (target._depth != nil) {
    desc.depthAttachment.texture = target._depth;
    desc.depthAttachment.slice = target._slice;
    desc.depthAttachment.loadAction = MTLLoadActionLoad;
    if (target._depth_resolve != nil) {
      desc.depthAttachment.resolveTexture = target._depth_resolve;
      desc.depthAttachment.resolveSlice = target._slice;
      desc.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
      desc.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
    } else {
      desc.depthAttachment.storeAction = MTLStoreActionStore;
    }
  }
  _target_has_stencil = (target._stencil != nil);
  if (target._stencil != nil) {
    desc.stencilAttachment.texture = target._stencil;
    desc.stencilAttachment.slice = target._slice;
    desc.stencilAttachment.loadAction = MTLLoadActionLoad;
    desc.stencilAttachment.storeAction = MTLStoreActionStore;
  }

  if (_visibility_buffer != nil) {
    desc.visibilityResultBuffer = _visibility_buffer;
  }

  _pass_desc = desc;
  _resolved_this_target = false;
  _target_width = target._width;
  _target_height = target._height;
  _target_inverted = target._inverted;
  _target_output_inverted = target._output_inverted;

  _viewport[0] = 0;
  _viewport[1] = 0;
  _viewport[2] = target._width;
  _viewport[3] = target._height;
  _viewport[4] = 0;
  _viewport[5] = 1;
  _scissor[0] = 0;
  _scissor[1] = 0;
  _scissor[2] = target._width;
  _scissor[3] = target._height;
}

/**
 * Convenience overload for a target with a single color attachment and a
 * combined depth/stencil texture, either of which may be nil.
 */
void MetalGraphicsStateGuardian::
set_render_target(MetalTexture color, MetalTexture depth_stencil,
                  int width, int height, bool inverted, MetalTexture resolve) {
  RenderTarget target;
  if (color != nil) {
    target._color[0] = color;
    target._resolve[0] = resolve;
    target._planes[0] = DrawableRegion::RTP_color;
    target._num_color = 1;
  }
  if (depth_stencil != nil) {
    MTLPixelFormat format = depth_stencil.pixelFormat;
    if (format != MTLPixelFormatStencil8) {
      target._depth = depth_stencil;
    }
    if (format == MTLPixelFormatDepth32Float_Stencil8 ||
        format == MTLPixelFormatStencil8) {
      target._stencil = depth_stencil;
    }
  }
  target._width = width;
  target._height = height;
  target._inverted = inverted;
  target._output_inverted = inverted;
  set_render_target(target);
}

/**
 * Returns the render command encoder for the current render target, starting
 * a new render pass if necessary.
 */
MetalRenderCommandEncoder MetalGraphicsStateGuardian::
get_render_encoder() {
  if (_render_encoder != nil) {
    return _render_encoder;
  }

  nassertr(_cmd != nil, nil);
  nassertr(_pass_desc != nil, nil);

  MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)_pass_desc;
  _render_encoder = [[_cmd renderCommandEncoderWithDescriptor:desc] retain];
  _resolved_this_target = true;
  if (metal_debug_labels) {
    _render_encoder.label = @"Panda3D render pass";
  }

  // A new encoder has none of the state we set on the previous one.
  reset_bound_state();

  if (_current_occlusion_query != nullptr) {
    // The samples of each render pass are counted separately.
    begin_visibility_slot();
  }

  // Shaders reading gl_FragCoord expect a lower-left origin.  Inverted targets
  // already have that, but windows have the origin in the upper-left corner.
  float frag_coord_flip[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  if (!_target_inverted) {
    frag_coord_flip[0] = -1.0f;
    frag_coord_flip[1] = (float)_target_height;
  }
  [_render_encoder setFragmentBytes:frag_coord_flip length:sizeof(frag_coord_flip) atIndex:4];

  // If the render pass is restarted later on in the frame, we want it to
  // keep the contents that were rendered so far.
  for (int i = 0; i < _target_num_color; ++i) {
    desc.colorAttachments[i].loadAction = MTLLoadActionLoad;
  }
  desc.depthAttachment.loadAction = MTLLoadActionLoad;
  desc.stencilAttachment.loadAction = MTLLoadActionLoad;

  apply_viewport();
  return _render_encoder;
}

/**
 * Ends the current render pass, if any.  Any pending clears are executed.
 */
void MetalGraphicsStateGuardian::
end_render_pass() {
  if (_render_encoder == nil && _pass_desc != nil && _cmd != nil) {
    // If a clear is pending, we need to start a render pass anyway in order
    // to perform the clear.
    MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)_pass_desc;
    bool needs_pass = desc.depthAttachment.loadAction == MTLLoadActionClear ||
                      desc.stencilAttachment.loadAction == MTLLoadActionClear;
    // Also run the pass at least once if we need to resolve the samples.
    bool needs_resolve = desc.depthAttachment.resolveTexture != nil;
    for (int i = 0; i < _target_num_color; ++i) {
      needs_pass = needs_pass || desc.colorAttachments[i].loadAction == MTLLoadActionClear;
      needs_resolve = needs_resolve || desc.colorAttachments[i].resolveTexture != nil;
    }
    if (needs_pass || (needs_resolve && !_resolved_this_target)) {
      get_render_encoder();
    }
  }

  if (_render_encoder != nil) {
    [_render_encoder endEncoding];
    [_render_encoder release];
    _render_encoder = nil;
  }
}

/**
 * Blocks until the GPU has finished all of the work that was submitted.
 */
void MetalGraphicsStateGuardian::
finish() {
  // Acquiring all the slots means that all submitted frames have completed.
  for (int i = 0; i < _frames_in_flight; ++i) {
    dispatch_semaphore_wait(_frame_semaphore, DISPATCH_TIME_FOREVER);
  }
  for (int i = 0; i < _frames_in_flight; ++i) {
    dispatch_semaphore_signal(_frame_semaphore);
  }
}

/**
 * Applies the current viewport and scissor rectangle to the render encoder.
 */
void MetalGraphicsStateGuardian::
apply_viewport() {
  nassertv(_render_encoder != nil);

  MTLViewport viewport = {_viewport[0], _viewport[1], _viewport[2],
                          _viewport[3], _viewport[4], _viewport[5]};
  [_render_encoder setViewport:viewport];
  _applied_depth_range[0] = _viewport[4];
  _applied_depth_range[1] = _viewport[5];

  MTLScissorRect scissor = {_scissor[0], _scissor[1], _scissor[2], _scissor[3]};
  [_render_encoder setScissorRect:scissor];
}

/**
 * Clears the current viewport region by drawing a quad over it.  This is used
 * when the clear can't be performed as part of the render pass load action.
 */
void MetalGraphicsStateGuardian::
draw_clear_quad(DrawableRegion *clearable, unsigned int color_mask,
                bool clear_depth, bool clear_stencil) {
  if (_scissor[2] == 0 || _scissor[3] == 0) {
    return;
  }

  MetalRenderCommandEncoder encoder = get_render_encoder();
  nassertv(encoder != nil);

  MetalRenderPipelineState pipeline = get_clear_pipeline(color_mask);
  MetalDepthStencilState ds_state = get_clear_depth_stencil_state(clear_depth, clear_stencil);
  nassertv(pipeline != nil && ds_state != nil);

  [encoder setRenderPipelineState:pipeline];
  [encoder setDepthStencilState:ds_state];
  [encoder setCullMode:MTLCullModeNone];
  [encoder setTriangleFillMode:MTLTriangleFillModeFill];
  if (clear_stencil) {
    [encoder setStencilReferenceValue:clearable->get_clear_stencil()];
  }

  // Use the full depth range, so that the depth we write is not affected by
  // the depth range of the display region.
  MTLViewport viewport = {_viewport[0], _viewport[1], _viewport[2],
                          _viewport[3], 0.0, 1.0};
  [encoder setViewport:viewport];

  float depth = clearable->get_clear_depth();
  [encoder setVertexBytes:&depth length:sizeof(depth) atIndex:0];

  LColorf colors[RenderTarget::max_color_attachments];
  for (int i = 0; i < _target_num_color; ++i) {
    colors[i] = LCAST(float, clearable->get_clear_value(_target_planes[i]));
  }
  [encoder setFragmentBytes:colors length:sizeof(colors) atIndex:0];

  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

  apply_viewport();
  _current_pipeline = nil;
  _current_depth_stencil_state = nil;
  _bound_cull_mode = -1;
  _bound_fill_mode = -1;
  _bound_stencil_ref_valid = false;
}

/**
 * Returns a pipeline suitable for clearing the current render target.
 */
MetalRenderPipelineState MetalGraphicsStateGuardian::
get_clear_pipeline(unsigned int color_mask) {
  MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)_pass_desc;
  MetalTexture color_tex = desc.colorAttachments[0].texture;
  MetalTexture depth_tex = desc.depthAttachment.texture;
  MetalTexture stencil_tex = desc.stencilAttachment.texture;

  ClearPipelineKey key;
  memset(&key, 0, sizeof(key));
  for (int i = 0; i < _target_num_color; ++i) {
    key._color_formats[i] = (uint16_t)desc.colorAttachments[i].texture.pixelFormat;
  }
  key._depth_format = (depth_tex != nil) ? depth_tex.pixelFormat : MTLPixelFormatInvalid;
  key._stencil_format = (stencil_tex != nil) ? stencil_tex.pixelFormat : MTLPixelFormatInvalid;
  key._sample_count = (color_tex != nil) ? color_tex.sampleCount
                    : (depth_tex != nil) ? depth_tex.sampleCount : 1;
  key._color_mask = (uint8_t)color_mask;

  auto it = _clear_pipelines.find(key);
  if (it != _clear_pipelines.end()) {
    return it->second;
  }

  MTLRenderPipelineDescriptor *pdesc = [[MTLRenderPipelineDescriptor alloc] init];
  pdesc.label = @"Panda3D clear";
  pdesc.vertexFunction = [[_builtin_library newFunctionWithName:@"clear_vertex"] autorelease];
  if (color_mask != 0) {
    MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];
    int num_colors = _target_num_color;
    [values setConstantValue:&num_colors type:MTLDataTypeInt atIndex:0];
    NSError *error = nil;
    pdesc.fragmentFunction = [[_builtin_library newFunctionWithName:@"clear_fragment"
                                                     constantValues:values
                                                              error:&error] autorelease];
    [values release];
  }
  for (int i = 0; i < _target_num_color; ++i) {
    pdesc.colorAttachments[i].pixelFormat = (MTLPixelFormat)key._color_formats[i];
    pdesc.colorAttachments[i].writeMask = (color_mask & (1u << i)) ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
  }
  pdesc.depthAttachmentPixelFormat = (MTLPixelFormat)key._depth_format;
  pdesc.stencilAttachmentPixelFormat = (MTLPixelFormat)key._stencil_format;
  pdesc.rasterSampleCount = key._sample_count;

  NSError *error = nil;
  MetalRenderPipelineState pipeline =
    [_device newRenderPipelineStateWithDescriptor:pdesc error:&error];
  [pdesc release];

  if (pipeline == nil) {
    metaldisplay_cat.error()
      << "Failed to create clear pipeline: "
      << [[error localizedDescription] UTF8String] << "\n";
    return nil;
  }

  _clear_pipelines[key] = pipeline;
  return pipeline;
}

/**
 * Returns a depth/stencil state that unconditionally writes the depth and/or
 * stencil values, as used for clearing.
 */
MetalDepthStencilState MetalGraphicsStateGuardian::
get_clear_depth_stencil_state(bool write_depth, bool write_stencil) {
  int index = (int)write_depth | ((int)write_stencil << 1);
  MetalDepthStencilState &state = _clear_depth_stencil_states[index];
  if (state != nil) {
    return state;
  }

  MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];
  desc.depthCompareFunction = MTLCompareFunctionAlways;
  desc.depthWriteEnabled = write_depth;

  if (write_stencil) {
    MTLStencilDescriptor *stencil = [[MTLStencilDescriptor alloc] init];
    stencil.stencilCompareFunction = MTLCompareFunctionAlways;
    stencil.stencilFailureOperation = MTLStencilOperationReplace;
    stencil.depthFailureOperation = MTLStencilOperationReplace;
    stencil.depthStencilPassOperation = MTLStencilOperationReplace;
    stencil.readMask = 0xff;
    stencil.writeMask = 0xff;
    desc.frontFaceStencil = stencil;
    desc.backFaceStencil = stencil;
    [stencil release];
  }

  state = [_device newDepthStencilStateWithDescriptor:desc];
  [desc release];
  return state;
}

/**
 * Orders depth/stencil keys so that they can be used in a map.
 */
bool MetalGraphicsStateGuardian::DepthStencilKey::
operator < (const DepthStencilKey &other) const {
  if (_depth_func != other._depth_func) {
    return _depth_func < other._depth_func;
  }
  if (_depth_write != other._depth_write) {
    return _depth_write < other._depth_write;
  }
  return _stencil < other._stencil;
}

/**
 * Creates whatever structures the GSG requires to represent the texture
 * internally, and returns a newly-allocated TextureContext object with this
 * data.  It is the responsibility of the calling function to later call
 * release_texture() with this same pointer (which will also delete the
 * pointer).
 *
 * This function should not be called directly to prepare a texture.  Instead,
 * call Texture::prepare().
 */
TextureContext *MetalGraphicsStateGuardian::
prepare_texture(Texture *tex) {
  MetalTextureContext *tc = new MetalTextureContext(_prepared_objects, tex);
  if (!upload_texture(tc)) {
    delete tc;
    return nullptr;
  }
  return tc;
}

/**
 * Ensures that the current Texture data is refreshed onto the GSG.  This
 * means updating the texture properties and/or re-uploading the texture
 * image, if necessary.  This should only be called within the draw thread.
 *
 * If force is true, this function will not return until the texture has been
 * fully uploaded.  If force is false, the function may choose to upload a
 * simple version of the texture instead, if the texture is not fully resident
 * (and if get_incomplete_render() is true).
 */
bool MetalGraphicsStateGuardian::
update_texture(TextureContext *tc, bool force) {
  return update_texture(tc, force, CompletionToken());
}

/**
 * Ensures that the current Texture data is refreshed onto the GSG.  The
 * completion token is completed when the upload has been submitted.
 */
bool MetalGraphicsStateGuardian::
update_texture(TextureContext *tc, bool force, CompletionToken token) {
  MetalTextureContext *mtc;
  DCAST_INTO_R(mtc, tc, false);

  bool success = true;
  if (mtc->_texture == nil || mtc->was_image_modified() || !texture_matches(mtc)) {
    success = upload_texture(mtc);
  }
  else if (mtc->was_modified()) {
    // Only the sampling properties changed, which are handled by separate
    // sampler objects, so there's no need to upload the image again (which
    // may no longer be in RAM).
    mtc->mark_loaded();
  }
  token.complete(success);
  return success;
}

/**
 * Frees the resources previously allocated via a call to prepare_texture(),
 * including deleting the TextureContext itself, if it is non-NULL.
 */
void MetalGraphicsStateGuardian::
release_texture(TextureContext *tc) {
  // The command buffers retain the textures they use, so it is safe to let
  // go of our reference right away.
  delete tc;
}

/**
 * This method should only be called by the GraphicsEngine.  Do not call it
 * directly; call GraphicsEngine::extract_texture_data() instead.
 *
 * This method will be called in the draw thread to download the texture
 * memory's image into its ram_image value.  It returns true on success, false
 * otherwise.
 */
bool MetalGraphicsStateGuardian::
extract_texture_data(Texture *tex) {
  TextureContext *tc = tex->prepare_now(_prepared_objects, this);
  if (tc == nullptr) {
    return false;
  }
  MetalTextureContext *mtc = DCAST(MetalTextureContext, tc);
  MetalTexture texture = mtc->_texture;
  if (texture == nil || texture.sampleCount > 1) {
    return false;
  }

  // Determine how the texel data maps onto Panda's representation.
  Texture::ComponentType component_type;
  Texture::Format format;
  size_t pixel_size;
  bool swap_red_blue = false;
  MTLBlitOption options = MTLBlitOptionNone;
  switch (texture.pixelFormat) {
  case MTLPixelFormatBGRA8Unorm:
    format = Texture::F_rgba; component_type = Texture::T_unsigned_byte; pixel_size = 4;
    break;
  case MTLPixelFormatBGRA8Unorm_sRGB:
    format = Texture::F_srgb_alpha; component_type = Texture::T_unsigned_byte; pixel_size = 4;
    break;
  case MTLPixelFormatRGBA8Unorm:
    format = Texture::F_rgba; component_type = Texture::T_unsigned_byte; pixel_size = 4;
    swap_red_blue = true;
    break;
  case MTLPixelFormatR8Unorm:
    format = tex->get_format(); component_type = Texture::T_unsigned_byte; pixel_size = 1;
    break;
  case MTLPixelFormatRGBA16Float:
    format = Texture::F_rgba16; component_type = Texture::T_half_float; pixel_size = 8;
    swap_red_blue = true;
    break;
  case MTLPixelFormatRGBA32Float:
    format = Texture::F_rgba32; component_type = Texture::T_float; pixel_size = 16;
    swap_red_blue = true;
    break;
  case MTLPixelFormatR32Float:
    format = Texture::F_r32; component_type = Texture::T_float; pixel_size = 4;
    break;
  case MTLPixelFormatDepth32Float_Stencil8:
    options = MTLBlitOptionDepthFromDepthStencil;
    // fall through
  case MTLPixelFormatDepth32Float:
    format = Texture::F_depth_component32; component_type = Texture::T_float; pixel_size = 4;
    break;
  default:
    metaldisplay_cat.error()
      << "Cannot extract texture with pixel format " << (int)texture.pixelFormat << "\n";
    return false;
  }

  int width = texture.width;
  int height = texture.height;
  int num_slices = (texture.textureType == MTLTextureType3D) ? (int)texture.depth
                 : (texture.textureType == MTLTextureTypeCube) ? 6
                 : (texture.textureType == MTLTextureTypeCubeArray) ? (int)texture.arrayLength * 6
                 : (int)texture.arrayLength;
  bool is_3d = (texture.textureType == MTLTextureType3D);

  NSUInteger row_size = width * pixel_size;
  NSUInteger page_size = row_size * height;
  MetalBuffer buffer = [_device newBufferWithLength:page_size * num_slices
                                            options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return false;
  }

  // If we're in the middle of a frame, the texture may have pending writes
  // in the current command buffer, so we need to flush it.
  bool in_frame = (_cmd != nil);
  MetalCommandBuffer cmd;
  if (in_frame) {
    end_render_pass();
    cmd = _cmd;
  } else {
    cmd = [_queue commandBuffer];
  }

  MetalBlitCommandEncoder blit = [cmd blitCommandEncoder];
  for (int slice = 0; slice < (is_3d ? 1 : num_slices); ++slice) {
    [blit copyFromTexture:texture
              sourceSlice:slice
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(width, height, is_3d ? num_slices : 1)
                 toBuffer:buffer
        destinationOffset:slice * page_size
   destinationBytesPerRow:row_size
 destinationBytesPerImage:page_size
                  options:options];
  }
  [blit endEncoding];

  if (in_frame) {
    flush_and_wait();
  } else {
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  // Keep the texture type and size; just replace the image.
  tex->set_component_type(component_type);
  tex->set_format(format);
  PTA_uchar image = PTA_uchar::empty_array(page_size * num_slices);
  const unsigned char *src = (const unsigned char *)buffer.contents;
  size_t component_width = tex->get_component_width();
  if (swap_red_blue) {
    size_t n = (size_t)width * height * num_slices;
    for (size_t i = 0; i < n; ++i) {
      const unsigned char *s = src + i * pixel_size;
      unsigned char *d = image.p() + i * pixel_size;
      memcpy(d, s, pixel_size);
      memcpy(d, s + 2 * component_width, component_width);
      memcpy(d + 2 * component_width, s, component_width);
    }
  } else {
    memcpy(image.p(), src, page_size * num_slices);
  }
  [buffer release];

  tex->set_ram_image(image);

  // Setting the RAM image marked the texture as modified, but the GPU copy is
  // already up-to-date.
  mtc->mark_loaded();
  return true;
}

/**
 * Information about how a Panda texture is represented in Metal.
 */
struct MetalTextureFormatInfo {
  MTLPixelFormat _format = MTLPixelFormatInvalid;
  int _num_components = 0;
  bool _swap_red_blue = false;
  bool _has_swizzle = false;
  MTLTextureSwizzleChannels _swizzle;
};

/**
 * Determines the Metal pixel format for the given texture.  Returns false if
 * the texture can't be represented.
 */
static bool
get_texture_format_info(const Texture *tex, MetalTextureFormatInfo &info) {
  Texture::Format format = tex->get_format();
  Texture::ComponentType ctype = tex->get_component_type();
  int num = tex->get_num_components();
  bool is_int = Texture::is_integer(format);
  bool is_srgb = Texture::is_srgb(format);

  switch (format) {
  case Texture::F_depth_stencil:
    info._format = MTLPixelFormatDepth32Float_Stencil8;
    info._num_components = 1;
    return true;

  case Texture::F_depth_component:
  case Texture::F_depth_component24:
  case Texture::F_depth_component32:
    info._format = MTLPixelFormatDepth32Float;
    info._num_components = 1;
    return (ctype == Texture::T_float || !tex->has_ram_image());

  case Texture::F_depth_component16:
    info._format = MTLPixelFormatDepth16Unorm;
    info._num_components = 1;
    return (ctype == Texture::T_unsigned_short || !tex->has_ram_image());

  default:
    break;
  }

  // Metal has no three-component formats, so these are expanded to four.
  int mnum = (num == 3) ? 4 : num;
  info._num_components = mnum;
  // Panda stores color images in BGR order, except for integer formats.
  info._swap_red_blue = (num >= 3 && !is_int);

  static const MTLPixelFormat formats[][4][2] = {
    // Indexed by [type][num-1][is_int]
    // T_unsigned_byte
    {{MTLPixelFormatR8Unorm, MTLPixelFormatR8Uint},
     {MTLPixelFormatRG8Unorm, MTLPixelFormatRG8Uint},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Uint}},
    // T_unsigned_short
    {{MTLPixelFormatR16Unorm, MTLPixelFormatR16Uint},
     {MTLPixelFormatRG16Unorm, MTLPixelFormatRG16Uint},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatRGBA16Unorm, MTLPixelFormatRGBA16Uint}},
    // T_float
    {{MTLPixelFormatR32Float, MTLPixelFormatInvalid},
     {MTLPixelFormatRG32Float, MTLPixelFormatInvalid},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatRGBA32Float, MTLPixelFormatInvalid}},
    // T_byte
    {{MTLPixelFormatR8Snorm, MTLPixelFormatR8Sint},
     {MTLPixelFormatRG8Snorm, MTLPixelFormatRG8Sint},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatRGBA8Snorm, MTLPixelFormatRGBA8Sint}},
    // T_short
    {{MTLPixelFormatR16Snorm, MTLPixelFormatR16Sint},
     {MTLPixelFormatRG16Snorm, MTLPixelFormatRG16Sint},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatRGBA16Snorm, MTLPixelFormatRGBA16Sint}},
    // T_half_float
    {{MTLPixelFormatR16Float, MTLPixelFormatInvalid},
     {MTLPixelFormatRG16Float, MTLPixelFormatInvalid},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid}},
    // T_int
    {{MTLPixelFormatInvalid, MTLPixelFormatR32Sint},
     {MTLPixelFormatInvalid, MTLPixelFormatRG32Sint},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatInvalid, MTLPixelFormatRGBA32Sint}},
    // T_unsigned_int
    {{MTLPixelFormatInvalid, MTLPixelFormatR32Uint},
     {MTLPixelFormatInvalid, MTLPixelFormatRG32Uint},
     {MTLPixelFormatInvalid, MTLPixelFormatInvalid},
     {MTLPixelFormatInvalid, MTLPixelFormatRGBA32Uint}},
  };

  int type_index;
  switch (ctype) {
  case Texture::T_unsigned_byte: type_index = 0; break;
  case Texture::T_unsigned_short: type_index = 1; break;
  case Texture::T_float: type_index = 2; break;
  case Texture::T_byte: type_index = 3; break;
  case Texture::T_short: type_index = 4; break;
  case Texture::T_half_float: type_index = 5; break;
  case Texture::T_int: type_index = 6; break;
  case Texture::T_unsigned_int: type_index = 7; break;
  default:
    return false;
  }

  if (mnum < 1 || mnum > 4) {
    return false;
  }
  info._format = formats[type_index][mnum - 1][is_int ? 1 : 0];

  // 8-bit four-component textures are stored in BGRA order, which Metal can
  // use directly.
  if (info._format == MTLPixelFormatRGBA8Unorm) {
    info._format = is_srgb ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatBGRA8Unorm;
    info._swap_red_blue = false;
  }
  else if (is_srgb && info._format == MTLPixelFormatR8Unorm) {
    info._format = MTLPixelFormatR8Unorm_sRGB;
  }
  else if (is_srgb && info._format == MTLPixelFormatRG8Unorm) {
    info._format = MTLPixelFormatRG8Unorm_sRGB;
  }

  // Emulate the single-channel and two-channel formats with a swizzle.
  info._swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                                                MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
  switch (format) {
  case Texture::F_alpha:
    info._swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleOne, MTLTextureSwizzleOne,
                                                  MTLTextureSwizzleOne, MTLTextureSwizzleRed);
    info._has_swizzle = true;
    break;
  case Texture::F_luminance:
  case Texture::F_sluminance:
    info._swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed,
                                                  MTLTextureSwizzleRed, MTLTextureSwizzleOne);
    info._has_swizzle = true;
    break;
  case Texture::F_luminance_alpha:
  case Texture::F_luminance_alphamask:
  case Texture::F_sluminance_alpha:
    info._swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed,
                                                  MTLTextureSwizzleRed, MTLTextureSwizzleGreen);
    info._has_swizzle = true;
    break;
  case Texture::F_green:
    info._swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleZero, MTLTextureSwizzleRed,
                                                  MTLTextureSwizzleZero, MTLTextureSwizzleOne);
    info._has_swizzle = true;
    break;
  case Texture::F_blue:
    info._swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleZero, MTLTextureSwizzleZero,
                                                  MTLTextureSwizzleRed, MTLTextureSwizzleOne);
    info._has_swizzle = true;
    break;
  default:
    break;
  }

  return info._format != MTLPixelFormatInvalid;
}

/**
 * Converts one page of image data from Panda's layout to the layout of the
 * Metal texture, swapping the red and blue channels and adding an alpha
 * channel where required.
 */
static void
convert_texture_page(const unsigned char *src, unsigned char *dest,
                     size_t num_pixels, int src_components, int dest_components,
                     int component_width, bool swap_red_blue,
                     Texture::ComponentType ctype) {
  // The value representing 1 in the alpha channel.
  unsigned char one[4] = {0, 0, 0, 0};
  switch (ctype) {
  case Texture::T_unsigned_byte:
    one[0] = 0xff;
    break;
  case Texture::T_byte:
    one[0] = 0x7f;
    break;
  case Texture::T_unsigned_short:
    one[0] = 0xff; one[1] = 0xff;
    break;
  case Texture::T_short:
    one[0] = 0xff; one[1] = 0x7f;
    break;
  case Texture::T_half_float:
    one[0] = 0x00; one[1] = 0x3c;
    break;
  case Texture::T_float:
    {
      float f = 1.0f;
      memcpy(one, &f, 4);
    }
    break;
  case Texture::T_int:
  case Texture::T_unsigned_int:
    one[0] = 1;
    break;
  default:
    break;
  }

  size_t src_stride = src_components * component_width;
  size_t dest_stride = dest_components * component_width;
  for (size_t i = 0; i < num_pixels; ++i) {
    const unsigned char *s = src + i * src_stride;
    unsigned char *d = dest + i * dest_stride;
    memcpy(d, s, src_stride);
    if (dest_components > src_components) {
      memcpy(d + src_stride, one, component_width);
    }
    if (swap_red_blue) {
      unsigned char tmp[4];
      memcpy(tmp, d, component_width);
      memcpy(d, d + 2 * component_width, component_width);
      memcpy(d + 2 * component_width, tmp, component_width);
    }
  }
}

/**
 * Creates the Metal texture for the given context and uploads the image.
 * Any existing Metal texture is replaced, since it may still be in use by the
 * GPU.
 */
bool MetalGraphicsStateGuardian::
upload_texture(MetalTextureContext *tc) {
  Texture *tex = tc->get_texture();

  MetalTextureFormatInfo info;
  if (!get_texture_format_info(tex, info)) {
    metaldisplay_cat.error()
      << "Texture " << tex->get_name() << " has unsupported format "
      << tex->get_format() << " with component type "
      << tex->get_component_type() << "\n";
    return false;
  }

  if (tex->get_texture_type() == Texture::TT_buffer_texture) {
    // As in OpenGL, the contents of a buffer texture are used as they are,
    // without the usual swapping of the red and blue channels.
    return upload_buffer_texture(tc, info._format, info._num_components, false);
  }

  int width = tex->get_x_size();
  int height = tex->get_y_size();
  int depth = tex->get_z_size();
  Texture::TextureType texture_type = tex->get_texture_type();

  MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
  desc.pixelFormat = info._format;
  desc.width = std::max(width, 1);
  desc.height = std::max(height, 1);

  switch (texture_type) {
  case Texture::TT_1d_texture:
    // 1-D textures are emulated with 2-D textures, since Metal does not
    // support mipmapping 1-D textures.
    desc.textureType = MTLTextureType2D;
    desc.height = 1;
    break;
  case Texture::TT_1d_texture_array:
    desc.textureType = MTLTextureType2DArray;
    desc.arrayLength = height;
    desc.height = 1;
    break;
  case Texture::TT_2d_texture:
    desc.textureType = MTLTextureType2D;
    break;
  case Texture::TT_3d_texture:
    desc.textureType = MTLTextureType3D;
    desc.depth = std::max(depth, 1);
    break;
  case Texture::TT_2d_texture_array:
    desc.textureType = MTLTextureType2DArray;
    desc.arrayLength = std::max(depth, 1);
    break;
  case Texture::TT_cube_map:
    desc.textureType = MTLTextureTypeCube;
    break;
  case Texture::TT_cube_map_array:
    desc.textureType = MTLTextureTypeCubeArray;
    desc.arrayLength = std::max(depth / 6, 1);
    break;
  default:
    metaldisplay_cat.error()
      << "Texture type " << texture_type << " is not yet supported.\n";
    [desc release];
    return false;
  }

  bool is_depth = (info._format == MTLPixelFormatDepth32Float ||
                   info._format == MTLPixelFormatDepth16Unorm ||
                   info._format == MTLPixelFormatDepth32Float_Stencil8);

  // This reloads the image from disk if the RAM copy was released after a
  // previous upload.
  CPTA_uchar image;
  if (!is_depth && tex->might_have_ram_image()) {
    image = tex->get_uncompressed_ram_image();
    if (image.is_null() || image.empty()) {
      image.clear();
    }
  }

  int num_levels = 1;
  if (tex->uses_mipmaps()) {
    num_levels = tex->get_expected_num_mipmap_levels();
  }
  desc.mipmapLevelCount = num_levels;

  desc.usage = MTLTextureUsageShaderRead;
  if ((image.is_null() || is_depth || tex->get_render_to_texture()) && !info._has_swizzle) {
    // This is probably a render target.  Metal does not allow rendering to
    // swizzled textures, but those formats aren't renderable in Panda anyway.
    desc.usage |= MTLTextureUsageRenderTarget;
  }
  // Textures are uploaded through a staging buffer, so that they can live in
  // GPU-private memory, which allows the GPU to compress them losslessly.
  desc.storageMode = MTLStorageModePrivate;
  if (info._has_swizzle) {
    desc.swizzle = info._swizzle;
  }

  MetalTexture texture = [_device newTextureWithDescriptor:desc];
  [desc release];
  if (texture == nil) {
    metaldisplay_cat.error()
      << "Failed to create Metal texture for " << tex->get_name() << "\n";
    return false;
  }
  if (metal_debug_labels) {
    texture.label = [NSString stringWithUTF8String:tex->get_name().c_str()];
  }

  int num_pages = (texture_type == Texture::TT_3d_texture) ? 1 : std::max(depth, 1);
  if (texture_type == Texture::TT_1d_texture_array) {
    num_pages = 1;
  }
  size_t component_width = tex->get_component_width();
  int src_components = tex->get_num_components();

  // Gather the mipmap levels we have into a staging buffer.
  int num_available_levels = 0;
  if (!image.is_null()) {
    num_available_levels = std::min(tex->get_num_ram_mipmap_images(), num_levels);
  }
  else if (tex->has_clear_color() && !is_depth) {
    num_available_levels = num_levels;
  }

  vector_uchar clear_data;
  if (image.is_null() && num_available_levels > 0) {
    clear_data = tex->get_clear_data();
  }

  struct CopyRegion {
    size_t _offset;
    NSUInteger _bytes_per_row;
    NSUInteger _bytes_per_image;
    MTLSize _size;
    NSUInteger _level;
    NSUInteger _slice;
  };
  pvector<CopyRegion> regions;
  pvector<unsigned char> staging;

  int num_uploaded_levels = 0;
  size_t dest_pixel_size = info._num_components * component_width;
  for (int level = 0; level < num_available_levels; ++level) {
    int level_width = std::max(width >> level, 1);
    int level_height = std::max(height >> level, 1);
    int level_depth = (texture_type == Texture::TT_3d_texture) ? std::max(depth >> level, 1) : 1;
    if (texture_type == Texture::TT_1d_texture_array) {
      level_depth = 1;
    }

    size_t src_page_pixels = (size_t)level_width * level_height * level_depth;
    if (texture_type == Texture::TT_1d_texture_array) {
      src_page_pixels = (size_t)level_width * height;
    }
    size_t src_page_size = src_page_pixels * src_components * component_width;

    CPTA_uchar level_image;
    if (!image.is_null()) {
      level_image = (level == 0) ? image : tex->get_ram_mipmap_image(level);
      if (level_image.is_null() || level_image.size() < src_page_size * num_pages) {
        // The rest will be generated.
        break;
      }
    }

    NSUInteger bytes_per_row = level_width * dest_pixel_size;
    size_t dest_page_size = src_page_pixels * dest_pixel_size;

    for (int page = 0; page < num_pages; ++page) {
      const unsigned char *src;
      pvector<unsigned char> cleared;
      if (!level_image.is_null()) {
        src = level_image.p() + page * src_page_size;
      } else {
        // Fill the page with the clear color.
        cleared.resize(src_page_size);
        size_t texel_size = src_components * component_width;
        nassertd(clear_data.size() >= texel_size) break;
        for (size_t i = 0; i < src_page_pixels; ++i) {
          memcpy(cleared.data() + i * texel_size, clear_data.data(), texel_size);
        }
        src = cleared.data();
      }

      size_t offset = staging.size();
      staging.resize(offset + dest_page_size);
      if (info._num_components != src_components || info._swap_red_blue) {
        convert_texture_page(src, staging.data() + offset, src_page_pixels,
                             src_components, info._num_components,
                             component_width, info._swap_red_blue,
                             tex->get_component_type());
      } else {
        memcpy(staging.data() + offset, src, dest_page_size);
      }

      if (texture_type == Texture::TT_1d_texture_array) {
        // Each row is a separate slice.
        for (int slice = 0; slice < height; ++slice) {
          regions.push_back({offset + slice * bytes_per_row, bytes_per_row, bytes_per_row,
                             MTLSizeMake(level_width, 1, 1), (NSUInteger)level, (NSUInteger)slice});
        }
      } else {
        regions.push_back({offset, bytes_per_row, bytes_per_row * level_height,
                           MTLSizeMake(level_width, level_height, level_depth), (NSUInteger)level,
                           (NSUInteger)((texture_type == Texture::TT_3d_texture) ? 0 : page)});
      }
    }
    num_uploaded_levels = level + 1;
  }

  bool generate_mipmaps = (num_uploaded_levels > 0 && num_uploaded_levels < num_levels &&
                           !Texture::is_integer(tex->get_format()));

  if (!regions.empty() || generate_mipmaps) {
    // This command buffer is committed before the frame's command buffer, so
    // it will have completed by the time the texture is used.
    MetalCommandBuffer cmd = [_queue commandBuffer];
    MetalBlitCommandEncoder blit = [cmd blitCommandEncoder];
    if (!regions.empty()) {
      MetalBuffer buffer = [_device newBufferWithBytes:staging.data()
                                                length:staging.size()
                                               options:MTLResourceStorageModeShared];
      for (const CopyRegion &region : regions) {
        [blit copyFromBuffer:buffer
                sourceOffset:region._offset
           sourceBytesPerRow:region._bytes_per_row
         sourceBytesPerImage:region._bytes_per_image
                  sourceSize:region._size
                   toTexture:texture
            destinationSlice:region._slice
            destinationLevel:region._level
           destinationOrigin:MTLOriginMake(0, 0, 0)];
      }
      [buffer release];
    }
    if (generate_mipmaps) {
      [blit generateMipmapsForTexture:texture];
    }
    [blit endEncoding];
    [cmd commit];
  }

  if (tc->_texture != nil) {
    release_later(tc->_texture);
  }
  tc->_texture = texture;
  tc->_pixel_format = info._format;
  tc->_width = width;
  tc->_height = height;
  tc->_depth = depth;
  tc->_num_levels = num_levels;
  tc->_texture_type = texture_type;

  size_t total_size = 0;
  for (int level = 0; level < num_levels; ++level) {
    total_size += (size_t)std::max(width >> level, 1) * std::max(height >> level, 1) *
                  std::max(depth, 1) * info._num_components * component_width;
  }
  tc->update_data_size_bytes(total_size);
  tc->mark_loaded();

  // The GPU has its own copy now, so let the engine release the RAM image
  // (subject to keep-texture-ram); it is reloaded from disk if needed.
  if (!image.is_null() && !tex->get_fullpath().empty()) {
    get_engine()->texture_uploaded(tex);
  }
  return true;
}

/**
 * Creates a buffer texture for the given context.  On Apple Silicon, the
 * memory is shared with the CPU, so we write the texels straight into a
 * buffer and let the texture refer to it.
 */
bool MetalGraphicsStateGuardian::
upload_buffer_texture(MetalTextureContext *tc, unsigned long format,
                      int num_components, bool swap_red_blue) {
  Texture *tex = tc->get_texture();
  int width = std::max(tex->get_x_size(), 1);
  size_t component_width = tex->get_component_width();
  int src_components = tex->get_num_components();
  size_t pixel_size = num_components * component_width;

  // Each "row" of the buffer must be suitably aligned.
  NSUInteger alignment = [_device minimumTextureBufferAlignmentForPixelFormat:(MTLPixelFormat)format];
  NSUInteger bytes_per_row = (width * pixel_size + alignment - 1) / alignment * alignment;

  MetalBuffer buffer = [_device newBufferWithLength:bytes_per_row
                                            options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return false;
  }
  unsigned char *dest = (unsigned char *)buffer.contents;

  CPTA_uchar image;
  if (tex->might_have_ram_image()) {
    image = tex->get_uncompressed_ram_image();
  }
  pvector<unsigned char> cleared;
  const unsigned char *src = nullptr;
  size_t src_size = (size_t)width * src_components * component_width;
  if (!image.is_null() && image.size() >= src_size) {
    src = image.p();
  } else if (tex->has_clear_color()) {
    vector_uchar clear_data = tex->get_clear_data();
    size_t texel_size = src_components * component_width;
    if (clear_data.size() >= texel_size) {
      cleared.resize(src_size);
      for (int i = 0; i < width; ++i) {
        memcpy(cleared.data() + i * texel_size, clear_data.data(), texel_size);
      }
      src = cleared.data();
    }
  }
  if (src != nullptr) {
    if (num_components != src_components || swap_red_blue) {
      convert_texture_page(src, dest, width, src_components, num_components,
                           component_width, swap_red_blue, tex->get_component_type());
    } else {
      memcpy(dest, src, src_size);
    }
  } else {
    memset(dest, 0, bytes_per_row);
  }

  MTLTextureDescriptor *desc =
    [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:(MTLPixelFormat)format
                                                           width:width
                                                 resourceOptions:MTLResourceStorageModeShared
                                                           usage:MTLTextureUsageShaderRead];
  MetalTexture texture = [buffer newTextureWithDescriptor:desc offset:0 bytesPerRow:bytes_per_row];
  [buffer release];
  if (texture == nil) {
    metaldisplay_cat.error()
      << "Failed to create buffer texture for " << tex->get_name() << "\n";
    return false;
  }

  if (tc->_texture != nil) {
    release_later(tc->_texture);
  }
  tc->_texture = texture;
  tc->_pixel_format = format;
  tc->_width = tex->get_x_size();
  tc->_height = tex->get_y_size();
  tc->_depth = tex->get_z_size();
  tc->_num_levels = 1;
  tc->_texture_type = Texture::TT_buffer_texture;
  tc->update_data_size_bytes(bytes_per_row);
  tc->mark_loaded();
  return true;
}

/**
 * Returns true if the Metal texture of the given context still has the size,
 * format and type that the Panda texture calls for.
 */
bool MetalGraphicsStateGuardian::
texture_matches(MetalTextureContext *tc) const {
  Texture *tex = tc->get_texture();
  if (tc->_width != tex->get_x_size() || tc->_height != tex->get_y_size() ||
      tc->_depth != tex->get_z_size() || tc->_texture_type != tex->get_texture_type()) {
    return false;
  }
  MetalTextureFormatInfo info;
  if (!get_texture_format_info(tex, info) || (unsigned long)info._format != tc->_pixel_format) {
    return false;
  }
  int num_levels = tex->uses_mipmaps() ? tex->get_expected_num_mipmap_levels() : 1;
  return num_levels == tc->_num_levels;
}

/**
 * Prepares and updates the given texture, and returns the Metal texture to
 * bind, or nil if the texture could not be prepared.
 */
MetalTexture MetalGraphicsStateGuardian::
use_texture(Texture *tex, int view) {
  TextureContext *tc = tex->prepare_now(_prepared_objects, this);
  if (tc == nullptr) {
    return nil;
  }
  MetalTextureContext *mtc = DCAST(MetalTextureContext, tc);
  if (!update_texture(mtc, true)) {
    return nil;
  }
  if (!mtc->get_active()) {
    mtc->set_active(true);
  }
  return mtc->_texture;
}

/**
 * Returns a 1x1 white texture of the given type, for use when a shader
 * samples a texture that hasn't been provided.
 */
MetalTexture MetalGraphicsStateGuardian::
get_white_texture(Texture::TextureType type) {
  auto it = _white_textures.find((int)type);
  if (it != _white_textures.end()) {
    return it->second;
  }

  if (type == Texture::TT_buffer_texture) {
    static const uint32_t white[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    NSUInteger alignment = [_device minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatBGRA8Unorm];
    MetalBuffer buffer = [_device newBufferWithLength:std::max(alignment, (NSUInteger)16)
                                              options:MTLResourceStorageModeShared];
    memcpy(buffer.contents, white, sizeof(white));
    MTLTextureDescriptor *desc =
      [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                             width:1
                                                   resourceOptions:MTLResourceStorageModeShared
                                                             usage:MTLTextureUsageShaderRead];
    MetalTexture texture = [buffer newTextureWithDescriptor:desc offset:0
                                                bytesPerRow:std::max(alignment, (NSUInteger)4)];
    [buffer release];
    _white_textures[(int)type] = texture;
    return texture;
  }

  MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
  desc.pixelFormat = MTLPixelFormatBGRA8Unorm;
  desc.width = 1;
  desc.height = 1;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;

  int num_slices = 1;
  switch (type) {
  case Texture::TT_1d_texture: desc.textureType = MTLTextureType2D; break;
  case Texture::TT_1d_texture_array: desc.textureType = MTLTextureType2DArray; break;
  case Texture::TT_3d_texture: desc.textureType = MTLTextureType3D; break;
  case Texture::TT_2d_texture_array: desc.textureType = MTLTextureType2DArray; break;
  case Texture::TT_cube_map: desc.textureType = MTLTextureTypeCube; num_slices = 6; break;
  case Texture::TT_cube_map_array: desc.textureType = MTLTextureTypeCubeArray; num_slices = 6; break;
  default: desc.textureType = MTLTextureType2D; break;
  }

  MetalTexture texture = [_device newTextureWithDescriptor:desc];
  [desc release];
  nassertr(texture != nil, nil);

  static const uint32_t white = 0xffffffff;
  for (int slice = 0; slice < num_slices; ++slice) {
    [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:slice
                 withBytes:&white bytesPerRow:4 bytesPerImage:4];
  }

  _white_textures[(int)type] = texture;
  return texture;
}

/**
 * Creates whatever structures the GSG requires to represent the sampler
 * internally, and returns a newly-allocated SamplerContext object with this
 * data.
 */
SamplerContext *MetalGraphicsStateGuardian::
prepare_sampler(const SamplerState &sampler) {
  auto get_address_mode = [](SamplerState::WrapMode mode) {
    switch (mode) {
    case SamplerState::WM_clamp: return MTLSamplerAddressModeClampToEdge;
    case SamplerState::WM_repeat: return MTLSamplerAddressModeRepeat;
    case SamplerState::WM_mirror: return MTLSamplerAddressModeMirrorRepeat;
    case SamplerState::WM_mirror_once: return MTLSamplerAddressModeMirrorClampToEdge;
    case SamplerState::WM_border_color: return MTLSamplerAddressModeClampToBorderColor;
    default: return MTLSamplerAddressModeRepeat;
    }
  };

  MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
  desc.sAddressMode = get_address_mode(sampler.get_wrap_u());
  desc.tAddressMode = get_address_mode(sampler.get_wrap_v());
  desc.rAddressMode = get_address_mode(sampler.get_wrap_w());

  SamplerState::FilterType minfilter = sampler.get_effective_minfilter();
  SamplerState::FilterType magfilter = sampler.get_effective_magfilter();

  switch (minfilter) {
  case SamplerState::FT_nearest:
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    break;
  case SamplerState::FT_nearest_mipmap_nearest:
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.mipFilter = MTLSamplerMipFilterNearest;
    break;
  case SamplerState::FT_nearest_mipmap_linear:
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.mipFilter = MTLSamplerMipFilterLinear;
    break;
  case SamplerState::FT_linear_mipmap_nearest:
    desc.minFilter = MTLSamplerMinMagFilterLinear;
    desc.mipFilter = MTLSamplerMipFilterNearest;
    break;
  case SamplerState::FT_linear_mipmap_linear:
    desc.minFilter = MTLSamplerMinMagFilterLinear;
    desc.mipFilter = MTLSamplerMipFilterLinear;
    break;
  case SamplerState::FT_shadow:
    desc.minFilter = MTLSamplerMinMagFilterLinear;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    desc.compareFunction = MTLCompareFunctionLessEqual;
    break;
  default:
    desc.minFilter = MTLSamplerMinMagFilterLinear;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    break;
  }
  desc.magFilter = (magfilter == SamplerState::FT_nearest)
                 ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;

  int anisotropy = sampler.get_effective_anisotropic_degree();
  desc.maxAnisotropy = std::max(1, std::min(anisotropy, 16));

  desc.lodMinClamp = std::max((float)sampler.get_min_lod(), 0.0f);
  desc.lodMaxClamp = std::max((float)sampler.get_max_lod(), desc.lodMinClamp);

  // Metal only supports three border colors, so pick the closest one.
  LColor border = sampler.get_border_color();
  if (border[3] < 0.5f) {
    desc.borderColor = MTLSamplerBorderColorTransparentBlack;
  } else if (border[0] + border[1] + border[2] >= 1.5f) {
    desc.borderColor = MTLSamplerBorderColorOpaqueWhite;
  } else {
    desc.borderColor = MTLSamplerBorderColorOpaqueBlack;
  }

  MetalSamplerState state = [_device newSamplerStateWithDescriptor:desc];
  [desc release];
  if (state == nil) {
    return nullptr;
  }
  return new MetalSamplerContext(sampler, state);
}

/**
 * Frees the resources previously allocated via a call to prepare_sampler(),
 * including deleting the SamplerContext itself, if it is non-NULL.
 */
void MetalGraphicsStateGuardian::
release_sampler(SamplerContext *sc) {
  delete sc;
}

/**
 * Returns the Metal sampler state corresponding to the given sampler.
 */
MetalSamplerState MetalGraphicsStateGuardian::
get_sampler_state(const SamplerState &sampler) {
  SamplerContext *sc = sampler.prepare_now(get_prepared_objects(), this);
  nassertr(sc != nullptr, nil);
  return DCAST(MetalSamplerContext, sc)->_sampler_state;
}

/**
 * Compile a vertex/fragment shader body.
 */
ShaderContext *MetalGraphicsStateGuardian::
prepare_shader(Shader *shader) {
  // Shaders written in GLSL versions before 3.30 (except 1.50) are not
  // compiled to SPIR-V, but passed straight to the OpenGL driver.  Since Metal
  // needs SPIR-V, we upgrade them to GLSL 3.30 and compile them ourselves.
  PT(Shader) source = shader;
  for (const Shader::LinkedModule &linked_module : shader->_modules) {
    CPT(ShaderModule) module = linked_module._module.get_read_pointer();
    if (module != nullptr && module->is_of_type(ShaderModuleGlsl::get_class_type())) {
      source = upgrade_legacy_glsl_shader(shader);
      if (source == nullptr) {
        return nullptr;
      }
      break;
    }
  }

  MetalShaderContext *sc = new MetalShaderContext(shader, source);
  if (!sc->create_functions(this)) {
    delete sc;
    return nullptr;
  }
  return sc;
}

/**
 * Rewrites the source of a legacy GLSL module so that it can be compiled as
 * GLSL 3.30, by defining away the deprecated features that the old versions
 * allow.  Returns the empty string on failure.
 */
static std::string
upgrade_legacy_glsl(const std::string &code, Shader::Stage stage) {
  size_t pos = code.find("#version");
  if (pos == std::string::npos) {
    pos = 0;
  }
  size_t eol = code.find('\n', pos);
  if (eol == std::string::npos) {
    return std::string();
  }

  // Keep any #extension directives right after it, since they must come
  // before anything else.
  size_t insert_pos = eol + 1;

  std::ostringstream shims;
  shims << "#define texture1D texture\n"
           "#define texture2D texture\n"
           "#define texture3D texture\n"
           "#define textureCube texture\n"
           "#define texture1DLod textureLod\n"
           "#define texture2DLod textureLod\n"
           "#define texture3DLod textureLod\n"
           "#define textureCubeLod textureLod\n"
           "#define texture2DProj textureProj\n"
           "#define texture2DProjLod textureProjLod\n"
           "#define shadow2D(s, c) vec4(texture(s, c))\n"
           "#define shadow2DProj(s, c) vec4(textureProj(s, c))\n";

  if (stage == Shader::Stage::VERTEX) {
    shims << "#define attribute in\n"
             "#define varying out\n";
  }
  else if (stage == Shader::Stage::FRAGMENT) {
    shims << "#define varying in\n";
    if (code.find("gl_FragColor") != std::string::npos) {
      shims << "layout(location = 0) out vec4 panda_FragColor;\n"
               "#define gl_FragColor panda_FragColor\n";
    }
    else if (code.find("gl_FragData") != std::string::npos) {
      shims << "layout(location = 0) out vec4 panda_FragData[8];\n"
               "#define gl_FragData panda_FragData\n";
    }
  }

  std::string result = "#version 330\n";
  // Copy any lines between the version directive and the first non-directive
  // that are #extension directives.
  size_t p = insert_pos;
  while (p < code.size()) {
    size_t line_end = code.find('\n', p);
    if (line_end == std::string::npos) {
      line_end = code.size();
    }
    std::string line = code.substr(p, line_end - p);
    size_t first = line.find_first_not_of(" \t\r");
    if (first != std::string::npos && line.compare(first, 10, "#extension") == 0) {
      result += line + "\n";
      p = line_end + 1;
      continue;
    }
    break;
  }
  result += shims.str();
  result += "#line 2\n";
  if (p < code.size()) {
    result += code.substr(p);
  }
  // Preserve whatever came before the #version directive (comments).
  return result;
}

/**
 * Recompiles a shader consisting of legacy GLSL modules as GLSL 3.30, so that
 * it goes through the SPIR-V pipeline.  Returns nullptr on failure.
 */
PT(Shader) MetalGraphicsStateGuardian::
upgrade_legacy_glsl_shader(Shader *shader) {
  std::string sources[(size_t)Shader::Stage::COMPUTE + 1];
  for (const Shader::LinkedModule &linked_module : shader->_modules) {
    CPT(ShaderModule) module = linked_module._module.get_read_pointer();
    if (module == nullptr) {
      continue;
    }
    Shader::Stage stage = module->get_stage();
    if (module->is_of_type(ShaderModuleGlsl::get_class_type())) {
      const ShaderModuleGlsl *glsl = (const ShaderModuleGlsl *)module.p();
      sources[(size_t)stage] = upgrade_legacy_glsl(glsl->_code, stage);
    } else {
      metaldisplay_cat.error()
        << "Cannot mix legacy GLSL with other shader modules in "
        << shader->get_debug_name() << "\n";
      return nullptr;
    }
    if (metal_dump_msl) {
      metaldisplay_cat.info()
        << "Upgraded legacy GLSL " << stage << " stage of " << shader->get_debug_name()
        << ":\n" << sources[(size_t)stage] << "\n";
    }
    if (sources[(size_t)stage].empty()) {
      metaldisplay_cat.error()
        << "Failed to upgrade legacy GLSL in " << shader->get_debug_name() << "\n";
      return nullptr;
    }
  }

  PT(Shader) result;
  if (!sources[(size_t)Shader::Stage::COMPUTE].empty()) {
    result = Shader::make_compute(Shader::SL_GLSL, sources[(size_t)Shader::Stage::COMPUTE]);
  } else {
    result = Shader::make(Shader::SL_GLSL,
                          sources[(size_t)Shader::Stage::VERTEX],
                          sources[(size_t)Shader::Stage::FRAGMENT],
                          sources[(size_t)Shader::Stage::GEOMETRY],
                          sources[(size_t)Shader::Stage::TESS_CONTROL],
                          sources[(size_t)Shader::Stage::TESS_EVALUATION]);
  }

  if (result == nullptr) {
    metaldisplay_cat.error()
      << "Failed to recompile legacy GLSL shader " << shader->get_debug_name()
      << " as GLSL 3.30 for the Metal renderer.\n";
    return nullptr;
  }

  if (metaldisplay_cat.is_debug()) {
    metaldisplay_cat.debug()
      << "Upgraded legacy GLSL shader " << shader->get_debug_name()
      << " to GLSL 3.30\n";
  }
  return result;
}

/**
 * Releases the resources allocated by prepare_shader
 */
void MetalGraphicsStateGuardian::
release_shader(ShaderContext *sc) {
  if (sc == _current_sc) {
    _current_sc = nullptr;
    _current_shader = nullptr;
  }
  delete sc;
}

/**
 * Prepares the indicated buffer for retained-mode rendering.
 */
VertexBufferContext *MetalGraphicsStateGuardian::
prepare_vertex_buffer(GeomVertexArrayData *data) {
  MetalVertexBufferContext *vbc = new MetalVertexBufferContext(_prepared_objects, data);
  CPT(GeomVertexArrayDataHandle) handle = data->get_handle();
  if (!update_vertex_buffer(vbc, handle, false)) {
    delete vbc;
    return nullptr;
  }
  return vbc;
}

/**
 * Makes sure that the data in the vertex buffer is up-to-date.
 */
bool MetalGraphicsStateGuardian::
update_vertex_buffer(MetalVertexBufferContext *vbc,
                     const GeomVertexArrayDataHandle *reader, bool force) {
  if (vbc->_buffer != nil && !vbc->was_modified(reader)) {
    return true;
  }

  size_t num_bytes = reader->get_data_size_bytes();
  const unsigned char *data = reader->get_read_pointer(force);
  if (data == nullptr && num_bytes > 0) {
    return false;
  }

  if (vbc->_buffer != nil && vbc->_buffer.length >= num_bytes &&
      vbc->_last_use_frame <= _completed_frame_number.load()) {
    // The GPU is done with it, so we can overwrite it in place.
    memcpy(vbc->_buffer.contents, data, num_bytes);
  } else {
    // The buffer may still be in use; the command buffers referencing it
    // keep it alive, so we can simply create a new one.
    if (vbc->_buffer != nil) {
      release_later(vbc->_buffer);
    }
    vbc->_buffer = [_device newBufferWithLength:std::max(num_bytes, (size_t)4)
                                        options:MTLResourceStorageModeShared];
    if (vbc->_buffer == nil) {
      return false;
    }
    if (num_bytes > 0) {
      memcpy(vbc->_buffer.contents, data, num_bytes);
    }
  }

  vbc->update_data_size_bytes(num_bytes);
  vbc->mark_loaded(reader);
  return true;
}

/**
 * Frees the GL resources previously allocated for the data.
 */
void MetalGraphicsStateGuardian::
release_vertex_buffer(VertexBufferContext *vbc) {
  delete vbc;
}

/**
 * Creates a new retained-mode representation of the given data, and returns
 * a newly-allocated IndexBufferContext pointer to reference it.
 */
IndexBufferContext *MetalGraphicsStateGuardian::
prepare_index_buffer(GeomPrimitive *primitive) {
  MetalIndexBufferContext *ibc = new MetalIndexBufferContext(_prepared_objects, primitive);
  GeomPrimitivePipelineReader reader(primitive, Thread::get_current_thread());
  if (!update_index_buffer(ibc, &reader, false)) {
    delete ibc;
    return nullptr;
  }
  return ibc;
}

/**
 * Makes sure that the data in the index buffer is up-to-date.
 */
bool MetalGraphicsStateGuardian::
update_index_buffer(MetalIndexBufferContext *ibc,
                    const GeomPrimitivePipelineReader *reader, bool force) {
  if (ibc->_buffer != nil && !ibc->was_modified(reader)) {
    return true;
  }

  GeomEnums::NumericType index_type = reader->get_index_type();
  const unsigned char *data = reader->get_read_pointer(force);
  size_t num_bytes = reader->get_data_size_bytes();
  if (data == nullptr && num_bytes > 0) {
    return false;
  }

  // Metal does not support 8-bit indices, so we widen them to 16 bits.
  pvector<uint16_t> widened;
  if (index_type == GeomEnums::NT_uint8) {
    widened.resize(num_bytes);
    for (size_t i = 0; i < num_bytes; ++i) {
      // Keep the strip-cut index working.
      widened[i] = (data[i] == 0xff) ? 0xffff : data[i];
    }
    data = (const unsigned char *)widened.data();
    num_bytes *= 2;
    ibc->_index_type = MTLIndexTypeUInt16;
    ibc->_widened = true;
  }
  else if (index_type == GeomEnums::NT_uint16) {
    ibc->_index_type = MTLIndexTypeUInt16;
  }
  else if (index_type == GeomEnums::NT_uint32) {
    ibc->_index_type = MTLIndexTypeUInt32;
  }
  else {
    metaldisplay_cat.error()
      << "Unsupported index type " << index_type << "\n";
    return false;
  }

  if (ibc->_buffer != nil && ibc->_buffer.length >= num_bytes &&
      ibc->_last_use_frame <= _completed_frame_number.load()) {
    memcpy(ibc->_buffer.contents, data, num_bytes);
  } else {
    if (ibc->_buffer != nil) {
      release_later(ibc->_buffer);
    }
    ibc->_buffer = [_device newBufferWithLength:std::max(num_bytes, (size_t)4)
                                        options:MTLResourceStorageModeShared];
    if (ibc->_buffer == nil) {
      return false;
    }
    if (num_bytes > 0) {
      memcpy(ibc->_buffer.contents, data, num_bytes);
    }
  }

  ibc->update_data_size_bytes(num_bytes);
  ibc->mark_loaded(reader);
  return true;
}

/**
 * Frees the GL resources previously allocated for the data.
 */
void MetalGraphicsStateGuardian::
release_index_buffer(IndexBufferContext *ibc) {
  delete ibc;
}

/**
 * Creates a new GeomMunger object to munge vertices appropriate to this GSG
 * for the indicated state.
 */
PT(GeomMunger) MetalGraphicsStateGuardian::
make_geom_munger(const RenderState *state, Thread *current_thread) {
  PT(MetalGeomMunger) munger = new MetalGeomMunger(this, state);
  return GeomMunger::register_munger(munger, current_thread);
}

/**
 * Allocates memory that remains valid until the GPU has finished the current
 * frame.  Returns a CPU pointer to the memory, and the buffer and offset at
 * which the GPU can find it.
 */
void *MetalGraphicsStateGuardian::
alloc_transient_buffer(size_t size, MetalBuffer &buffer, size_t &offset) {
  if (_transient_buffer != nil) {
    ptrdiff_t result = _transient_allocator.alloc(size, 256);
    if (result < 0 && size * 4 <= _transient_allocator.get_capacity() * 16) {
      // The ring buffer is full, so replace it with one twice the size.  The
      // generation number keeps old completion handlers from touching it.
      size_t new_size = _transient_allocator.get_capacity() * 2;
      MetalBuffer new_buffer = [_device newBufferWithLength:new_size options:MTLResourceStorageModeShared];
      if (new_buffer != nil) {
        if (metaldisplay_cat.is_debug()) {
          metaldisplay_cat.debug()
            << "Growing transient buffer to " << new_size << " bytes\n";
        }
        release_later(_transient_buffer);
        _transient_buffer = new_buffer;
        _transient_allocator.reset(new_size);
        _transient_generation.fetch_add(1);
        result = _transient_allocator.alloc(size, 256);
      }
    }
    if (result >= 0) {
      buffer = _transient_buffer;
      offset = (size_t)result;
      return (unsigned char *)_transient_buffer.contents + result;
    }
  }

  // The ring buffer is full, so make a one-off buffer.  The command buffer
  // keeps it alive for as long as it is needed.
  buffer = [[_device newBufferWithLength:size options:MTLResourceStorageModeShared] autorelease];
  offset = 0;
  return buffer.contents;
}

/**
 * Simultaneously resets the render state and the transform state.
 *
 * This transform specified is the "internal" net transform, already converted
 * into the GSG's internal coordinate space by composing it to
 * get_cs_transform().  (Previously, this used to be the "external" net
 * transform, with the assumption that that GSG would convert it internally,
 * but that is no longer the case.)
 */
void MetalGraphicsStateGuardian::
set_state_and_transform(const RenderState *state,
                        const TransformState *trans) {
  // The state is actually applied in begin_draw_primitives(), because the
  // pipeline state depends on the vertex format.
  _target_rs = state;
  _internal_transform = trans;

  determine_target_shader();

  // Metal has no fixed-function pipeline, so emulate it using the shader
  // generator.
  if (_target_shader->get_shader() == nullptr &&
      !_target_shader->auto_shader()) {
    if (state != _last_ff_state) {
      const ShaderAttrib *sattr;
      state->get_attrib_def(sattr);
      _last_ff_state = state;
      _last_ff_auto_state = state->set_attrib(sattr->set_shader_auto());
    }
    ensure_generated_shader(_last_ff_auto_state);
    _target_rs = _last_ff_auto_state;
    determine_target_shader();
    _target_rs = state;
  }

  Shader *shader = (Shader *)_target_shader->get_shader();
  if (_current_sc == nullptr || shader != _current_shader) {
    _current_shader = shader;
    if (shader != nullptr) {
      ShaderContext *sc = shader->prepare_now(get_prepared_objects(), this);
      _current_sc = (sc != nullptr) ? DCAST(MetalShaderContext, sc) : nullptr;
    } else {
      _current_sc = _default_sc;
    }
  }

  _state_rs = state;
}

/**
 * Called between begin_frame() and end_frame() to mark the end of drawing
 * commands for a "scene" (usually a particular DisplayRegion) within a frame.
 */
void MetalGraphicsStateGuardian::
end_scene() {
  GraphicsStateGuardian::end_scene();
  _current_sc = nullptr;
  _current_shader = nullptr;
}

/**
 * Called before a sequence of draw_primitive() functions are called, this
 * should prepare the vertex data for rendering.  It returns true if the
 * vertices are ok, false to abort this group of primitives.
 */
bool MetalGraphicsStateGuardian::
begin_draw_primitives(const GeomPipelineReader *geom_reader,
                      const GeomVertexDataPipelineReader *data_reader,
                      const InstanceList *instances, bool force) {
  if (!GraphicsStateGuardian::begin_draw_primitives(geom_reader, data_reader, instances, force)) {
    return false;
  }

  MetalShaderContext *sc = _current_sc;
  if (sc == nullptr || sc->_is_compute) {
    return false;
  }

  if (instances != nullptr) {
    if (!sc->can_inject_instancing()) {
      if (!_instanced_warning_shown) {
        _instanced_warning_shown = true;
        metaldisplay_cat.warning()
          << "Shader " << sc->_source->get_debug_name()
          << " has no model matrix inputs, so it can't be instanced.\n";
      }
      return false;
    }
    if (instances->size() == 0) {
      return false;
    }
    _instance_count = instances->size();
  } else {
    int count = _target_shader->get_instance_count();
    _instance_count = (count > 0) ? (unsigned long)count : 1;
  }
  if (sc->_is_tessellated && _instance_count != 1) {
    if (!_tess_instancing_warning_shown) {
      _tess_instancing_warning_shown = true;
      metaldisplay_cat.warning()
        << "Instancing is not supported with tessellation shaders.\n";
    }
    _instance_count = 1;
  }

  // Determine the pipeline state.
  MTLRenderPassDescriptor *pass = (MTLRenderPassDescriptor *)_pass_desc;
  MetalTexture color_tex = pass.colorAttachments[0].texture;
  MetalTexture depth_tex = pass.depthAttachment.texture;
  MetalTexture stencil_tex = pass.stencilAttachment.texture;

  const ColorAttrib *color_attrib;
  _target_rs->get_attrib_def(color_attrib);

  const ColorWriteAttrib *color_write;
  _target_rs->get_attrib_def(color_write);
  unsigned int channels = color_write->get_channels() & _color_write_mask;
  if (_target_shader->get_flag(ShaderAttrib::F_disable_alpha_write)) {
    channels &= ~ColorWriteAttrib::C_alpha;
  }

  MetalShaderContext::PipelineKey key;
  static_assert(sizeof(key) <= sizeof(_last_pipeline_key), "pipeline key too large");
  memset(&key, 0, sizeof(key));
  key._format = data_reader->get_format();
  for (int i = 0; i < _target_num_color; ++i) {
    key._color_formats[i] = (uint16_t)pass.colorAttachments[i].texture.pixelFormat;
  }
  key._depth_format = (depth_tex != nil) ? depth_tex.pixelFormat : MTLPixelFormatInvalid;
  key._stencil_format = (stencil_tex != nil) ? stencil_tex.pixelFormat : MTLPixelFormatInvalid;
  key._sample_count = (color_tex != nil) ? color_tex.sampleCount
                    : (depth_tex != nil) ? depth_tex.sampleCount : 1;
  key._blend = get_blend_key(_target_rs);
  key._color_write_mask =
    ((channels & ColorWriteAttrib::C_red) ? MTLColorWriteMaskRed : 0) |
    ((channels & ColorWriteAttrib::C_green) ? MTLColorWriteMaskGreen : 0) |
    ((channels & ColorWriteAttrib::C_blue) ? MTLColorWriteMaskBlue : 0) |
    ((channels & ColorWriteAttrib::C_alpha) ? MTLColorWriteMaskAlpha : 0);
  key._flat_color = (color_attrib->get_color_type() != ColorAttrib::T_vertex);

  key._instanced = (instances != nullptr);

  // If the vertices weren't animated on the CPU, the shader has to do it.
  key._animated = false;
  if (data_reader->get_format()->get_animation().get_animation_type() == Geom::AT_hardware &&
      !_target_shader->get_flag(ShaderAttrib::F_hardware_skinning) &&
      sc->can_inject_animation() &&
      data_reader->get_transform_table() != nullptr) {
    key._animated = true;
  }

  const TextureAttrib *texture_attrib;
  _target_rs->get_attrib_def(texture_attrib);
  sc->resolve_uv_names(texture_attrib, key);

  key._alpha_test_mode = _target_rs->get_alpha_test_mode();
  key._alpha_test_ref = 0;
  if (key._alpha_test_mode != RenderAttrib::M_none &&
      _target_shader->get_flag(ShaderAttrib::F_subsume_alpha_test)) {
    key._alpha_test_mode = RenderAttrib::M_none;
  }
  if (key._alpha_test_mode != RenderAttrib::M_none) {
    const AlphaTestAttrib *alpha_test;
    _target_rs->get_attrib_def(alpha_test);
    float ref = alpha_test->get_reference_alpha();
    memcpy(&key._alpha_test_ref, &ref, sizeof(float));
  }

  MetalRenderPipelineState pipeline;
  if (sc == _last_pipeline_sc && memcmp(&key, _last_pipeline_key, sizeof(key)) == 0) {
    pipeline = _last_pipeline;
  } else {
    pipeline = sc->get_pipeline(this, key);
    _last_pipeline_sc = sc;
    memcpy(_last_pipeline_key, &key, sizeof(key));
    _last_pipeline = pipeline;
  }
  if (pipeline == nil) {
    return false;
  }

  if (sc->_is_tessellated && !_tess_drawing) {
    // The vertex and tessellation control stages run as compute kernels in
    // draw_patches(), which has to happen before the render pass.  It calls
    // this function again to set up the render pass afterwards.
    sc->update_uniforms(this, _target_rs, _internal_transform,
                        _scene_setup->get_camera_transform(), _projection_mat);
    sc->resolve_textures(this);
    _tess_geom_reader = geom_reader;
    return true;
  }

  MetalRenderCommandEncoder encoder = get_render_encoder();
  if (encoder == nil) {
    return false;
  }
  if (pipeline != _current_pipeline) {
    [encoder setRenderPipelineState:pipeline];
    _current_pipeline = pipeline;
  }

  MetalDepthStencilState ds_state = get_depth_stencil_state(_target_rs);
  if (ds_state != _current_depth_stencil_state) {
    [encoder setDepthStencilState:ds_state];
    _current_depth_stencil_state = ds_state;
  }

  apply_dynamic_state(encoder, _target_rs);

  sc->update_uniforms(this, _target_rs, _internal_transform,
                      _scene_setup->get_camera_transform(), _projection_mat);
  sc->bind_resources(this, encoder);
  if (sc->_table_binding != nullptr) {
    bind_table_input(encoder, sc);
  }

  // Upload the skinning matrices, if we're injecting the skinning code.
  if (key._animated) {
    bind_transform_table(encoder, data_reader->get_transform_table());
  }
  if (key._instanced) {
    bind_instances(encoder, instances);
  }

  // The defaults buffer provides constant values for missing vertex columns.
  float defaults[16];
  get_vertex_defaults(defaults);
  if (!_bound_defaults_valid || memcmp(defaults, _bound_defaults, sizeof(defaults)) != 0) {
    [encoder setVertexBytes:defaults length:sizeof(defaults)
                    atIndex:MetalShaderContext::defaults_buffer_slot];
    memcpy(_bound_defaults, defaults, sizeof(defaults));
    _bound_defaults_valid = true;
  }

  if (sc->_is_tessellated) {
    // The vertex stage has already run in draw_patches().
    return true;
  }

  size_t num_arrays = data_reader->get_num_arrays();
  for (size_t i = 0; i < num_arrays; ++i) {
    MetalBuffer buffer = use_vertex_array(data_reader->get_array_reader(i), force);
    if (buffer == nil) {
      return false;
    }
    bind_vertex_buffer(encoder, MetalShaderContext::get_vertex_buffer_slot(i), buffer, 0);
  }

  return true;
}

/**
 * Makes sure the given vertex array is uploaded, and returns its buffer, or
 * nil on failure.
 */
MetalBuffer MetalGraphicsStateGuardian::
use_vertex_array(const GeomVertexArrayDataHandle *handle, bool force) {
  VertexBufferContext *vbc = handle->prepare_now(get_prepared_objects(), this);
  if (vbc == nullptr) {
    return nil;
  }
  MetalVertexBufferContext *mvbc = DCAST(MetalVertexBufferContext, vbc);
  if (!update_vertex_buffer(mvbc, handle, force)) {
    return nil;
  }
  mvbc->_last_use_frame = _frame_number;
  if (!mvbc->get_active()) {
    mvbc->set_active(true);
  }
  return mvbc->_buffer;
}

/**
 * Makes sure the index buffer of the given primitive is uploaded, and returns
 * its context, or nullptr on failure.
 */
MetalIndexBufferContext *MetalGraphicsStateGuardian::
use_index_buffer(const GeomPrimitivePipelineReader *reader, bool force) {
  IndexBufferContext *ibc = reader->prepare_now(get_prepared_objects(), this);
  if (ibc == nullptr) {
    return nullptr;
  }
  MetalIndexBufferContext *mibc = DCAST(MetalIndexBufferContext, ibc);
  if (!update_index_buffer(mibc, reader, force)) {
    return nullptr;
  }
  mibc->_last_use_frame = _frame_number;
  if (!mibc->get_active()) {
    mibc->set_active(true);
  }
  return mibc;
}

/**
 * Fills in the values that are used for vertex inputs whose column is
 * missing.  See get_default_offset() in metalShaderContext.mm for the layout.
 */
void MetalGraphicsStateGuardian::
get_vertex_defaults(float defaults[16]) const {
  static const float initial[12] = {1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0};
  static const uint32_t indices[4] = {0, 1, 2, 3};
  memcpy(defaults, initial, sizeof(initial));
  memcpy(defaults + 12, indices, sizeof(indices));

  const ColorAttrib *color_attrib;
  _target_rs->get_attrib_def(color_attrib);
  if (color_attrib->get_color_type() == ColorAttrib::T_flat) {
    LColorf color = LCAST(float, color_attrib->get_color());
    memcpy(defaults, color.get_data(), sizeof(float) * 4);
  }
}

/**
 * Clears the per-frame caches of uploaded transform tables if a new frame has
 * started, since their contents live in the transient buffer.
 */
void MetalGraphicsStateGuardian::
check_table_caches() {
  if (_transform_table_frame != _frame_number) {
    _transform_table_cache.clear();
    _table_input_cache.clear();
    _instance_cache.clear();
    _last_table_input_entry = nullptr;
    _transform_table_frame = _frame_number;
  }
}

/**
 * Uploads the matrices of the given transform table for use by the injected
 * skinning code, and binds them to buffer slot 1.  Tables are only uploaded
 * once per frame, since a character usually consists of several Geoms that
 * share the same table.
 */
void MetalGraphicsStateGuardian::
bind_transform_table(MetalRenderCommandEncoder encoder, const TransformTable *table) {
  check_table_caches();

  UpdateSeq modified = table->get_modified(Thread::get_current_thread());
  TransformTableEntry &entry = _transform_table_cache[table];
  if (entry._buffer == nil || entry._modified != modified) {
    static const size_t max_transforms = 256;
    size_t num_transforms = std::min(table->get_num_transforms(), max_transforms);

    MetalBuffer buffer;
    size_t offset;
    float *data = (float *)alloc_transient_buffer(max_transforms * 12 * sizeof(float), buffer, offset);
    nassertv(data != nullptr);

    // Each matrix is stored transposed, as three rows of four floats.
    for (size_t i = 0; i < num_transforms; ++i) {
      LMatrix4f mat;
#ifdef STDFLOAT_DOUBLE
      LMatrix4d matd;
      table->get_transform(i)->get_matrix(matd);
      mat = LCAST(float, matd);
#else
      table->get_transform(i)->get_matrix(mat);
#endif
      float *row = data + i * 12;
      row[0] = mat[0][0]; row[1] = mat[1][0]; row[2] = mat[2][0]; row[3] = mat[3][0];
      row[4] = mat[0][1]; row[5] = mat[1][1]; row[6] = mat[2][1]; row[7] = mat[3][1];
      row[8] = mat[0][2]; row[9] = mat[1][2]; row[10] = mat[2][2]; row[11] = mat[3][2];
    }
    entry._buffer = buffer;
    entry._offset = offset;
    entry._modified = modified;
  }

  bind_vertex_buffer(encoder, 2, entry._buffer, entry._offset);
}

/**
 * Releases the given object at the end of the current frame, rather than
 * right away.  This keeps it valid for any caches that might still refer to
 * it during this frame.
 */
void MetalGraphicsStateGuardian::
release_later(id object) {
  if (_autorelease_pool != nil) {
    [object autorelease];
  } else {
    [object release];
  }
}

/**
 * Forgets what has been bound to the current encoder, and sets up the state
 * that is constant for all encoders.  Called whenever a new render encoder is
 * started.
 */
void MetalGraphicsStateGuardian::
reset_bound_state() {
  ++_encoder_id;
  _bound_sc = nullptr;
  _current_pipeline = nil;
  _current_depth_stencil_state = nil;
  memset(_bound_textures, 0, sizeof(_bound_textures));
  memset(_bound_samplers, 0, sizeof(_bound_samplers));
  memset(_bound_vertex_buffers, 0, sizeof(_bound_vertex_buffers));
  memset(_bound_vertex_offsets, 0, sizeof(_bound_vertex_offsets));
  _bound_cull_mode = -1;
  _bound_fill_mode = -1;
  _bound_depth_bias_valid = false;
  _bound_stencil_ref_valid = false;
  _bound_blend_color_valid = false;
  _bound_defaults_valid = false;

  if (_render_encoder != nil) {
    // A flipped projection reverses the winding, unless the cull traverser
    // already reversed the polygon state for an inverted output.
    [_render_encoder setFrontFacingWinding:
      (_target_inverted != _target_output_inverted) ? MTLWindingClockwise
                                                     : MTLWindingCounterClockwise];
  }
}

/**
 * Binds the given texture and sampler to the given slot of both the vertex and
 * fragment stages, unless they are already bound.
 */
void MetalGraphicsStateGuardian::
bind_texture(MetalRenderCommandEncoder encoder, int slot,
             MetalTexture texture, MetalSamplerState sampler) {
  nassertv(slot >= 0 && slot < max_bound_slots);
  if (_bound_textures[slot] != texture) {
    [encoder setVertexTexture:texture atIndex:slot];
    [encoder setFragmentTexture:texture atIndex:slot];
    _bound_textures[slot] = texture;
  }
  if (_bound_samplers[slot] != sampler) {
    [encoder setVertexSamplerState:sampler atIndex:slot];
    [encoder setFragmentSamplerState:sampler atIndex:slot];
    _bound_samplers[slot] = sampler;
  }
}

/**
 * Binds the given buffer to the given vertex buffer slot, unless it is already
 * bound there.
 */
void MetalGraphicsStateGuardian::
bind_vertex_buffer(MetalRenderCommandEncoder encoder, int slot,
                   MetalBuffer buffer, size_t offset) {
  nassertv(slot >= 0 && slot < max_bound_slots);
  if (_bound_vertex_buffers[slot] == buffer) {
    if (_bound_vertex_offsets[slot] != offset) {
      [encoder setVertexBufferOffset:offset atIndex:slot];
      _bound_vertex_offsets[slot] = offset;
    }
    return;
  }
  [encoder setVertexBuffer:buffer offset:offset atIndex:slot];
  _bound_vertex_buffers[slot] = buffer;
  _bound_vertex_offsets[slot] = offset;
}

/**
 * Uploads the matrices of the given instance list, and binds them as
 * per-instance vertex data.  Each list is uploaded once per frame, since it is
 * usually shared by several Geoms.
 */
void MetalGraphicsStateGuardian::
bind_instances(MetalRenderCommandEncoder encoder, const InstanceList *instances) {
  check_table_caches();

  // InstanceList is copy-on-write, so a modified list is a different object.
  TransformTableEntry &entry = _instance_cache[instances];
  if (entry._buffer == nil) {
    size_t num_instances = instances->size();
    MetalBuffer buffer;
    size_t offset;
    float *data = (float *)alloc_transient_buffer(num_instances * 12 * sizeof(float), buffer, offset);
    nassertv(data != nullptr);

    // Each matrix is stored transposed, as three rows of four floats.
    for (size_t i = 0; i < num_instances; ++i) {
      LMatrix4f mat = LCAST(float, (*instances)[i].get_mat());
      float *row = data + i * 12;
      row[0] = mat[0][0]; row[1] = mat[1][0]; row[2] = mat[2][0]; row[3] = mat[3][0];
      row[4] = mat[0][1]; row[5] = mat[1][1]; row[6] = mat[2][1]; row[7] = mat[3][1];
      row[8] = mat[0][2]; row[9] = mat[1][2]; row[10] = mat[2][2]; row[11] = mat[3][2];
    }
    entry._buffer = buffer;
    entry._offset = offset;
  }

  bind_vertex_buffer(encoder, MetalShaderContext::instance_buffer_slot, entry._buffer, entry._offset);
}

/**
 * Binds the separate buffer holding the shader's p3d_TransformTable input.
 */
void MetalGraphicsStateGuardian::
bind_table_input(MetalRenderCommandEncoder encoder, MetalShaderContext *sc) {
  const TransformTableEntry *entry = get_table_input(sc);
  if (entry != nullptr) {
    bind_vertex_buffer(encoder, 1, entry->_buffer, entry->_offset);
  }
}

/**
 * Returns the buffer holding the shader's p3d_TransformTable input for the
 * current vertex data, filling it in if necessary.  The contents only depend
 * on the transform table, so they are computed once per table per frame.
 */
const MetalGraphicsStateGuardian::TransformTableEntry *MetalGraphicsStateGuardian::
get_table_input(MetalShaderContext *sc) {
  check_table_caches();

  const TransformTable *table = _data_reader->get_transform_table();
  UpdateSeq modified;
  if (table != nullptr) {
    modified = table->get_modified(Thread::get_current_thread());
  }

  // Consecutive draws often use the same table, so check that first.
  auto key = std::make_pair(table, (const void *)sc->_table_block_type);
  if (_last_table_input_entry != nullptr && _last_table_input_key == key &&
      _last_table_input_entry->_modified == modified) {
    return _last_table_input_entry;
  }
  TransformTableEntry &entry = _table_input_cache[key];
  _last_table_input_key = key;
  _last_table_input_entry = &entry;
  if (entry._buffer == nil || entry._modified != modified) {
    size_t size = sc->_table_block_type->get_size_bytes();
    MetalBuffer buffer;
    size_t offset;
    void *ptr = alloc_transient_buffer(size, buffer, offset);
    nassertr(ptr != nullptr, nullptr);

    // For the common case of a mat4 array, only write the matrices that the
    // table actually has; the shader won't index past them.  Tables usually
    // have far fewer transforms than the shader declares room for.
    const ShaderType *member_type = sc->_table_block_type->get_member(0).type;
    ShaderType::ScalarType scalar_type;
    uint32_t num_elements, num_rows, num_cols;
    if (table != nullptr &&
        member_type->as_scalar_type(scalar_type, num_elements, num_rows, num_cols) &&
        scalar_type == ShaderType::ST_float && num_rows == 4 && num_cols == 4) {
      LMatrix4f *matrices = (LMatrix4f *)ptr;
      size_t num_transforms = std::min((size_t)num_elements, table->get_num_transforms());
      for (size_t i = 0; i < num_transforms; ++i) {
#ifdef STDFLOAT_DOUBLE
        LMatrix4d matrix;
        table->get_transform(i)->get_matrix(matrix);
        matrices[i] = LCAST(float, matrix);
#else
        table->get_transform(i)->get_matrix(matrices[i]);
#endif
      }
    } else {
      ShaderInputBinding::State state;
      state.gsg = this;
      state.matrix_cache = nullptr;
      sc->_table_binding->fetch_data(state, ptr, false);
    }

    entry._buffer = buffer;
    entry._offset = offset;
    entry._modified = modified;
  }
  return &entry;
}

/**
 * Draws a series of patches, which requires a tessellation shader.  The
 * vertex and tessellation control stages run as compute kernels, which write
 * their outputs to buffers, after which the patches are drawn using the
 * tessellation evaluation stage as vertex function.
 */
bool MetalGraphicsStateGuardian::
draw_patches(const GeomPrimitivePipelineReader *reader, bool force) {
  MetalShaderContext *sc = _current_sc;
  if (sc == nullptr || !sc->_is_tessellated) {
    // There's no meaningful way to draw patches without tessellation.
    return true;
  }
  nassertr(_tess_geom_reader != nullptr, false);

  int num_vertices = reader->get_num_vertices();
  int patch_size = ((const GeomPatches *)reader->get_object())->get_num_vertices_per_primitive();
  int num_patches = (patch_size > 0) ? num_vertices / patch_size : 0;
  uint32_t output_vertices = sc->_tess_output_vertices;
  if (num_patches == 0 || output_vertices == 0) {
    return true;
  }

  int index_type = 0;
  MetalBuffer index_buffer = nil;
  if (reader->is_indexed()) {
    MetalIndexBufferContext *mibc = use_index_buffer(reader, force);
    if (mibc == nullptr) {
      return false;
    }
    index_type = (mibc->_index_type == MTLIndexTypeUInt16) ? 1 : 2;
    index_buffer = mibc->_buffer;
  }

  if (metaldisplay_cat.is_debug()) {
    metaldisplay_cat.debug()
      << "Drawing " << num_patches << " patches of " << patch_size << " vertices ("
      << output_vertices << " output control points, index type " << index_type << ")\n";
  }

  const MetalShaderContext::PipelineKey &key = *(const MetalShaderContext::PipelineKey *)_last_pipeline_key;
  MetalComputePipelineState vertex_pipeline = sc->get_tess_vertex_pipeline(this, key, index_type);
  MetalComputePipelineState control_pipeline = sc->get_tess_control_pipeline(this);
  if (vertex_pipeline == nil || control_pipeline == nil) {
    return false;
  }

  // The buffers that the stages write their outputs to.
  MetalBuffer vertex_out, control_out, patch_out, factors;
  size_t vertex_out_offset, control_out_offset, patch_out_offset, factors_offset;
  size_t factor_size = sc->_tess_quads ? sizeof(MTLQuadTessellationFactorsHalf)
                                       : sizeof(MTLTriangleTessellationFactorsHalf);
  if (alloc_transient_buffer(num_vertices * sc->_tess_vertex_stride, vertex_out, vertex_out_offset) == nullptr ||
      alloc_transient_buffer(num_patches * output_vertices * sc->_tess_control_stride, control_out, control_out_offset) == nullptr ||
      alloc_transient_buffer(num_patches * sc->_tess_patch_stride, patch_out, patch_out_offset) == nullptr ||
      alloc_transient_buffer(num_patches * factor_size, factors, factors_offset) == nullptr) {
    return false;
  }

  // Compute work can't happen during a render pass.
  end_render_pass();
  MetalComputeCommandEncoder encoder = [_cmd computeCommandEncoder];
  if (metal_debug_labels) {
    encoder.label = @"Panda3D tessellation";
  }
  sc->bind_compute_resources(this, encoder);
  if (sc->_table_binding != nullptr) {
    const TransformTableEntry *entry = get_table_input(sc);
    if (entry != nullptr) {
      [encoder setBuffer:entry->_buffer offset:entry->_offset atIndex:1];
    }
  }

  // Run the vertex stage over all the vertices of the patches, in order.
  [encoder setComputePipelineState:vertex_pipeline];
  const GeomVertexDataPipelineReader *data_reader = _data_reader;
  int first_vertex = reader->is_indexed() ? 0 : reader->get_first_vertex();
  for (size_t i = 0; i < data_reader->get_num_arrays(); ++i) {
    CPT(GeomVertexArrayDataHandle) handle = data_reader->get_array_reader(i);
    MetalBuffer buffer = use_vertex_array(handle, force);
    if (buffer == nil) {
      [encoder endEncoding];
      return false;
    }
    // For non-indexed drawing, the grid always starts at vertex 0.
    const GeomVertexArrayFormat *array_format = handle->get_array_format();
    size_t offset = (array_format->get_divisor() == 0) ? (size_t)first_vertex * array_format->get_stride() : 0;
    [encoder setBuffer:buffer offset:offset
               atIndex:MetalShaderContext::get_tess_vertex_buffer_slot(i)];
  }
  float defaults[16];
  get_vertex_defaults(defaults);
  [encoder setBytes:defaults length:sizeof(defaults)
            atIndex:MetalShaderContext::tess_defaults_buffer_slot];
  if (index_buffer != nil) {
    [encoder setBuffer:index_buffer offset:0 atIndex:MetalShaderContext::tess_index_buffer_slot];
  }
  [encoder setBuffer:vertex_out offset:vertex_out_offset atIndex:28];
  // The range of vertices (or indices) that the stage input is fetched for.
  [encoder setStageInRegion:MTLRegionMake2D(0, 0, num_vertices, 1)];
  NSUInteger width = std::min((NSUInteger)64, vertex_pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(num_vertices, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];

  // Run the tessellation control stage once for each output control point.
  [encoder setComputePipelineState:control_pipeline];
  [encoder setBuffer:vertex_out offset:vertex_out_offset atIndex:22];
  [encoder setBuffer:control_out offset:control_out_offset atIndex:28];
  [encoder setBuffer:patch_out offset:patch_out_offset atIndex:27];
  [encoder setBuffer:factors offset:factors_offset atIndex:26];
  uint32_t params[2] = {(uint32_t)patch_size, (uint32_t)num_patches};
  [encoder setBytes:params length:sizeof(params) atIndex:29];
  width = std::min((NSUInteger)(output_vertices * std::max(1u, 64u / output_vertices)),
                   control_pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(num_patches * output_vertices, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];

  // Now set up the render pass and the draw state.
  _tess_drawing = true;
  bool success = begin_draw_primitives(_tess_geom_reader, data_reader, nullptr, force);
  _tess_drawing = false;
  if (!success) {
    return false;
  }
  MetalRenderCommandEncoder render_encoder = _render_encoder;
  bind_vertex_buffer(render_encoder, 22, control_out, control_out_offset);
  bind_vertex_buffer(render_encoder, 20, patch_out, patch_out_offset);
  [render_encoder setTessellationFactorBuffer:factors offset:factors_offset instanceStride:0];
  [render_encoder drawPatches:output_vertices
                   patchStart:0
                   patchCount:num_patches
             patchIndexBuffer:nil
       patchIndexBufferOffset:0
                instanceCount:1
                 baseInstance:0];
  return true;
}

/**
 * Draws a series of disconnected triangles.
 */
bool MetalGraphicsStateGuardian::
draw_triangles(const GeomPrimitivePipelineReader *reader, bool force) {
  return do_draw_primitive(reader, force, MTLPrimitiveTypeTriangle);
}

/**
 * Draws a series of triangle strips.
 */
bool MetalGraphicsStateGuardian::
draw_tristrips(const GeomPrimitivePipelineReader *reader, bool force) {
  return do_draw_primitive(reader, force, MTLPrimitiveTypeTriangleStrip);
}

/**
 * Draws a series of disconnected line segments.
 */
bool MetalGraphicsStateGuardian::
draw_lines(const GeomPrimitivePipelineReader *reader, bool force) {
  return do_draw_primitive(reader, force, MTLPrimitiveTypeLine);
}

/**
 * Draws a series of line strips.
 */
bool MetalGraphicsStateGuardian::
draw_linestrips(const GeomPrimitivePipelineReader *reader, bool force) {
  return do_draw_primitive(reader, force, MTLPrimitiveTypeLineStrip);
}

/**
 * Draws a series of disconnected points.
 */
bool MetalGraphicsStateGuardian::
draw_points(const GeomPrimitivePipelineReader *reader, bool force) {
  return do_draw_primitive(reader, force, MTLPrimitiveTypePoint);
}

/**
 * Called after a sequence of draw_primitive() functions are called, this
 * should do whatever cleanup is appropriate.
 */
void MetalGraphicsStateGuardian::
end_draw_primitives() {
  GraphicsStateGuardian::end_draw_primitives();
}

/**
 * Invoked by all the draw_xyz methods.
 */
bool MetalGraphicsStateGuardian::
do_draw_primitive(const GeomPrimitivePipelineReader *reader, bool force,
                  unsigned long primitive_type) {
  MetalRenderCommandEncoder encoder = _render_encoder;
  nassertr(encoder != nil, false);

  int num_vertices = reader->get_num_vertices();
  if (num_vertices == 0) {
    return true;
  }

  if (reader->is_indexed()) {
    MetalIndexBufferContext *mibc = use_index_buffer(reader, force);
    if (mibc == nullptr) {
      return false;
    }

    [encoder drawIndexedPrimitives:(MTLPrimitiveType)primitive_type
                        indexCount:num_vertices
                         indexType:(MTLIndexType)mibc->_index_type
                       indexBuffer:mibc->_buffer
                 indexBufferOffset:0
                     instanceCount:_instance_count];
  } else {
    [encoder drawPrimitives:(MTLPrimitiveType)primitive_type
                vertexStart:reader->get_first_vertex()
                vertexCount:num_vertices
              instanceCount:_instance_count];
  }
  return true;
}

/**
 * Converts a Panda blend mode to a Metal one.
 */
static MTLBlendOperation
get_blend_operation(ColorBlendAttrib::Mode mode) {
  switch (mode) {
  case ColorBlendAttrib::M_subtract: return MTLBlendOperationSubtract;
  case ColorBlendAttrib::M_inv_subtract: return MTLBlendOperationReverseSubtract;
  case ColorBlendAttrib::M_min: return MTLBlendOperationMin;
  case ColorBlendAttrib::M_max: return MTLBlendOperationMax;
  default: return MTLBlendOperationAdd;
  }
}

/**
 * Converts a Panda blend operand to a Metal one.  The color scale operands
 * are implemented using the blend constant color.
 */
static MTLBlendFactor
get_blend_factor(ColorBlendAttrib::Operand operand) {
  switch (operand) {
  case ColorBlendAttrib::O_zero: return MTLBlendFactorZero;
  case ColorBlendAttrib::O_one: return MTLBlendFactorOne;
  case ColorBlendAttrib::O_incoming_color: return MTLBlendFactorSourceColor;
  case ColorBlendAttrib::O_one_minus_incoming_color: return MTLBlendFactorOneMinusSourceColor;
  case ColorBlendAttrib::O_fbuffer_color: return MTLBlendFactorDestinationColor;
  case ColorBlendAttrib::O_one_minus_fbuffer_color: return MTLBlendFactorOneMinusDestinationColor;
  case ColorBlendAttrib::O_incoming_alpha: return MTLBlendFactorSourceAlpha;
  case ColorBlendAttrib::O_one_minus_incoming_alpha: return MTLBlendFactorOneMinusSourceAlpha;
  case ColorBlendAttrib::O_fbuffer_alpha: return MTLBlendFactorDestinationAlpha;
  case ColorBlendAttrib::O_one_minus_fbuffer_alpha: return MTLBlendFactorOneMinusDestinationAlpha;
  case ColorBlendAttrib::O_constant_color:
  case ColorBlendAttrib::O_color_scale: return MTLBlendFactorBlendColor;
  case ColorBlendAttrib::O_one_minus_constant_color:
  case ColorBlendAttrib::O_one_minus_color_scale: return MTLBlendFactorOneMinusBlendColor;
  case ColorBlendAttrib::O_constant_alpha:
  case ColorBlendAttrib::O_alpha_scale: return MTLBlendFactorBlendAlpha;
  case ColorBlendAttrib::O_one_minus_constant_alpha:
  case ColorBlendAttrib::O_one_minus_alpha_scale: return MTLBlendFactorOneMinusBlendAlpha;
  case ColorBlendAttrib::O_incoming_color_saturate: return MTLBlendFactorSourceAlphaSaturated;
  case ColorBlendAttrib::O_incoming1_color: return MTLBlendFactorSource1Color;
  case ColorBlendAttrib::O_one_minus_incoming1_color: return MTLBlendFactorOneMinusSource1Color;
  case ColorBlendAttrib::O_incoming1_alpha: return MTLBlendFactorSource1Alpha;
  case ColorBlendAttrib::O_one_minus_incoming1_alpha: return MTLBlendFactorOneMinusSource1Alpha;
  default: return MTLBlendFactorOne;
  }
}

/**
 * Packs the blending state of the given render state into a key.
 *  bit 0: enabled, bits 1-3: color op, 4-6: alpha op, 7-11: src color,
 *  12-16: dest color, 17-21: src alpha, 22-26: dest alpha.
 */
uint32_t MetalGraphicsStateGuardian::
get_blend_key(const RenderState *state) const {
  auto pack = [](int color_op, int alpha_op, int src_c, int dst_c, int src_a, int dst_a) {
    return 1u | (color_op << 1) | (alpha_op << 4) | (src_c << 7) |
           (dst_c << 12) | (src_a << 17) | ((uint32_t)dst_a << 22);
  };

  const ColorBlendAttrib *color_blend;
  state->get_attrib_def(color_blend);
  if (color_blend->get_mode() != ColorBlendAttrib::M_none) {
    return pack(color_blend->get_mode(), color_blend->get_alpha_mode(),
                color_blend->get_operand_a(), color_blend->get_operand_b(),
                color_blend->get_alpha_operand_a(), color_blend->get_alpha_operand_b());
  }

  const TransparencyAttrib *transparency;
  state->get_attrib_def(transparency);
  switch (transparency->get_mode()) {
  case TransparencyAttrib::M_alpha:
  case TransparencyAttrib::M_dual:
    if (old_alpha_blend) {
      return pack(ColorBlendAttrib::M_add, ColorBlendAttrib::M_add,
                  ColorBlendAttrib::O_incoming_alpha, ColorBlendAttrib::O_one_minus_incoming_alpha,
                  ColorBlendAttrib::O_incoming_alpha, ColorBlendAttrib::O_one_minus_incoming_alpha);
    } else {
      return pack(ColorBlendAttrib::M_add, ColorBlendAttrib::M_add,
                  ColorBlendAttrib::O_incoming_alpha, ColorBlendAttrib::O_one_minus_incoming_alpha,
                  ColorBlendAttrib::O_one, ColorBlendAttrib::O_one_minus_incoming_alpha);
    }

  case TransparencyAttrib::M_premultiplied_alpha:
    return pack(ColorBlendAttrib::M_add, ColorBlendAttrib::M_add,
                ColorBlendAttrib::O_one, ColorBlendAttrib::O_one_minus_incoming_alpha,
                ColorBlendAttrib::O_one, ColorBlendAttrib::O_one_minus_incoming_alpha);

  case TransparencyAttrib::M_multisample:
  case TransparencyAttrib::M_multisample_mask:
    // No blending, but alpha-to-coverage (bit 27) if multisampling.
    return (1u << 27);

  default:
    return 0;
  }
}

/**
 * Applies a key returned by get_blend_key() to the given color attachment
 * descriptor.
 */
void MetalGraphicsStateGuardian::
apply_blend_key(id color_attachment, uint32_t key) const {
  MTLRenderPipelineColorAttachmentDescriptor *desc = color_attachment;
  if ((key & 1) == 0) {
    desc.blendingEnabled = NO;
    return;
  }
  desc.blendingEnabled = YES;
  desc.rgbBlendOperation = get_blend_operation((ColorBlendAttrib::Mode)((key >> 1) & 7));
  desc.alphaBlendOperation = get_blend_operation((ColorBlendAttrib::Mode)((key >> 4) & 7));
  desc.sourceRGBBlendFactor = get_blend_factor((ColorBlendAttrib::Operand)((key >> 7) & 31));
  desc.destinationRGBBlendFactor = get_blend_factor((ColorBlendAttrib::Operand)((key >> 12) & 31));
  desc.sourceAlphaBlendFactor = get_blend_factor((ColorBlendAttrib::Operand)((key >> 17) & 31));
  desc.destinationAlphaBlendFactor = get_blend_factor((ColorBlendAttrib::Operand)((key >> 22) & 31));
}

/**
 * Converts a Panda comparison function to a Metal one.
 */
static MTLCompareFunction
get_compare_function(RenderAttrib::PandaCompareFunc func) {
  switch (func) {
  case RenderAttrib::M_none: return MTLCompareFunctionAlways;
  case RenderAttrib::M_never: return MTLCompareFunctionNever;
  case RenderAttrib::M_less: return MTLCompareFunctionLess;
  case RenderAttrib::M_equal: return MTLCompareFunctionEqual;
  case RenderAttrib::M_less_equal: return MTLCompareFunctionLessEqual;
  case RenderAttrib::M_greater: return MTLCompareFunctionGreater;
  case RenderAttrib::M_not_equal: return MTLCompareFunctionNotEqual;
  case RenderAttrib::M_greater_equal: return MTLCompareFunctionGreaterEqual;
  case RenderAttrib::M_always: return MTLCompareFunctionAlways;
  default: return MTLCompareFunctionAlways;
  }
}

/**
 * Converts a Panda stencil operation to a Metal one.
 */
static MTLStencilOperation
get_stencil_operation(unsigned int op) {
  switch (op) {
  case StencilAttrib::SO_zero: return MTLStencilOperationZero;
  case StencilAttrib::SO_replace: return MTLStencilOperationReplace;
  case StencilAttrib::SO_increment: return MTLStencilOperationIncrementWrap;
  case StencilAttrib::SO_decrement: return MTLStencilOperationDecrementWrap;
  case StencilAttrib::SO_invert: return MTLStencilOperationInvert;
  case StencilAttrib::SO_increment_saturate: return MTLStencilOperationIncrementClamp;
  case StencilAttrib::SO_decrement_saturate: return MTLStencilOperationDecrementClamp;
  default: return MTLStencilOperationKeep;
  }
}

/**
 * Returns the depth/stencil state object for the given render state.
 */
MetalDepthStencilState MetalGraphicsStateGuardian::
get_depth_stencil_state(const RenderState *state) {
  const DepthTestAttrib *depth_test;
  state->get_attrib_def(depth_test);
  const DepthWriteAttrib *depth_write;
  state->get_attrib_def(depth_write);
  const StencilAttrib *stencil = nullptr;
  state->get_attrib(stencil);

  DepthStencilKey key;
  key._depth_func = _target_has_depth ? depth_test->get_mode() : RenderAttrib::M_none;
  key._depth_write = _target_has_depth && depth_write->get_mode() == DepthWriteAttrib::M_on;
  key._stencil = _target_has_stencil ? stencil : nullptr;

  auto it = _depth_stencil_states.find(key);
  if (it != _depth_stencil_states.end()) {
    return it->second;
  }

  MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];
  desc.depthCompareFunction = get_compare_function((RenderAttrib::PandaCompareFunc)key._depth_func);
  desc.depthWriteEnabled = key._depth_write;

  if (key._stencil != nullptr) {
    const StencilAttrib *sa = key._stencil;
    {
      MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
      front.stencilCompareFunction = get_compare_function(
        (RenderAttrib::PandaCompareFunc)sa->get_render_state(StencilAttrib::SRS_front_comparison_function));
      front.stencilFailureOperation = get_stencil_operation(
        sa->get_render_state(StencilAttrib::SRS_front_stencil_fail_operation));
      front.depthFailureOperation = get_stencil_operation(
        sa->get_render_state(StencilAttrib::SRS_front_stencil_pass_z_fail_operation));
      front.depthStencilPassOperation = get_stencil_operation(
        sa->get_render_state(StencilAttrib::SRS_front_stencil_pass_z_pass_operation));
      front.readMask = sa->get_render_state(StencilAttrib::SRS_read_mask);
      front.writeMask = sa->get_render_state(StencilAttrib::SRS_write_mask);
      desc.frontFaceStencil = front;

      if (sa->get_render_state(StencilAttrib::SRS_back_comparison_function) != RenderAttrib::M_none) {
        MTLStencilDescriptor *back = [[MTLStencilDescriptor alloc] init];
        back.stencilCompareFunction = get_compare_function(
          (RenderAttrib::PandaCompareFunc)sa->get_render_state(StencilAttrib::SRS_back_comparison_function));
        back.stencilFailureOperation = get_stencil_operation(
          sa->get_render_state(StencilAttrib::SRS_back_stencil_fail_operation));
        back.depthFailureOperation = get_stencil_operation(
          sa->get_render_state(StencilAttrib::SRS_back_stencil_pass_z_fail_operation));
        back.depthStencilPassOperation = get_stencil_operation(
          sa->get_render_state(StencilAttrib::SRS_back_stencil_pass_z_pass_operation));
        back.readMask = front.readMask;
        back.writeMask = front.writeMask;
        desc.backFaceStencil = back;
        [back release];
      } else {
        desc.backFaceStencil = front;
      }
      [front release];
    }
  }

  MetalDepthStencilState ds_state = [_device newDepthStencilStateWithDescriptor:desc];
  [desc release];
  _depth_stencil_states[key] = ds_state;
  return ds_state;
}

/**
 * Applies the state that is not baked into the pipeline state object.
 */
void MetalGraphicsStateGuardian::
apply_dynamic_state(MetalRenderCommandEncoder encoder, const RenderState *state) {
  // Panda considers counter-clockwise triangles to be front-facing; the
  // winding is set when the encoder is created.
  const CullFaceAttrib *cull_face;
  state->get_attrib_def(cull_face);
  int cull_mode;
  switch (cull_face->get_effective_mode()) {
  case CullFaceAttrib::M_cull_clockwise:
    cull_mode = MTLCullModeBack;
    break;
  case CullFaceAttrib::M_cull_counter_clockwise:
    cull_mode = MTLCullModeFront;
    break;
  default:
    cull_mode = MTLCullModeNone;
    break;
  }
  if (cull_mode != _bound_cull_mode) {
    [encoder setCullMode:(MTLCullMode)cull_mode];
    _bound_cull_mode = cull_mode;
  }

  const RenderModeAttrib *render_mode;
  state->get_attrib_def(render_mode);
  int fill_mode = (render_mode->get_mode() == RenderModeAttrib::M_wireframe)
                ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
  if (fill_mode != _bound_fill_mode) {
    [encoder setTriangleFillMode:(MTLTriangleFillMode)fill_mode];
    _bound_fill_mode = fill_mode;
  }

  const DepthOffsetAttrib *depth_offset;
  state->get_attrib_def(depth_offset);
  int offset = depth_offset->get_offset();

  // The DepthOffsetAttrib may narrow the depth range of the display region.
  double near_depth = _viewport[4];
  double far_depth = _viewport[5];
  PN_stdfloat min_value = depth_offset->get_min_value();
  PN_stdfloat max_value = depth_offset->get_max_value();
  if (min_value != 0 || max_value != 1) {
    double range = far_depth - near_depth;
    far_depth = near_depth + range * max_value;
    near_depth = near_depth + range * min_value;
  }
  if (near_depth != _applied_depth_range[0] || far_depth != _applied_depth_range[1]) {
    MTLViewport viewport = {_viewport[0], _viewport[1], _viewport[2],
                            _viewport[3], near_depth, far_depth};
    [encoder setViewport:viewport];
    _applied_depth_range[0] = near_depth;
    _applied_depth_range[1] = far_depth;
  }
  float bias[3];
  const DepthBiasAttrib *depth_bias;
  if (state->get_attrib(depth_bias)) {
    bias[0] = depth_bias->get_constant_factor() - offset;
    bias[1] = depth_bias->get_slope_factor() - offset;
    bias[2] = depth_bias->get_clamp();
  } else {
    bias[0] = -offset;
    bias[1] = -offset;
    bias[2] = 0;
  }
  if (!_bound_depth_bias_valid || memcmp(bias, _bound_depth_bias, sizeof(bias)) != 0) {
    [encoder setDepthBias:bias[0] slopeScale:bias[1] clamp:bias[2]];
    memcpy(_bound_depth_bias, bias, sizeof(bias));
    _bound_depth_bias_valid = true;
  }

  const StencilAttrib *stencil;
  if (_target_has_stencil && state->get_attrib(stencil)) {
    uint32_t ref = stencil->get_render_state(StencilAttrib::SRS_reference);
    if (!_bound_stencil_ref_valid || ref != _bound_stencil_ref) {
      [encoder setStencilReferenceValue:ref];
      _bound_stencil_ref = ref;
      _bound_stencil_ref_valid = true;
    }
  }

  const ColorBlendAttrib *color_blend;
  state->get_attrib_def(color_blend);
  if (color_blend->get_mode() != ColorBlendAttrib::M_none) {
    LColor c = color_blend->involves_color_scale() ? _current_color_scale : color_blend->get_color();
    if (!_bound_blend_color_valid || c != _bound_blend_color) {
      [encoder setBlendColorRed:c[0] green:c[1] blue:c[2] alpha:c[3]];
      _bound_blend_color = c;
      _bound_blend_color_valid = true;
    }
  }
}
