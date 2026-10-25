/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsBuffer.mm
 * @author Jace Mayer
 * @date 2026-10-21
 */

#include "metalGraphicsBuffer.h"
#include "metalGraphicsPipe.h"
#include "metalGraphicsStateGuardian.h"
#include "config_metaldisplay.h"

TypeHandle MetalGraphicsBuffer::_type_handle;

/**
 * Sets up the given texture so that its Metal texture will have the given
 * pixel format, which must be one that we use for render targets.  Returns
 * false if the format has no Panda equivalent.
 */
static bool
setup_texture_format(Texture *tex, unsigned long format) {
  switch ((MTLPixelFormat)format) {
  case MTLPixelFormatBGRA8Unorm:
    tex->set_format(Texture::F_rgba);
    tex->set_component_type(Texture::T_unsigned_byte);
    return true;
  case MTLPixelFormatBGRA8Unorm_sRGB:
    tex->set_format(Texture::F_srgb_alpha);
    tex->set_component_type(Texture::T_unsigned_byte);
    return true;
  case MTLPixelFormatRGBA16Float:
    tex->set_format(Texture::F_rgba16);
    tex->set_component_type(Texture::T_half_float);
    return true;
  case MTLPixelFormatRGBA32Float:
    tex->set_format(Texture::F_rgba32);
    tex->set_component_type(Texture::T_float);
    return true;
  case MTLPixelFormatDepth32Float:
    tex->set_format(Texture::F_depth_component32);
    tex->set_component_type(Texture::T_float);
    return true;
  case MTLPixelFormatDepth32Float_Stencil8:
    tex->set_format(Texture::F_depth_stencil);
    tex->set_component_type(Texture::T_float);
    return true;
  default:
    return false;
  }
}

/**
 *
 */
MetalGraphicsBuffer::
MetalGraphicsBuffer(GraphicsEngine *engine, GraphicsPipe *pipe,
                    std::string name,
                    const FrameBufferProperties &fb_prop,
                    const WindowProperties &win_prop,
                    int flags,
                    GraphicsStateGuardian *gsg,
                    GraphicsOutput *host) :
  GraphicsBuffer(engine, pipe, std::move(name), fb_prop, win_prop, flags, gsg, host),
  _attachment_size(0, 0)
{
}

/**
 *
 */
MetalGraphicsBuffer::
~MetalGraphicsBuffer() {
  release_attachments();
}

/**
 * Returns true, since we can always render directly into a texture.
 */
bool MetalGraphicsBuffer::
get_supports_render_texture() const {
  return true;
}

/**
 * Called internally when the window is in render-to-a-texture mode and we are
 * in the process of rendering the six faces of a cube map, or any other
 * multi-page texture.  This should do whatever needs to be done to switch the
 * buffer to the indicated page.
 */
void MetalGraphicsBuffer::
select_target_tex_page(int page) {
  if (page != _page) {
    _page = page;
    set_render_target();
  }
}

/**
 * This function will be called within the draw thread before beginning
 * rendering for a given frame.  It should do whatever setup is required, and
 * return true if the frame should be rendered, or false if it should be
 * skipped.
 */
bool MetalGraphicsBuffer::
begin_frame(FrameMode mode, Thread *current_thread) {
  begin_frame_spam(mode);
  if (_gsg == nullptr) {
    return false;
  }

  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_R(mgsg, _gsg, false);
  if (!mgsg->is_valid()) {
    return false;
  }

  if (mode == FM_render) {
    clear_cube_map_selection();

    // If the set of render textures has changed, or we have to follow the
    // size of the host window, the bitplanes need to be rebuilt.
    {
      CDReader cdata(_cycler);
      if (cdata->_textures_seq != _last_textures_seq) {
        _last_textures_seq = cdata->_textures_seq;
        _needs_rebuild = true;
      }
    }
    if ((_creation_flags & GraphicsPipe::BF_size_track_host) != 0 &&
        _host != nullptr && _host->get_size() != _size) {
      set_size_and_recalc(_host->get_x_size(), _host->get_y_size());
    }
  }
  else if (_attachment_size == LVecBase2i(0, 0)) {
    // There's nothing to refresh.
    return false;
  }

  if (_size[0] <= 0 || _size[1] <= 0) {
    return false;
  }

  if (_attachment_size != _size) {
    _needs_rebuild = true;
  }

  if (!mgsg->begin_frame(current_thread)) {
    return false;
  }

  if (_needs_rebuild && !rebuild_bitplanes()) {
    mgsg->end_frame(current_thread);
    return false;
  }

  _page = 0;
  set_render_target();
  return true;
}

/**
 * This function will be called within the draw thread after rendering is
 * completed for a given frame.  It should do whatever finalization is
 * required.
 */
void MetalGraphicsBuffer::
end_frame(FrameMode mode, Thread *current_thread) {
  end_frame_spam(mode);

  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_V(mgsg, _gsg);

  mgsg->end_render_pass();

  if (mode == FM_render) {
    copy_to_textures();

    // Bound textures that use mipmaps need them regenerated.
    MetalBlitCommandEncoder blit = nil;
    auto generate_mipmaps = [&](const Bitplane &plane) {
      if (plane._bound != nullptr && plane._bound->uses_mipmaps()) {
        MetalTexture texture = mgsg->use_texture(plane._bound, 0);
        if (texture != nil && texture.mipmapLevelCount > 1) {
          if (blit == nil) {
            blit = [mgsg->get_command_buffer() blitCommandEncoder];
          }
          [blit generateMipmapsForTexture:texture];
        }
      }
    };
    for (const Bitplane &plane : _color_planes) {
      generate_mipmaps(plane);
    }
    generate_mipmaps(_depth_plane);
    if (blit != nil) {
      [blit endEncoding];
    }
  }

  mgsg->end_frame(current_thread);

  if (mode == FM_render) {
    trigger_flip();
    clear_cube_map_selection();
  }
}

/**
 * Closes the buffer right now.  Called from the window thread.
 */
void MetalGraphicsBuffer::
close_buffer() {
  if (_gsg != nullptr) {
    MetalGraphicsStateGuardian *mgsg;
    DCAST_INTO_V(mgsg, _gsg);
    mgsg->finish();
  }
  release_attachments();
  _gsg.clear();
  _is_valid = false;
}

/**
 * Opens the buffer right now.  Called from the window thread.  Returns true
 * if the buffer is successfully opened, or false if there was a problem.
 */
bool MetalGraphicsBuffer::
open_buffer() {
  MetalGraphicsPipe *mpipe;
  DCAST_INTO_R(mpipe, _pipe, false);

  MetalGraphicsStateGuardian *mgsg;
  if (_gsg == nullptr) {
    mgsg = new MetalGraphicsStateGuardian(_engine, mpipe, nullptr);
    _gsg = mgsg;
  } else {
    DCAST_INTO_R(mgsg, _gsg, false);
    if (mgsg->get_engine() != _engine) {
      mgsg = new MetalGraphicsStateGuardian(_engine, mpipe, mgsg);
      _gsg = mgsg;
    }
  }

  mgsg->reset_if_new();
  if (!mgsg->is_valid()) {
    _gsg.clear();
    return false;
  }

  if (!choose_formats()) {
    _gsg.clear();
    return false;
  }

  _needs_rebuild = true;
  _is_valid = true;
  return true;
}

/**
 * Determines the pixel formats to use based on the requested framebuffer
 * properties, and updates _fb_properties to reflect what we actually got.
 */
bool MetalGraphicsBuffer::
choose_formats() {
  FrameBufferProperties props;

  if (_fb_properties.get_color_bits() > 0 || _fb_properties.get_alpha_bits() > 0 ||
      _fb_properties.get_depth_bits() == 0) {
    if (_fb_properties.get_float_color()) {
      if (_fb_properties.get_red_bits() > 16 || _fb_properties.get_color_bits() > 48) {
        _color_format = MTLPixelFormatRGBA32Float;
        props.set_rgba_bits(32, 32, 32, 32);
      } else {
        _color_format = MTLPixelFormatRGBA16Float;
        props.set_rgba_bits(16, 16, 16, 16);
      }
      props.set_float_color(true);
    }
    else if (_fb_properties.get_srgb_color()) {
      _color_format = MTLPixelFormatBGRA8Unorm_sRGB;
      props.set_rgba_bits(8, 8, 8, 8);
      props.set_srgb_color(true);
    }
    else {
      _color_format = MTLPixelFormatBGRA8Unorm;
      props.set_rgba_bits(8, 8, 8, 8);
    }
  } else {
    _color_format = MTLPixelFormatInvalid;
  }

  if (_fb_properties.get_stencil_bits() > 0) {
    _depth_stencil_format = MTLPixelFormatDepth32Float_Stencil8;
    props.set_depth_bits(32);
    props.set_stencil_bits(8);
    props.set_float_depth(true);
  }
  else if (_fb_properties.get_depth_bits() > 0) {
    _depth_stencil_format = MTLPixelFormatDepth32Float;
    props.set_depth_bits(32);
    props.set_float_depth(true);
  }
  else {
    _depth_stencil_format = MTLPixelFormatInvalid;
  }

  if (_color_format != MTLPixelFormatInvalid) {
    props.set_rgb_color(true);
  }

  // Aux bitplanes are extra color attachments, of which Metal allows eight in
  // total, including the main color buffer.
  int available = MetalGraphicsStateGuardian::RenderTarget::max_color_attachments
                - (_color_format != MTLPixelFormatInvalid ? 1 : 0);
  int aux_rgba = std::min(_fb_properties.get_aux_rgba(), std::min(available, 4));
  available -= aux_rgba;
  int aux_hrgba = std::min(_fb_properties.get_aux_hrgba(), std::min(available, 4));
  available -= aux_hrgba;
  int aux_float = std::min(_fb_properties.get_aux_float(), std::min(available, 4));
  props.set_aux_rgba(aux_rgba);
  props.set_aux_hrgba(aux_hrgba);
  props.set_aux_float(aux_float);

  // Multisampling.
  MetalGraphicsStateGuardian *mgsg = DCAST(MetalGraphicsStateGuardian, _gsg);
  _sample_count = 1;
  if (_fb_properties.get_multisamples() > 1) {
    _sample_count = mgsg->choose_sample_count(_fb_properties.get_multisamples());
    if (_sample_count > 1) {
      props.set_multisamples(_sample_count);
    }
  }

  props.set_force_hardware(true);
  _fb_properties = props;
  return true;
}

/**
 * Prepares the given texture for being rendered into directly, with the
 * given pixel format and the size of the buffer.  Returns false if it can't
 * be used as a render target.
 */
bool MetalGraphicsBuffer::
bind_texture(Texture *tex, unsigned long format) {
  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_R(mgsg, _gsg, false);

  switch (tex->get_texture_type()) {
  case Texture::TT_2d_texture:
    if (tex->get_x_size() != _size[0] || tex->get_y_size() != _size[1]) {
      tex->set_x_size(_size[0]);
      tex->set_y_size(_size[1]);
    }
    tex->set_pad_size(0, 0);
    break;

  case Texture::TT_cube_map:
  case Texture::TT_cube_map_array:
  case Texture::TT_2d_texture_array:
  case Texture::TT_3d_texture:
    // We render into one page at a time; the size has to match.
    if (tex->get_x_size() != _size[0] || tex->get_y_size() != _size[1]) {
      return false;
    }
    break;

  default:
    return false;
  }

  if (!setup_texture_format(tex, format)) {
    return false;
  }
  if (tex->has_ram_image()) {
    // The contents will live on the GPU from now on.
    tex->clear_ram_image();
  }
  if (!tex->has_clear_color()) {
    tex->set_clear_color(LColor(0, 0, 0, 1));
  }

  MetalTexture texture = mgsg->use_texture(tex, 0);
  return texture != nil && texture.pixelFormat == (MTLPixelFormat)format &&
         (texture.usage & MTLTextureUsageRenderTarget) != 0;
}

/**
 * Determines which textures are rendered into directly, and (re)creates the
 * textures that we render into ourselves.  Must be called between
 * begin_frame() and end_frame() on the GSG.
 */
bool MetalGraphicsBuffer::
rebuild_bitplanes() {
  release_attachments();

  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_R(mgsg, _gsg, false);
  MetalDevice device = mgsg->get_device();

  // Decide on the bitplanes that we have.
  if (_color_format != MTLPixelFormatInvalid) {
    Bitplane plane;
    plane._plane = RTP_color;
    plane._format = _color_format;
    _color_planes.push_back(plane);
  }
  for (int i = 0; i < _fb_properties.get_aux_rgba(); ++i) {
    Bitplane plane;
    plane._plane = RTP_aux_rgba_0 + i;
    plane._format = MTLPixelFormatBGRA8Unorm;
    _color_planes.push_back(plane);
  }
  for (int i = 0; i < _fb_properties.get_aux_hrgba(); ++i) {
    Bitplane plane;
    plane._plane = RTP_aux_hrgba_0 + i;
    plane._format = MTLPixelFormatRGBA16Float;
    _color_planes.push_back(plane);
  }
  for (int i = 0; i < _fb_properties.get_aux_float(); ++i) {
    Bitplane plane;
    plane._plane = RTP_aux_float_0 + i;
    plane._format = MTLPixelFormatRGBA32Float;
    _color_planes.push_back(plane);
  }

  bool want_stencil = (_depth_stencil_format == MTLPixelFormatDepth32Float_Stencil8);
  _depth_plane._plane = RTP_depth_stencil;
  _depth_plane._format = _depth_stencil_format;

  // Assign the textures to be bound.  Ones that we can't render into
  // directly fall back to being copied.
  bool need_depth_copy = false;
  {
    CDReader cdata(_cycler);
    for (size_t i = 0; i < cdata->_textures.size(); ++i) {
      const RenderTexture &rt = cdata->_textures[i];
      RenderTexturePlane plane = rt._plane;
      Texture *tex = rt._texture;

      if (plane == RTP_depth || plane == RTP_depth_stencil) {
        need_depth_copy = true;
      }
      if (rt._rtm_mode != RTM_bind_or_copy && rt._rtm_mode != RTM_bind_layered) {
        continue;
      }

      bool bound = false;
      if (plane == RTP_depth || plane == RTP_depth_stencil) {
        if (_depth_stencil_format != MTLPixelFormatInvalid && _depth_plane._bound == nullptr) {
          // A texture bound to RTP_depth only gets the depth; the stencil
          // goes into a separate texture of our own.
          unsigned long format = (plane == RTP_depth_stencil)
            ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
          if (bind_texture(tex, format)) {
            _depth_plane._bound = tex;
            _depth_plane._plane = plane;
            _depth_plane._format = format;
            bound = true;
          }
        }
      } else {
        for (Bitplane &bp : _color_planes) {
          if (bp._plane == plane && bp._bound == nullptr) {
            if (bind_texture(tex, bp._format)) {
              bp._bound = tex;
              bound = true;
            }
            break;
          }
        }
      }

      if (!bound) {
        if (metaldisplay_cat.is_debug()) {
          metaldisplay_cat.debug()
            << "Cannot bind " << *tex << " to " << plane << " of " << get_name()
            << "; copying it instead.\n";
        }
        ((CData *)cdata.p())->_textures[i]._rtm_mode = RTM_copy_texture;
      }
    }
  }

  if (_depth_plane._bound != nullptr && _depth_plane._plane == RTP_depth && want_stencil) {
    _stencil_plane._plane = RTP_stencil;
    _stencil_plane._format = MTLPixelFormatStencil8;
  }

  auto make_texture = [&](unsigned long format, int samples) -> MetalTexture {
    MTLTextureDescriptor *desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)format
                                                         width:_size[0]
                                                        height:_size[1]
                                                     mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    if (samples > 1) {
      desc.textureType = MTLTextureType2DMultisample;
      desc.sampleCount = samples;
      desc.usage = MTLTextureUsageRenderTarget;
    }
    MetalTexture texture = [device newTextureWithDescriptor:desc];
    if (texture != nil && metal_debug_labels) {
      texture.label = [NSString stringWithUTF8String:get_name().c_str()];
    }
    return texture;
  };

  // Create our own textures where needed.
  for (Bitplane &bp : _color_planes) {
    if (_sample_count > 1) {
      bp._texture = make_texture(bp._format, _sample_count);
      if (bp._bound == nullptr) {
        // The samples are resolved into this, which is what gets copied.
        bp._resolve = make_texture(bp._format, 1);
      }
    } else if (bp._bound == nullptr) {
      bp._texture = make_texture(bp._format, 1);
    }
    if ((bp._texture == nil && bp._bound == nullptr) ||
        (_sample_count > 1 && (bp._texture == nil || (bp._bound == nullptr && bp._resolve == nil)))) {
      return false;
    }
  }

  if (_depth_plane._format != MTLPixelFormatInvalid) {
    if (_sample_count > 1) {
      _depth_plane._texture = make_texture(_depth_plane._format, _sample_count);
      if (_depth_plane._bound == nullptr && need_depth_copy) {
        // Resolve the depth too, so that it can be copied.
        _depth_plane._resolve = make_texture(_depth_plane._format, 1);
      }
    } else if (_depth_plane._bound == nullptr) {
      _depth_plane._texture = make_texture(_depth_plane._format, 1);
    }
    if (_depth_plane._texture == nil && _depth_plane._bound == nullptr) {
      return false;
    }
  }

  if (_stencil_plane._format != MTLPixelFormatInvalid) {
    _stencil_plane._texture = make_texture(_stencil_plane._format, _sample_count);
    if (_stencil_plane._texture == nil) {
      return false;
    }
  }

  _attachment_size = _size;
  _needs_rebuild = false;
  return true;
}

/**
 * Points the GSG at our bitplanes, for the current page.
 */
void MetalGraphicsBuffer::
set_render_target() {
  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_V(mgsg, _gsg);

  // A bound texture's Metal texture can be replaced if the texture is
  // modified, so we ask for it every time.
  auto bound_texture = [&](const Bitplane &bp) -> MetalTexture {
    return (bp._bound != nullptr) ? mgsg->use_texture(bp._bound, 0) : nil;
  };

  MetalGraphicsStateGuardian::RenderTarget target;
  for (const Bitplane &bp : _color_planes) {
    int i = target._num_color++;
    target._planes[i] = bp._plane;
    if (_sample_count > 1) {
      target._color[i] = bp._texture;
      target._resolve[i] = (bp._bound != nullptr) ? bound_texture(bp) : bp._resolve;
    } else {
      target._color[i] = (bp._bound != nullptr) ? bound_texture(bp) : bp._texture;
    }
  }

  if (_depth_plane._format != MTLPixelFormatInvalid) {
    MetalTexture depth;
    if (_sample_count > 1) {
      depth = _depth_plane._texture;
      target._depth_resolve = (_depth_plane._bound != nullptr)
        ? bound_texture(_depth_plane) : _depth_plane._resolve;
    } else {
      depth = (_depth_plane._bound != nullptr) ? bound_texture(_depth_plane) : _depth_plane._texture;
    }
    target._depth = depth;
    if (_depth_plane._format == MTLPixelFormatDepth32Float_Stencil8) {
      target._stencil = depth;
    }
  }
  if (_stencil_plane._texture != nil) {
    target._stencil = _stencil_plane._texture;
  }

  target._slice = _page;
  target._width = _size[0];
  target._height = _size[1];
  // Render with the rows bottom-up, so that the textures have the same
  // orientation as in OpenGL, unless the buffer is asked to be inverted.
  target._inverted = !get_inverted();
  target._output_inverted = get_inverted();
  mgsg->set_render_target(target);
}

/**
 * Releases the textures that we render into, and forgets the bound textures.
 */
void MetalGraphicsBuffer::
release_attachments() {
  auto release = [](Bitplane &bp) {
    if (bp._texture != nil) {
      [bp._texture release];
      bp._texture = nil;
    }
    if (bp._resolve != nil) {
      [bp._resolve release];
      bp._resolve = nil;
    }
    bp._bound.clear();
    bp._format = 0;
  };
  for (Bitplane &bp : _color_planes) {
    release(bp);
  }
  _color_planes.clear();
  release(_depth_plane);
  release(_stencil_plane);
  _attachment_size.set(0, 0);
  _needs_rebuild = true;
}
