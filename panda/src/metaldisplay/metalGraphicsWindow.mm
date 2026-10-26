/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsWindow.mm
 * @author Jace Mayer
 * @date 2026-10-16
 */

#include "metalGraphicsWindow.h"
#include "metalGraphicsPipe.h"
#include "metalGraphicsStateGuardian.h"
#include "config_metaldisplay.h"
#include "config_cocoadisplay.h"
#include "config_display.h"
#include "pStatTimer.h"

#import <MetalFX/MetalFX.h>

// MetalFX pulls in <simd/simd.h>, which defines macros that clash with Panda's
// vector typedefs in files included after this one in the composite file.
#undef vector_uchar
#undef vector_ushort
#undef vector_int
#undef vector_float
#undef vector_double
#undef vector_ulong

#import <AppKit/NSView.h>

TypeHandle MetalGraphicsWindow::_type_handle;

// Time spent waiting for the window server to give us a drawable, which is
// how waiting for vertical sync shows up.
static PStatCollector _wait_drawable_pcollector("Wait:Drawable");

/**
 *
 */
MetalGraphicsWindow::
MetalGraphicsWindow(GraphicsEngine *engine, GraphicsPipe *pipe,
                    std::string name,
                    const FrameBufferProperties &fb_prop,
                    const WindowProperties &win_prop,
                    int flags,
                    GraphicsStateGuardian *gsg,
                    GraphicsOutput *host) :
  CocoaGraphicsWindow(engine, pipe, std::move(name), fb_prop, win_prop, flags, gsg, host),
  _drawable_size(0, 0)
{
}

/**
 *
 */
MetalGraphicsWindow::
~MetalGraphicsWindow() {
}

/**
 * This function will be called within the draw thread before beginning
 * rendering for a given frame.  It should do whatever setup is required, and
 * return true if the frame should be rendered, or false if it should be
 * skipped.
 */
bool MetalGraphicsWindow::
begin_frame(FrameMode mode, Thread *current_thread) {
  begin_frame_spam(mode);
  if (_gsg == nullptr || _layer == nil) {
    return false;
  }

  if (mode == FM_refresh) {
    // We don't redraw the previous frame, but this lets the contents of the
    // last frame be copied, eg. for a screenshot.
    MetalTexture last_frame = (_scaled_color != nil) ? _scaled_color
                            : _decoupled ? _offscreen_color : _last_frame;
    MetalGraphicsStateGuardian *mgsg;
    DCAST_INTO_R(mgsg, _gsg, false);
    if (last_frame == nil || !mgsg->is_valid() || !mgsg->begin_frame(current_thread)) {
      return false;
    }
    mgsg->set_render_target(last_frame, (_sample_count > 1) ? nil : _depth_stencil,
                            _render_size[0], _render_size[1], false);
    return true;
  }

  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_R(mgsg, _gsg, false);
  if (!mgsg->is_valid()) {
    return false;
  }

  if (_size[0] <= 0 || _size[1] <= 0) {
    return false;
  }

  if (!update_drawable_size()) {
    return false;
  }

  if (!mgsg->begin_frame(current_thread)) {
    return false;
  }

  MetalTexture target = nil;
  if (_scaled_color != nil) {
    // Render at the lower resolution; this is upscaled when presenting.
    target = _scaled_color;
  }
  else if (_decoupled) {
    // Render into our own texture; we'll present it at the end of the frame
    // if a drawable is available.
    target = _offscreen_color;
  }
  if (!_decoupled) {
    // This may block until a drawable becomes available.
    {
      PStatTimer timer(_wait_drawable_pcollector, current_thread);
      _drawable = [[_layer nextDrawable] retain];
    }
    if (_drawable == nil) {
      if (metaldisplay_cat.is_debug()) {
        metaldisplay_cat.debug()
          << "No drawable available for " << get_name() << "\n";
      }
      mgsg->end_frame(current_thread);
      return false;
    }
    if (target == nil) {
      target = _drawable.texture;
    }
  }

  if (_msaa_color != nil) {
    // Render into the multisample texture, and resolve into the target.
    mgsg->set_render_target(_msaa_color, _depth_stencil,
                            _render_size[0], _render_size[1], false,
                            target);
  } else {
    mgsg->set_render_target(target, _depth_stencil,
                            _render_size[0], _render_size[1], false);
  }
  return true;
}

/**
 * This function will be called within the draw thread after rendering is
 * completed for a given frame.  It should do whatever finalization is
 * required.
 */
void MetalGraphicsWindow::
end_frame(FrameMode mode, Thread *current_thread) {
  end_frame_spam(mode);

  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_V(mgsg, _gsg);

  mgsg->end_render_pass();

  if (mode == FM_render) {
    copy_to_textures();
  }

  // In decoupled mode, only present if the previous frame has been presented,
  // in which case a drawable is available without waiting.
  MetalTexture frame = (_scaled_color != nil) ? _scaled_color : _offscreen_color;
  if (_decoupled && mode == FM_render && frame != nil &&
      _drawables_in_flight.load() < 1) {
    MetalDrawable drawable;
    {
      PStatTimer timer(_wait_drawable_pcollector, current_thread);
      drawable = [_layer nextDrawable];
    }
    if (drawable != nil &&
        drawable.texture.width == (NSUInteger)_drawable_size[0] &&
        drawable.texture.height == (NSUInteger)_drawable_size[1]) {
      if (_scaled_color != nil) {
        upscale(drawable.texture);
      } else {
        MetalBlitCommandEncoder blit = [mgsg->get_command_buffer() blitCommandEncoder];
        [blit copyFromTexture:_offscreen_color toTexture:drawable.texture];
        [blit endEncoding];
      }

      _drawables_in_flight.fetch_add(1);
      patomic<int> *in_flight = &_drawables_in_flight;
      [drawable addPresentedHandler:^(id<MTLDrawable> d) {
        in_flight->fetch_sub(1);
      }];
      _drawable = [drawable retain];
    }
  }
  else if (!_decoupled && _scaled_color != nil && _drawable != nil) {
    upscale(_drawable.texture);
  }

  // Metal schedules the presentation for when the command buffer completes,
  // synchronized with the display if displaySyncEnabled is set on the layer.
  if (_drawable != nil) {
    [mgsg->get_command_buffer() presentDrawable:_drawable];
  }

  mgsg->end_frame(current_thread);

  if (_drawable != nil) {
    if (!_decoupled && _scaled_color == nil) {
      // Keep the texture around so that we can still copy the frame from it
      // after it has been presented.  The window server won't give it back to
      // us to render into for at least another two frames.
      [_drawable.texture retain];
      if (_last_frame != nil) {
        [_last_frame release];
      }
      _last_frame = _drawable.texture;
    }
    [_drawable release];
    _drawable = nil;
  }

  if (mode == FM_render) {
    nassertv(!_flip_ready);
    _flip_ready = true;
    clear_cube_map_selection();
  }
}

/**
 * Closes the window right now.  Called from the window thread.
 */
void MetalGraphicsWindow::
close_window() {
  if (_gsg != nullptr) {
    // Make sure the GPU is no longer using our textures.
    MetalGraphicsStateGuardian *mgsg;
    DCAST_INTO_V(mgsg, _gsg);
    mgsg->finish();
  }

  release_textures();

  if (_layer != nil) {
    if (_view != nil) {
      [_view setLayer:nil];
      [_view setWantsLayer:NO];
    }
    [_layer release];
    _layer = nil;
  }
  _drawable_size.set(0, 0);

  CocoaGraphicsWindow::close_window();
}

/**
 * Opens the window right now.  Called from the window thread.  Returns true
 * if the window is successfully opened, or false if there was a problem.
 */
bool MetalGraphicsWindow::
open_window() {
  MetalGraphicsPipe *mpipe;
  DCAST_INTO_R(mpipe, _pipe, false);

  MetalGraphicsStateGuardian *mgsg;
  if (_gsg == nullptr) {
    // There is no old gsg.  Create a new one.
    mgsg = new MetalGraphicsStateGuardian(_engine, mpipe, nullptr);
    _gsg = mgsg;
  } else {
    // Metal render state does not depend on the framebuffer properties, so we
    // can share the GSG unless it belongs to a different engine.
    DCAST_INTO_R(mgsg, _gsg, false);
    if (mgsg->get_engine() != _engine) {
      mgsg = new MetalGraphicsStateGuardian(_engine, mpipe, mgsg);
      _gsg = mgsg;
    }
  }

  if (!CocoaGraphicsWindow::open_window()) {
    return false;
  }

  mgsg->reset_if_new();
  if (!mgsg->is_valid()) {
    close_window();
    return false;
  }

  if (!choose_formats()) {
    close_window();
    return false;
  }

  CGFloat contents_scale = 1;
  if (dpi_aware) {
    contents_scale = get_backing_scale_factor();
  }

  CAMetalLayer *layer = [[CAMetalLayer alloc] init];
  layer.device = mgsg->get_device();
  layer.pixelFormat = (MTLPixelFormat)_color_format;
  // We need to be able to read back from the drawable for screenshots and
  // copy-to-texture operations.
  layer.framebufferOnly = NO;
  layer.displaySyncEnabled = sync_video;
  _decoupled = !sync_video;
  layer.maximumDrawableCount = 3;
  layer.contentsScale = contents_scale;
  layer.delegate = (id<CALayerDelegate>)_view;
  _layer = layer;

  // Make the view layer-hosting, with our layer.
  [_view setLayer:layer];
  [_view setWantsLayer:YES];

  if (!update_drawable_size()) {
    close_window();
    return false;
  }

  return true;
}

/**
 * Determines the pixel formats to use based on the requested framebuffer
 * properties, and updates _fb_properties to reflect what we actually got.
 */
bool MetalGraphicsWindow::
choose_formats() {
  FrameBufferProperties props;

  if (_fb_properties.get_float_color()) {
    _color_format = MTLPixelFormatRGBA16Float;
    props.set_rgba_bits(16, 16, 16, 16);
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

  // Multisampling.
  MetalGraphicsStateGuardian *mgsg = DCAST(MetalGraphicsStateGuardian, _gsg);
  _sample_count = 1;
  if (_fb_properties.get_multisamples() > 1) {
    _sample_count = mgsg->choose_sample_count(_fb_properties.get_multisamples());
    if (_sample_count > 1) {
      props.set_multisamples(_sample_count);
    }
  }

  props.set_back_buffers(1);
  if (_color_format != MTLPixelFormatInvalid) {
    props.set_rgb_color(true);
  }
  props.set_force_hardware(true);
  _fb_properties = props;
  return true;
}

/**
 * Returns true if a call to set_pixel_zoom() will be respected.  We render at
 * the reduced resolution and upscale the result when presenting it.
 */
bool MetalGraphicsWindow::
supports_pixel_zoom() const {
  return true;
}

/**
 * Makes sure the drawables and the textures we render into match the window
 * size and the pixel zoom.  Returns false if something went wrong.
 */
bool MetalGraphicsWindow::
update_drawable_size() {
  LVecBase2i render_size = get_fb_size();
  if (_drawable_size == _size && _render_size == render_size) {
    return true;
  }

  if (metaldisplay_cat.is_debug()) {
    metaldisplay_cat.debug()
      << "Resizing drawables of " << get_name() << " to "
      << _size[0] << "x" << _size[1] << ", rendering at "
      << render_size[0] << "x" << render_size[1] << "\n";
  }

  _layer.drawableSize = CGSizeMake(_size[0], _size[1]);

  // The GPU may still be using the old textures, but the command buffers keep
  // them alive for as long as they need them.
  release_textures();

  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_R(mgsg, _gsg, false);
  MetalDevice device = mgsg->get_device();

  auto make_texture = [&](unsigned long format, const LVecBase2i &size,
                          MTLTextureUsage usage, int samples) -> MetalTexture {
    MTLTextureDescriptor *desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)format
                                                         width:size[0]
                                                        height:size[1]
                                                     mipmapped:NO];
    desc.usage = usage;
    desc.storageMode = MTLStorageModePrivate;
    if (samples > 1) {
      desc.textureType = MTLTextureType2DMultisample;
      desc.sampleCount = samples;
      desc.usage = MTLTextureUsageRenderTarget;
    }
    return [device newTextureWithDescriptor:desc];
  };

  bool scaled = (render_size != _size);
  if (scaled) {
    MTLTextureUsage color_usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    std::string upscaler = metal_upscaler.get_value();
    if (upscaler == "metalfx") {
      if (@available(macOS 13.0, *)) {
        if ([MTLFXSpatialScalerDescriptor supportsDevice:device]) {
          MTLFXSpatialScalerDescriptor *desc = [[MTLFXSpatialScalerDescriptor alloc] init];
          desc.colorTextureFormat = (MTLPixelFormat)_color_format;
          desc.outputTextureFormat = (MTLPixelFormat)_color_format;
          desc.inputWidth = render_size[0];
          desc.inputHeight = render_size[1];
          desc.outputWidth = _size[0];
          desc.outputHeight = _size[1];
          // An sRGB or floating-point texture holds linear values; otherwise, the
          // values are perceptually (gamma) encoded.
          desc.colorProcessingMode = (_color_format == MTLPixelFormatBGRA8Unorm)
            ? MTLFXSpatialScalerColorProcessingModePerceptual
            : MTLFXSpatialScalerColorProcessingModeLinear;
          id<MTLFXSpatialScaler> scaler = [desc newSpatialScalerWithDevice:device];
          [desc release];

          if (scaler != nil) {
            _upscaled = make_texture(_color_format, _size, scaler.outputTextureUsage, 1);
            if (_upscaled != nil) {
              color_usage |= scaler.colorTextureUsage;
              scaler.inputContentWidth = render_size[0];
              scaler.inputContentHeight = render_size[1];
              _spatial_scaler = scaler;
            } else {
              [scaler release];
            }
          }
        }
      }
      if (_spatial_scaler == nil) {
        metaldisplay_cat.warning()
          << "Could not create MetalFX upscaler; using linear filtering instead.\n";
      }
    }
    else if (upscaler != "linear") {
      metaldisplay_cat.warning()
        << "Unknown metal-upscaler " << upscaler << "; using linear filtering.\n";
    }

    _scaled_color = make_texture(_color_format, render_size, color_usage, 1);
    if (_scaled_color == nil) {
      return false;
    }
  }
  else if (_decoupled) {
    _offscreen_color = make_texture(_color_format, _size,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, 1);
    if (_offscreen_color == nil) {
      return false;
    }
  }

  if (_sample_count > 1) {
    _msaa_color = make_texture(_color_format, render_size, MTLTextureUsageRenderTarget, _sample_count);
    if (_msaa_color == nil) {
      metaldisplay_cat.error()
        << "Failed to create multisample color buffer for " << get_name() << "\n";
      return false;
    }
  }

  if (_depth_stencil_format != MTLPixelFormatInvalid) {
    // Shader read access is needed for copying the depth buffer to a texture.
    _depth_stencil = make_texture(_depth_stencil_format, render_size,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, _sample_count);
    if (_depth_stencil == nil) {
      metaldisplay_cat.error()
        << "Failed to create depth buffer for " << get_name() << "\n";
      return false;
    }
    if (metal_debug_labels) {
      _depth_stencil.label = @"Panda3D window depth/stencil";
    }
  }

  _drawable_size = _size;
  _render_size = render_size;
  return true;
}

/**
 * Releases the textures we render into, which depend on the window size.
 */
void MetalGraphicsWindow::
release_textures() {
  MetalTexture *textures[] = {&_depth_stencil, &_msaa_color, &_offscreen_color,
                              &_scaled_color, &_upscaled, &_last_frame};
  for (MetalTexture *texture : textures) {
    if (*texture != nil) {
      [*texture release];
      *texture = nil;
    }
  }
  if (_spatial_scaler != nil) {
    [_spatial_scaler release];
    _spatial_scaler = nil;
  }
  _drawable_size.set(0, 0);
  _render_size.set(0, 0);
}

/**
 * Upscales the scene, which was rendered into _scaled_color, to fill the
 * given texture, which is the size of the window.
 */
void MetalGraphicsWindow::
upscale(MetalTexture dest) {
  MetalGraphicsStateGuardian *mgsg;
  DCAST_INTO_V(mgsg, _gsg);
  MetalCommandBuffer cmd = mgsg->get_command_buffer();

  if (_spatial_scaler != nil) {
    // We only create the scaler if MetalFX is available.
    if (@available(macOS 13.0, *)) {
      id<MTLFXSpatialScaler> scaler = _spatial_scaler;
      scaler.colorTexture = _scaled_color;
      scaler.outputTexture = _upscaled;
      [scaler encodeToCommandBuffer:cmd];

      MetalBlitCommandEncoder blit = [cmd blitCommandEncoder];
      [blit copyFromTexture:_upscaled toTexture:dest];
      [blit endEncoding];
      return;
    }
  }
  mgsg->copy_scaled(_scaled_color, dest);
}
