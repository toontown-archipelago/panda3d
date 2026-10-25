if(NOT APPLE OR NOT HAVE_COCOA OR NOT HAVE_METAL)
  return()
endif()

if(NOT TARGET SPIRV-Cross::MSL)
  message("")
  message(WARNING "HAVE_METAL is enabled, but SPIRV-Cross was not found with "
    "MSL support; not building the Metal renderer.")
  return()
endif()

set(P3METALDISPLAY_HEADERS
  config_metaldisplay.h
  metalGeomMunger.h
  metalGraphicsBuffer.h
  metalGraphicsPipe.h metalGraphicsPipe.I
  metalGraphicsStateGuardian.h metalGraphicsStateGuardian.I
  metalGraphicsWindow.h
  metalIndexBufferContext.h
  metalOcclusionQueryContext.h
  metalSamplerContext.h
  metalShaderContext.h metalShaderContext.I
  metalTextureContext.h
  metalTypes.h
  metalVertexBufferContext.h
)

set(P3METALDISPLAY_SOURCES
  config_metaldisplay.mm
  metalGeomMunger.mm
  metalGraphicsBuffer.mm
  metalGraphicsPipe.mm
  metalGraphicsStateGuardian.mm
  metalGraphicsWindow.mm
  metalIndexBufferContext.mm
  metalOcclusionQueryContext.mm
  metalSamplerContext.mm
  metalShaderContext.mm
  metalTextureContext.mm
  metalVertexBufferContext.mm
)

composite_sources(p3metaldisplay P3METALDISPLAY_SOURCES)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "Metal")
add_metalib(p3metaldisplay ${MODULE_TYPE}
  ${P3METALDISPLAY_HEADERS} ${P3METALDISPLAY_SOURCES}
  COMPONENTS p3cocoadisplay)
unset(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME)

# SPIRV-Cross reports errors by throwing exceptions.
set_target_properties(p3metaldisplay PROPERTIES
  DEFINE_SYMBOL BUILDING_METALDISPLAY
  CXX_EXCEPTIONS ON)

# The renderer requires macOS 11.  MetalFX (macOS 13) is weakly linked, and
# only used if it is available at runtime.
target_compile_options(p3metaldisplay PRIVATE "-mmacosx-version-min=11.0")
target_link_options(p3metaldisplay PRIVATE "-mmacosx-version-min=11.0")

# Frameworks:
find_library(METAL_LIBRARY Metal)
find_library(QUARTZCORE_LIBRARY QuartzCore)
target_link_libraries(p3metaldisplay panda SPIRV-Cross::MSL
  ${METAL_LIBRARY} ${QUARTZCORE_LIBRARY} "-weak_framework MetalFX")

mark_as_advanced(METAL_LIBRARY QUARTZCORE_LIBRARY)

install(TARGETS p3metaldisplay
  EXPORT Metal COMPONENT Metal
  DESTINATION ${MODULE_DESTINATION}
  ARCHIVE COMPONENT MetalDevel)

export_targets(Metal NAMESPACE "Panda3D::Metal::" COMPONENT MetalDevel)
