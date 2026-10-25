/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalGraphicsPipe.I
 * @author Jace Mayer
 * @date 2026-10-14
 */

/**
 * Returns the Metal device associated with this pipe.
 */
INLINE MetalDevice MetalGraphicsPipe::
get_device() const {
  return _device;
}
