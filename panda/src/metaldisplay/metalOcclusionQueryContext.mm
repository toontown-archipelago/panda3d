/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file metalOcclusionQueryContext.mm
 * @author Jace Mayer
 * @date 2026-10-24
 */

#include "metalOcclusionQueryContext.h"
#include "metalGraphicsStateGuardian.h"

TypeHandle MetalOcclusionQueryContext::_type_handle;

/**
 *
 */
MetalOcclusionQueryContext::
MetalOcclusionQueryContext(MetalGraphicsStateGuardian *gsg) :
  _gsg(gsg)
{
}

/**
 *
 */
MetalOcclusionQueryContext::
~MetalOcclusionQueryContext() {
  for (Slot &slot : _slots) {
    [slot._buffer release];
  }
  if (_cmd != nil) {
    [_cmd release];
  }
}

/**
 * Records that the samples of a render pass are counted in the given slot of
 * the visibility result buffer, by the given command buffer.
 */
void MetalOcclusionQueryContext::
add_slot(MetalBuffer buffer, size_t offset, MetalCommandBuffer cmd) {
  _slots.push_back({[buffer retain], offset});
  if (cmd != _cmd) {
    [cmd retain];
    if (_cmd != nil) {
      [_cmd release];
    }
    _cmd = cmd;
  }
}

/**
 * Returns true if the query's answer is ready, false otherwise.  If this
 * returns false, the application must continue to poll until it returns
 * true.
 */
bool MetalOcclusionQueryContext::
is_answer_ready() const {
  return _cmd == nil || [_cmd status] >= MTLCommandBufferStatusCompleted;
}

/**
 * Requests the graphics engine to expedite the pending answer--the application
 * is now waiting until the answer is ready.
 */
void MetalOcclusionQueryContext::
waiting_for_answer() {
  if (is_answer_ready()) {
    return;
  }
  if (_cmd == _gsg->get_command_buffer()) {
    // It's still being recorded, so it has to be submitted first.
    _gsg->flush_and_wait();
  } else {
    [_cmd waitUntilCompleted];
  }
}

/**
 * Returns the number of fragments (samples) that passed the depth test and
 * were rendered during the query.  This may block until the answer is ready.
 */
int MetalOcclusionQueryContext::
get_num_fragments() const {
  if (!is_answer_ready()) {
    ((MetalOcclusionQueryContext *)this)->waiting_for_answer();
  }
  if (_incomplete) {
    return 1;
  }
  uint64_t total = 0;
  for (const Slot &slot : _slots) {
    total += *(const uint64_t *)((const unsigned char *)slot._buffer.contents + slot._offset);
  }
  return (int)std::min(total, (uint64_t)INT_MAX);
}
