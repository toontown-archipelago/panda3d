/**
 * PANDA 3D SOFTWARE
 * Copyright (c) Carnegie Mellon University.  All rights reserved.
 *
 * All use of this software is subject to the terms of the revised BSD
 * license.  You should have received a copy of this license along
 * with this source code in a file named "LICENSE."
 *
 * @file config_metaldisplay.h
 * @author Jace Mayer
 * @date 2026-10-14
 */

#ifndef CONFIG_METALDISPLAY_H
#define CONFIG_METALDISPLAY_H

#include "pandabase.h"
#include "notifyCategoryProxy.h"
#include "configVariableBool.h"
#include "configVariableInt.h"
#include "configVariableString.h"

NotifyCategoryDecl(metaldisplay, EXPCL_METALDISPLAY, EXPTP_METALDISPLAY);

extern ConfigVariableInt metal_frames_in_flight;
extern ConfigVariableBool metal_dump_msl;
extern ConfigVariableBool metal_debug_labels;
extern ConfigVariableString metal_upscaler;

extern EXPCL_METALDISPLAY void init_libmetaldisplay();
extern "C" EXPCL_METALDISPLAY int get_pipe_type_p3metaldisplay();

#endif
