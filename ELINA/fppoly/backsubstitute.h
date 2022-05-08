/*
 *
 *  This source file is part of ELINA (ETH LIbrary for Numerical Analysis).
 *  ELINA is Copyright © 2019 Department of Computer Science, ETH Zurich
 *  This software is distributed under GNU Lesser General Public License Version 3.0.
 *  For more information, see the ELINA project website at:
 *  http://elina.ethz.ch
 *
 *  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
 *  EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
 *  THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
 *  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
 *  TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY     
 *  DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
 *  SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
 *  ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
 *  CONTRACT, TORT OR OTHERWISE).
 *
 */



#ifndef __BACKSUBSTITUTE_H_INCLUDED__
#define __BACKSUBSTITUTE_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

#include "fppoly.h"
#include "expr.h"
#include "compute_bounds.h"
#include "relu_approx.h"

void update_state_using_previous_layers_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno, bool layer_by_layer,  bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement);

void update_state_layer_by_layer_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno, bool layer_by_layer,  bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement);

void update_state_layer_by_layer_parallel_until_certain_layer(elina_manager_t *man, fppoly_t *fp, size_t layerno, bool is_blk_segmentation, int block_start_layer);

void update_layer_for_concrete_img_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno, int defined_layer);
#ifdef __cplusplus
 }
#endif

#endif

