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


#ifndef __OPT_PK_TEST_H
#define __OPT_PK_TEST_H

#include "opt_pk_config.h"
#include "opt_pk_internal.h"
#include "opt_pk.h"

#ifdef __cplusplus
extern "C" {
#endif

bool opt_poly_leq(opt_pk_internal_t * opk, opt_matrix_t * C, opt_matrix_t * F);

void fuse_generators_intersecting_blocks(opt_matrix_t *F, opt_pk_t ** poly_a, array_comp_list_t *acla, unsigned short int *ca, 
					 size_t * num_vertex_a, char *intersect_map, unsigned short int maxcols);

#ifdef __cplusplus
}
#endif

#endif
