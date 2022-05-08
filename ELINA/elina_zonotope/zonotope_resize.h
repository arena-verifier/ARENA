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

#ifndef _ZONOTOPE_RESIZE_H_
#define _ZONOTOPE_RESIZE_H_

#include "zonotope_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resize dimensions */
/*********************/
zonotope_t* zonotope_add_dimensions(elina_manager_t* man,
		bool destructive, zonotope_t* z,
		elina_dimchange_t* dimchange,
		bool project);

zonotope_t* zonotope_remove_dimensions(elina_manager_t* man,
		bool destructive, zonotope_t* z,
		elina_dimchange_t* dimchange);

zonotope_t* zonotope_permute_dimensions(elina_manager_t* man,
		bool destructive,
		zonotope_t* z,
		elina_dimperm_t* permutation);

#ifdef __cplusplus
}
#endif

#endif
