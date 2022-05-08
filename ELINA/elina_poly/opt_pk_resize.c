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


/* ********************************************************************** */
/* opt_pk_resize.c: change and permutation of dimensions  */
/* ********************************************************************** */


#include "opt_pk_config.h"
#include "opt_pk_vector.h"
#include "opt_pk_project.h"
#include "opt_pk_matrix.h"
#include "opt_pk.h"
#include "opt_pk_representation.h"
#include "opt_pk_user.h"
#include "opt_pk_constructor.h"



/* ********************************************************************** */
/* 		Add Dimensions with constraints*/
/* ********************************************************************** */

opt_pk_array_t* opt_pk_add_dimensions_cons(elina_manager_t* man,
			bool destructive,
			opt_pk_array_t* oa,
			elina_dimchange_t* dimchange,
			bool project)
{
  opt_pk_array_t* op;
  opt_pk_internal_t* opk = opt_pk_init_from_manager(man,ELINA_FUNID_ADD_DIMENSIONS);
  opt_pk_internal_realloc_lazy(opk,oa->maxcols+dimchange->intdim+dimchange->realdim - 2);
  
  array_comp_list_t * acla = oa->acl;
  unsigned short int size = dimchange->intdim + dimchange->realdim;
  /* Return empty if empty */
  if(oa->is_bottom || !acla){
	 man->result.flag_best = man->result.flag_exact = true;
	 if (destructive){
	      	oa->maxcols += size;
	      	return oa;
	 }
	 else {
	      	return opt_pk_bottom(man,
			oa->maxcols + size - opk->dec,
			0);
	}
  }
  unsigned short int num_compa = acla->size;
  unsigned short int i, l,k;
  /*************************************
		Minimized the input
  *************************************/
  opt_pk_t ** poly_a = oa->poly;
  if(opk->funopt->algorithm>0){
	for(k=0; k < num_compa; k++){
	     opt_pk_t * oak = poly_a[k];
	     opt_poly_chernikova(man,oak,"add dimensions");
	     if (opk->exn){
	         opk->exn = ELINA_EXC_NONE;
		/* we can still operate on the available matrix */
    	     }
	     if(!oak->C && !oak->F){
		 man->result.flag_best = man->result.flag_exact = true;
		 if (destructive){
		     opt_poly_set_bottom(opk,oa);
		     oa->maxcols += dimchange->intdim + dimchange->realdim;
		     return oa;
		 }
		 else {
		      return opt_pk_bottom(man,
			     oa->maxcols + dimchange->intdim - 2,
			     0);
		 }
	     }
	}
  }
  /*******************************
		Create mapping for independent components
  ********************************/
  unsigned short int maxcols = oa->maxcols;
  unsigned short int * cmap = (unsigned short int *)calloc(maxcols+1,sizeof(unsigned short int));
  unsigned short int * ncmap = (unsigned short int *)calloc(size,sizeof(unsigned short int));
  l = 0;
  k = opk->dec;
  elina_dim_t * dim = dimchange->dim;
  for(i=k; i <=maxcols; i++){
	//unsigned short int var = dim[l] + opk->dec;
	while((l < size) && ((dim[l] + opk->dec)==i)){
		ncmap[l] = k;
		//var = dim[l] + opk->dec;
		l++;
		k++;
	}
	cmap[i] = k;
	k++;
  }
  /*****************************************
	Handle independent components
  ******************************************/
  
  array_comp_list_t * acl = create_array_comp_list();
  comp_list_t * cla = acla->head;
  while(cla != NULL){
	comp_list_t * cl = create_comp_list();
	comp_t * c = cla->head;
	while(c != NULL){
		unsigned short int numa = c->num;
		unsigned short int num = cmap[numa];
		//printf("numa: %d num: %d\n",numa,num);
		insert_comp(cl,num);
		c = c->next;
	}
	insert_comp_list_tail(acl,cl);
  	cla = cla->next;
  }
  opt_pk_t ** poly = destructive ? poly_a : (opt_pk_t **)malloc(num_compa*sizeof(opt_pk_t *));
  if(!destructive){
	  for(k=0;  k< num_compa; k++){
		  unsigned short int k1 =  k;
		  opt_pk_t * src = poly_a[k];
		  poly[k1] = opt_poly_alloc(src->intdim,src->realdim);
		  poly[k1]->nbeq = src->nbeq;
		  poly[k1]->C = src->C ? opt_matrix_copy(src->C) : NULL;
		  poly[k1]->F = src->F ? opt_matrix_copy(src->F) : NULL;
		  poly[k1]->satC = src->satC ? opt_satmat_copy(src->satC) : NULL;
		  poly[k1]->satF = src->satF ? opt_satmat_copy(src->satF) : NULL;
		  
		  poly[k1]->nbline = src->nbline;
		  poly[k1]->status = src->status;
		  poly[k1]->is_minimized = src->is_minimized;
		  //op = opt_matrix_add_dimensions(opk, destructive, oa->C, dimchange, project);
	  }
  }
 
  if(project){
	unsigned short int num_comp = num_compa + size;
	//if(destructive){
	poly = (opt_pk_t **)realloc(poly, num_comp*sizeof(opt_pk_t*));
	//}else{
	//	poly = (opt_pk_t **)realloc(poly, num_comp*sizeof(opt_pk_t*));
	//}
	// Handle project with generators
	for(i = 0; i < size; i++){
		unsigned short int i1 = num_compa + i;
		poly[i1] = opt_poly_alloc(1,0);
		poly[i1]->nbeq = 1;
		poly[i1]->C = opt_matrix_alloc(2,1+opk->dec,false);
		opt_numint_t **p = poly[i1]->C->p;
		opt_numint_t *pi = p[0];
		pi[0] = 0;
		pi[1] = 0;
		pi[2] = 1;
		pi = p[1];
                pi[0] = 1;
		pi[1] = 1;
		pi[2] = 0;
		poly[i1]->is_minimized = true;
		comp_list_t * cl = create_comp_list();
		insert_comp(cl,ncmap[i]);
		insert_comp_list_tail(acl,cl);
		opt_poly_chernikova(man,poly[i1],"convert to gen");
	}
  }
  free(cmap);
  free(ncmap);
  if(destructive){
	op = oa;
	op->maxcols = maxcols + size;
	free(acla);
	op->acl = acl;
  }
  else{
  	op = opt_pk_array_alloc(poly,acl,maxcols+size);
	
  }
 
  return op;
}


/***********************************
	Add dimensions
************************************/
opt_pk_array_t* opt_pk_add_dimensions(elina_manager_t* man,
			bool destructive,
			opt_pk_array_t* oa,
			elina_dimchange_t* dimchange,
			bool project){
	#if defined (TIMING)
		start_timing();
	#endif

        opt_pk_array_t * op;
	op = opt_pk_add_dimensions_cons(man,destructive,oa,dimchange,project);
	#if defined (TIMING)
		record_timing(add_dimension_time);
	#endif
	return op;
}


/*******************************
	Remove Dimensions
********************************/
opt_pk_array_t* opt_pk_remove_dimensions(elina_manager_t* man,
			   bool destructive,
			   opt_pk_array_t* oa,
			   elina_dimchange_t* dimchange)
{
  //printf("remove start %p\n",oa);
  //elina_lincons0_array_t arr1 = opt_pk_to_lincons_array(man,oa);
  //elina_lincons0_array_fprint(stdout,&arr1,NULL);
  //fflush(stdout);
   #if defined(TIMING)
 	 start_timing();
  #endif  
  opt_pk_array_t* op;
  size_t dimsup;
  dimsup = dimchange->intdim+dimchange->realdim;
  opt_pk_internal_t* opk = opt_pk_init_from_manager(man,ELINA_FUNID_REMOVE_DIMENSIONS);
  array_comp_list_t * acla = oa->acl;
  /* Return empty if empty */
  if(oa->is_bottom || !acla){
	man->result.flag_best = man->result.flag_exact = true;
      if (destructive){
          oa->maxcols -= dimsup;
	  #if defined(TIMING)
 	 		record_timing(remove_dimension_time);
  	  #endif
          return oa;
      }
      else {
          // Fix Me
	  #if defined(TIMING)
 	 		record_timing(remove_dimension_time);
  	  #endif
          return opt_pk_bottom(man,
                               oa->maxcols - dimsup - opk->dec,
                               dimchange->realdim);
      }	
    	
  }
  unsigned short int num_compa = acla->size;
  unsigned short int maxcols = oa->maxcols;
  unsigned short int k;
  opt_pk_t ** poly_a = oa->poly;
  for(k=0; k < num_compa; k++){
        opt_pk_t * oak = poly_a[k];
	if(opk->funopt->algorithm < 0){
		opt_poly_obtain_F(man,oak,"convert to gen");
	}
	else{
		opt_poly_chernikova(man,oak,"convert to gen");
	}
		
		//if overflow exception
	if(opk->exn){
	   opk->exn = ELINA_EXC_NONE;
	   if (!oak->F){
		man->result.flag_best = man->result.flag_exact = false;
		opt_pk_array_t * op = destructive ? oa : opt_pk_array_alloc(NULL,NULL,oa->maxcols);
		op->maxcols -= dimsup;
		opt_poly_set_top(opk,op);
		record_timing(remove_dimension_time);
		return op;
	   }
	}

	if(!oak->C && !oak->F){
	    man->result.flag_best = man->result.flag_exact = true;
   	    if (destructive){
		oa->maxcols -= dimsup;
	  	#if defined(TIMING)
 	 	     record_timing(remove_dimension_time);
  	        #endif
          	return oa;
    	    }
	    else{
		#if defined(TIMING)
 	 	     record_timing(remove_dimension_time);
  	  	#endif
          	return opt_pk_bottom(man,
                       oa->maxcols - dimsup - opk->dec,
                       dimchange->realdim);
	    }
	}
  }

  elina_dim_t * dima = dimchange->dim;
  
  
  /*********************************
	Handle independent components
  *********************************/
  unsigned short int * map = (unsigned short int *)calloc(maxcols, sizeof(unsigned short int));
 
  unsigned short int l = 0;
  for(k=opk->dec; k < maxcols; k++){
	//unsigned short int var = dima[l] + opk->dec;
	if((l < dimsup) && (k==(dima[l] + opk->dec))){
		map[k] = maxcols+1;
		l++;
	}
	else{
		map[k] = k - l;
	}
  }

  comp_list_t * cla = acla->head; 
  array_comp_list_t * acl = create_array_comp_list();
  unsigned short int num_comp = 0;
  while(cla!=NULL){
	comp_list_t * cl = create_comp_list();
	comp_t * c = cla->head;
	while(c!=NULL){
		unsigned short int numa = c->num;
		unsigned short int num = map[numa];
		if(num != (maxcols +1)){
			insert_comp(cl,num);
		}
		c = c->next;
	}
	if(cl->size){
		insert_comp_list(acl,cl);
		num_comp++;
	}
	else{
		free_comp_list(cl);
	}
	cla = cla->next;
  }
  /*********************************
	Remove variables from the blocks
  **********************************/
  cla = acla->head;
  if(destructive){
	k=0;
	while(k < num_compa){
		//printf("AA %d %d\n",k,num_compa);
		//fflush(stdout);
		unsigned short int comp_size = cla->size;
		unsigned short int * ca = to_sorted_array(cla,maxcols);
		opt_pk_t * oak = poly_a[k];
    		opt_pk_t * ot;
		/**************************
			Find the variables to remove for this component.
		****************************/
		elina_dim_t * ndim = (elina_dim_t *)calloc(comp_size, sizeof(elina_dim_t));
		unsigned short int size = 0;
		unsigned short int i,j;
        	for(i=0; i < comp_size; i++){
			unsigned short int num = ca[i];
			for(j=0; j < dimsup; j++){
				unsigned short int var = dima[j] + opk->dec;
				if(var==num){
					ndim[size] = i;
					size++;
				}
			}
		}
		
		if(size==comp_size){
			comp_list_t * tcla = cla->next;
			remove_comp_list(acla,cla);
			cla = tcla;
			free(ca);
			free(ndim);
			opt_pk_t * tpoly = poly_a[k];
			unsigned short int k1;
			for(k1=k; k1 < num_compa - 1; k1++){
				poly_a[k1] = poly_a[k1+1];
			}
			opt_poly_clear(tpoly);
			num_compa--;
			continue;
		}
		else if(size){
			//ot = opt_poly_alloc(oak->intdim, oak->realdim);
			if(oak->C){
			   opt_matrix_free(oak->C);
			   oak->C = NULL;
			}
			if(oak->satC){
			   opt_satmat_free(oak->satC);
			   oak->satC = NULL;
			}
			if(oak->satF){
			   opt_satmat_free(oak->satF);
			   oak->satF = NULL;
			}
			elina_dimchange_t dimchange1;
			dimchange1.dim = ndim;
			dimchange1.intdim = size;
			dimchange1.realdim = 0;
			poly_a[k]->F = opt_matrix_remove_dimensions(opk,false,oak->F,&dimchange1);
			//TODO: check its impact
			//opt_matrix_free(oak->F);
			//oak->F=NULL;
			poly_a[k]->intdim = comp_size - size;
			poly_a[k]->nbline = oak->nbline;
			poly_a[k]->nbeq = 0;
			
			//if (opk->funopt->algorithm>0){
			    opt_poly_chernikova(man,poly_a[k],"of the result");
				
			    if (opk->exn){
				opk->exn = ELINA_EXC_NONE;
			    }
		        //}
			man->result.flag_best = man->result.flag_exact =
			     dimchange1.intdim==0;
		}
		free(ndim);
		cla = cla->next;
		k++;
	}
	
	array_comp_list_t * tmp = acl;
        oa->acl = copy_array_comp_list(acl);
	//printf("remove output %p\n",oa->poly[0]->F);
	//fflush(stdout);
	free_array_comp_list(tmp);
	oa->maxcols = oa->maxcols - dimsup;
	#if defined(TIMING)
 	 	record_timing(remove_dimension_time);
  	#endif
	free(map);  
	return oa;
  }
  else{
	opt_pk_t ** poly = (opt_pk_t **)malloc(num_comp*sizeof(opt_pk_t*));
	unsigned short int k1 = 0;
	for(k=0; k < num_compa; k++){
		unsigned short int comp_size = cla->size;
		unsigned short int * ca = to_sorted_array(cla,maxcols);
		opt_pk_t * oak = poly_a[k];
		
    		opt_pk_t * ot;
		/**************************
			Find the variables to remove for this component.
		****************************/
		elina_dim_t * ndim = (elina_dim_t *)calloc(comp_size, sizeof(elina_dim_t));
		unsigned short int size = 0;
		unsigned short int i,j;
        	for(i=0; i < comp_size; i++){
			unsigned short int num = ca[i];
			for(j=0; j < dimsup; j++){
				unsigned short int var = dima[j] + opk->dec;
				if(var==num){
					ndim[size] = i;
					size++;
				}
			}
		}
		if(size==comp_size){
			free(ca);
			free(ndim);
			cla = cla->next;
			continue;
		}
		else if(size){
		        elina_dimchange_t dimchange1;
			dimchange1.dim = ndim;
			dimchange1.intdim = size;
			dimchange1.realdim = 0;
			poly[k1] = opt_poly_alloc(oak->intdim - size,
				                      oak->realdim);
			poly[k1]->F = opt_matrix_remove_dimensions(opk,false,oak->F,&dimchange1);
			poly[k1]->nbeq = 0;
			poly[k1]->nbline = oak->nbline;
			//if (opk->funopt->algorithm>0){
			    opt_poly_chernikova(man,poly[k1],"of the result");
			    if (opk->exn){
				opk->exn = ELINA_EXC_NONE;
			    }
			//}
			man->result.flag_best = man->result.flag_exact =
				    dimchange1.intdim==0;
		}
		else{
			poly[k1] = opt_poly_alloc(oak->intdim,oak->realdim);
			opt_poly_copy(poly[k1],oak);
			poly[k1]->is_minimized = oak->is_minimized;
		}

		k1++;
		free(ca);
        	free(ndim);
		cla = cla->next;
	}
	poly = (opt_pk_t **)realloc(poly,k1*sizeof(opt_pk_t*));
	array_comp_list_t * res = copy_array_comp_list(acl);
	free_array_comp_list(acl);
	op = opt_pk_array_alloc(poly,res,maxcols - dimsup);
	#if defined(TIMING)
 	 	record_timing(remove_dimension_time);
   	#endif
	free(map);  
	return op;
  }
  
}


/*******************************
	Permute Dimensions
********************************/
opt_pk_array_t* opt_pk_permute_dimensions(elina_manager_t* man,
			    bool destructive,
			    opt_pk_array_t* oa,
			    elina_dimperm_t* permutation)
{
  #if defined(TIMING)
 	 start_timing();
  #endif
  opt_pk_array_t* op;
  opt_pk_internal_t* opk = opt_pk_init_from_manager(man,ELINA_FUNID_PERMUTE_DIMENSIONS);
  array_comp_list_t * acla = oa->acl;
  if(oa->is_bottom || !acla){
	if(destructive){
		#if defined(TIMING)
 	 		record_timing(permute_dimension_time);
  		#endif
		return oa;
	}
	else{
		#if defined(TIMING)
 	 		record_timing(permute_dimension_time);
  		#endif
		return opt_pk_bottom(man,oa->maxcols -2,0);
	}
  }
  unsigned short int num_comp = acla->size;
  opt_pk_t ** poly_a = oa->poly;
   unsigned short int k;
  /***************************************
	Minimize the input
  ***************************************/
  if(opk->funopt->algorithm>0){
	for(k=0; k < num_comp; k++){
	    if (opk->funopt->algorithm>0){
    		/* Minimize the argument */
    		opt_poly_chernikova(man,poly_a[k],"of the argument");
    		if (opk->exn){
     		    opk->exn = ELINA_EXC_NONE;
      		/* we can still operate on the available matrix */
    		}
  	    }
	}
  }
  unsigned short int **nca_arr = (unsigned short int **)calloc(num_comp,sizeof(unsigned short int **));
  unsigned short int maxcols = oa->maxcols;
  /**********************
	Handle the independent components
  ***********************/
  array_comp_list_t * acl;  
  acl = create_array_comp_list();
  comp_list_t * cla = acla->head;
  elina_dim_t * dima = permutation->dim;
  k = 0;
  while(cla!=NULL){
	comp_list_t *cl = create_comp_list();
	comp_t * c = cla->head;
	while(c!=NULL){
		unsigned short int numa = c->num - opk->dec;
		unsigned short int num = dima[numa] + opk->dec; 
		//printf("numa: %d num: %d\n",numa,dima[numa]);
		insert_comp(cl,num);
		c = c->next;
	}
	//print_comp_list(cl,oa->maxcols);
	insert_comp_list(acl,cl);
	nca_arr[k] = to_sorted_array(cl,maxcols);
	k++;
	cla = cla->next;
  }
  
  opt_pk_t ** poly;
  poly = destructive ? poly_a : (opt_pk_t **)malloc(num_comp*sizeof(opt_pk_t *));
  if(!destructive){
	for(k=0; k < num_comp; k++){
		opt_pk_t * src = poly_a[k];
		unsigned short int k1 = num_comp - k - 1;
		poly[k1] = opt_poly_alloc(src->intdim,src->realdim);
		poly[k1]->nbeq = src->nbeq;
		poly[k1]->nbline = src->nbline;
		poly[k1]->satC = src->satC ? opt_satmat_copy(src->satC) : NULL;
		poly[k1]->satF = src->satF ? opt_satmat_copy(src->satF) : NULL;
		poly[k1]->status = src->status;
		poly[k1]->is_minimized = src->is_minimized;
	}
	op = opt_pk_array_alloc(poly,acl,maxcols);
  }
  else{
	//
        op = oa;
	//free_array_comp_list(oa->acl);
	op->acl = copy_array_comp_list(acl);
	free_array_comp_list(acl);
  }
  cla = acla->head;
  
  for(k=0; k < num_comp; k++){
      unsigned short int k1 = num_comp - k - 1;
      unsigned short int k2 = destructive ? k : k1;
	  unsigned short int comp_size = cla->size;
	  unsigned short int * ca = to_sorted_array(cla,maxcols);
	  unsigned short int * nca = nca_arr[k];
	  opt_pk_t *oak = poly_a[k];
	  man->result.flag_best = man->result.flag_exact = true;
	  elina_dim_t * dim = (elina_dim_t *)calloc(comp_size, sizeof(elina_dim_t));
	  unsigned short int i,j;
	  for(i=0; i < comp_size; i++){
		unsigned short int nvar = nca[i] - opk->dec;
		for(j=0; j < comp_size; j++){
			unsigned short int var = ca[j];
			if(dima[var-opk->dec]==nvar){
				//printf("nvar: %d %d %d");
				dim[j] = i;
				break;
			}
		}
		//dim[i] = permutation->dim[var-opk->dec];
		
	  }
	  if(oak->C){
	   	poly[k2]->C = opt_matrix_permute_dimensions(opk,destructive,oak->C,dim);
	  }
	  if(oak->F){
		 poly[k2]->F = opt_matrix_permute_dimensions(opk,destructive,oak->F,dim);
	  }
	  cla = cla->next;
	  free(ca);
	  free(dim);
	  free(nca);
  }
  if(destructive){
	free_array_comp_list(acla);
  }
  free(nca_arr);
  #if defined(TIMING)
 	 record_timing(permute_dimension_time);
  #endif
  return op;
}

void remove_block_and_factor(opt_pk_array_t *op, comp_list_t *cl){
	comp_list_t *iter = op->acl->head;
	unsigned short int num_comp = op->acl->size;
	unsigned short int k = 0;
        while(iter!=NULL){
		if(iter==cl){
			comp_list_t * tcl = cl->next;
			remove_comp_list(op->acl,cl);
			cl = tcl;
			opt_pk_t * tpoly = op->poly[k];
			unsigned short int k1;
			for(k1=k; k1 < num_comp - 1; k1++){
				op->poly[k1] = op->poly[k1+1];
			}
			opt_poly_clear(tpoly);
			break;
		}
		else{
			iter = iter->next;
		}
		k++;
	}

}
