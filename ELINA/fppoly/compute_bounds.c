#include "compute_bounds.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include "gurobi_c.h"

expr_t * replace_input_poly_cons_in_lexpr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp){
	size_t dims = expr->size;
	size_t i,k;
	double tmp1, tmp2;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
			
	if(expr->sup_coeff[0] <0){
		mul_expr = fp->input_uexpr[k];
	}
	else if(expr->inf_coeff[0] < 0){
		mul_expr = fp->input_lexpr[k];
	}
		
	if(mul_expr!=NULL){
		if(mul_expr->size==0){
			res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
	}
		
	else{
		elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[0],expr->sup_coeff[0],fp->input_inf[k],fp->input_sup[k]);
		res = create_cst_expr(tmp1, -tmp1);
	}
	for(i=1; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
			
		expr_t * mul_expr = NULL;
		expr_t * sum_expr = NULL;
		if(expr->sup_coeff[i] <0){
			mul_expr = fp->input_uexpr[k];
		}
		else if(expr->inf_coeff[i] <0){
			mul_expr = fp->input_lexpr[k];
		}
			
		if(mul_expr!=NULL){
			if(mul_expr->size==0){
				sum_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,sum_expr);
			}	
			else if(expr->inf_coeff[i]!=0 && expr->sup_coeff[i]!=0){
				sum_expr = multiply_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_expr(pr,res,sum_expr);
			}
				//free_expr(mul_expr);
			if(sum_expr!=NULL){
				free_expr(sum_expr);
			}
		}
		else{
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			res->inf_cst = res->inf_cst + tmp1;
			res->sup_cst = res->sup_cst - tmp1;
		}
	}
		
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	return res;
}


expr_t * replace_input_poly_cons_in_uexpr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp){
	size_t dims = expr->size;
	size_t i,k;
	double tmp1, tmp2;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
			
	if(expr->sup_coeff[0] <0){
		mul_expr = fp->input_lexpr[k];
	}
	else if(expr->inf_coeff[0] < 0){
		mul_expr = fp->input_uexpr[k];
	}
		
	if(mul_expr!=NULL){
		if(mul_expr->size==0){
			res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
	}
	else{
		elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[0],expr->sup_coeff[0],fp->input_inf[k],fp->input_sup[k]);
		res = create_cst_expr(-tmp2, tmp2);
	}
                //printf("finish\n");
		//fflush(stdout);
	for(i=1; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		expr_t * mul_expr = NULL;
		expr_t * sum_expr = NULL;
		if(expr->sup_coeff[i] <0){
			mul_expr = fp->input_lexpr[k];
		}
		else if(expr->inf_coeff[i] <0){
			mul_expr = fp->input_uexpr[k];
		}
			
		if(mul_expr!=NULL){
			if(mul_expr->size==0){
				sum_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,sum_expr);
			}	
			else if(expr->inf_coeff[i]!=0 && expr->sup_coeff[i]!=0){
				sum_expr = multiply_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_expr(pr,res,sum_expr);
			}
				//free_expr(mul_expr);
			if(sum_expr!=NULL){
				free_expr(sum_expr);
			}
		}
		else{
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			res->inf_cst = res->inf_cst - tmp2;
			res->sup_cst = res->sup_cst + tmp2;
		}
	}
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	return res;
}

#ifdef GUROBI
void handle_gurobi_error(int error, GRBenv *env) {
    if (error) {
        printf("Gurobi error: %s\n", GRBgeterrormsg(env));
        exit(1);
    }
}

double substitute_spatial_gurobi(expr_t *expr, fppoly_t *fp, const int opt_sense) {
    GRBenv *env = NULL;
    GRBmodel *model = NULL;

    int error;

    error = GRBemptyenv(&env);
    handle_gurobi_error(error, env);
    error = GRBsetintparam(env, "OutputFlag", 0);
    handle_gurobi_error(error, env);
    error = GRBsetintparam(env, "NumericFocus", 2);
    handle_gurobi_error(error, env);
    error = GRBstartenv(env);
    handle_gurobi_error(error, env);

    double *lb, *ub, *obj;
    const size_t dims = expr->size;
    const size_t numvars = 3 * dims;

    lb = malloc(numvars * sizeof(double));
    ub = malloc(numvars * sizeof(double));
    obj = malloc(numvars * sizeof(double));

    for (size_t i = 0; i < dims; ++i) {
        const size_t k = expr->type == DENSE ? i : expr->dim[i];
        lb[i] = -fp->input_inf[k];
        ub[i] = fp->input_sup[k];
        obj[i] = opt_sense == GRB_MINIMIZE ? -expr->inf_coeff[i] : expr->sup_coeff[i];

        for (size_t j = 0; j < 2; ++j) {
            const size_t l = fp->input_uexpr[k]->dim[j];
            lb[dims + 2 * i + j] = -fp->input_inf[l];
            ub[dims + 2 * i + j] = fp->input_sup[l];
            obj[dims + 2 * i + j] = 0;
        }
    }

    error = GRBnewmodel(env, &model, NULL, numvars, obj, lb, ub, NULL, NULL);
    handle_gurobi_error(error, env);
    error = GRBsetintattr(model, "ModelSense", opt_sense);
    handle_gurobi_error(error, env);

    for (size_t i = 0; i < dims; ++i) {
        const size_t k = expr->type == DENSE ? i : expr->dim[i];

        int ind[] = {i, dims + 2 * i, dims + 2 * i + 1};

        double lb_val[] = {
            -1, -fp->input_lexpr[k]->inf_coeff[0], -fp->input_lexpr[k]->inf_coeff[1]
        };
        error = GRBaddconstr(model, 3, ind, lb_val, GRB_LESS_EQUAL, fp->input_lexpr[k]->inf_cst, NULL);
        handle_gurobi_error(error, env);

        double ub_val[] = {
            1, -fp->input_uexpr[k]->sup_coeff[0], -fp->input_uexpr[k]->sup_coeff[1]
        };
        error = GRBaddconstr(model, 3, ind, ub_val, GRB_LESS_EQUAL, fp->input_uexpr[k]->sup_cst, NULL);
        handle_gurobi_error(error, env);
    }

    size_t idx, nbr, s_idx, s_nbr;
    const size_t num_pixels = fp->num_pixels;

    for (size_t i = 0; i < fp->spatial_size; ++i) {
        idx = fp->spatial_indices[i];
        nbr = fp->spatial_neighbors[i];

        if (expr->type == DENSE) {
            s_idx = idx;
            s_nbr = nbr;
        } else {
            s_idx = s_nbr = num_pixels;

            for (size_t j = 0; j < dims; ++j) {
                if (expr->dim[j] == idx) {
                    s_idx = j;
                }
                if (expr->dim[j] == nbr) {
                    s_nbr = j;
                }
            }

            if ((s_idx == num_pixels) || (s_nbr == num_pixels)) {
                continue;
            }
        }

        int ind_x[] = {dims + 2 * s_idx, dims + 2 * s_nbr};
        int ind_y[] = {dims + 2 * s_idx + 1, dims + 2 * s_nbr + 1};
        double val[] = {1., -1.};

        error = GRBaddconstr(model, 2, ind_x, val, GRB_LESS_EQUAL, fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
        error = GRBaddconstr(model, 2, ind_y, val, GRB_LESS_EQUAL, fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
        error = GRBaddconstr(model, 2, ind_x, val, GRB_GREATER_EQUAL, -fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
        error = GRBaddconstr(model, 2, ind_y, val, GRB_GREATER_EQUAL, -fp->spatial_gamma, NULL);
        handle_gurobi_error(error, env);
    }

    int opt_status;
    double obj_val;

    error = GRBoptimize(model);
    handle_gurobi_error(error, env);
    error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &opt_status);
    handle_gurobi_error(error, env);
    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &obj_val);
    handle_gurobi_error(error, env);

    if (opt_status != GRB_OPTIMAL) {
        printf("Gurobi model status not optimal %i\n", opt_status);
        exit(1);
    }

    free(lb);
    free(ub);
    free(obj);

    GRBfreemodel(model);
    GRBfreeenv(env);

    return obj_val;
}
#endif

double compute_lb_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
//This function will directly use the concrete bounds of variables shown in the expression to compute the value
//Can handle both expression replaced in intermediate layer or the input layer.
#ifdef GUROBI
    if ((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1 && fp->spatial_size > 0) {
        return expr->inf_cst - substitute_spatial_gurobi(expr, fp, GRB_MINIMIZE);
    }
#endif
	size_t i,k;
	double tmp1, tmp2;
	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1){
		expr =  replace_input_poly_cons_in_lexpr(pr, expr, fp);
	}
	size_t dims = expr->size;
	double res_inf = expr->inf_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_inf;
	}
	for(i=0; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		if(layerno==-1){
			//Until the input layer, use the bounds for input neurons to do the ocmputation
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
		}
		else{
			//Compute using each of the variable, by passing the concrete bounds
			//printf("constant values are %f, %f\n", fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
		}
		//All the values + the bias in the expression, res_inf is the final value
		//printf("res_inf and temp1 before computation are %f, %f\n", res_inf, tmp1);
		res_inf = res_inf + tmp1;		
		//printf("res_inf after computation are %f\n", res_inf);
	}
    if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL && layerno==-1){
		free_expr(expr);
	}
 	//printf("The inf coeffs in the lexpr are %f and %f\n",expr->inf_coeff[0],expr->inf_coeff[1]);
        // printf("compute lb from expr returns %f\n",res_inf);
        //fflush(stdout);
	return res_inf;
}

double compute_ub_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
#ifdef GUROBI
    if ((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1 && fp->spatial_size > 0) {
        return expr->sup_cst + substitute_spatial_gurobi(expr, fp, GRB_MAXIMIZE);
    }
#endif
	size_t i,k;
	double tmp1, tmp2;

	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1){
		expr =  replace_input_poly_cons_in_uexpr(pr, expr, fp);
	}

	size_t dims = expr->size;
	double res_sup = expr->sup_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_sup;
	}
	for(i=0; i < dims; i++){
		//if(expr->inf_coeff[i]<0){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}		
		if(layerno==-1){
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
		}
		else{
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
		}
		res_sup = res_sup + tmp2;
			
	}
	//printf("sup: %g\n",res_sup);
	//fflush(stdout);
	if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL && layerno==-1){
		free_expr(expr);
	}
	return res_sup;
}

double get_lb_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **lexpr_ptr, int k, int original_layer, expr_t **blk_lsum_defined_over_start_layer, bool is_sum_def_over_input, bool is_greater_comp){
	expr_t * tmp_l;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *lexpr = *lexpr_ptr;
	double res = INFINITY;
	res = compute_lb_from_expr(pr,lexpr,fp,k);
	if(!is_sum_def_over_input && !is_greater_comp && fp->layers[original_layer]->is_end_layer_of_blk && (k==fp->layers[original_layer]->start_idx_in_same_blk)){
		*blk_lsum_defined_over_start_layer = copy_expr(lexpr);
	}
	tmp_l = lexpr;
	*lexpr_ptr = lexpr_replace_bounds(pr,lexpr,aux_neurons, fp->layers[k]->is_activation);
	free_expr(tmp_l);
	return res;
}

double get_ub_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **uexpr_ptr, int k, int original_layer, expr_t **blk_usum_defined_over_start_layer, bool is_sum_def_over_input, bool is_greater_comp){
	expr_t * tmp_u;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *uexpr = *uexpr_ptr;
	double res = INFINITY;
	tmp_u = uexpr;
	res = compute_ub_from_expr(pr,uexpr,fp,k);
	if(!is_sum_def_over_input && !is_greater_comp && fp->layers[original_layer]->is_end_layer_of_blk && (k==fp->layers[original_layer]->start_idx_in_same_blk)){
		*blk_usum_defined_over_start_layer = copy_expr(uexpr);
	}
	*uexpr_ptr = uexpr_replace_bounds(pr,uexpr,aux_neurons, fp->layers[k]->is_activation);
	free_expr(tmp_u);
	return res;
}

//begin of my new functions
double compute_concrete_value_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
//This function is used to compute a concrete value when preceding neurons all take one value also
#ifdef GUROBI
    if ((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1 && fp->spatial_size > 0) {
        return expr->sup_cst + substitute_spatial_gurobi(expr, fp, GRB_MAXIMIZE);
    }
#endif
	size_t i,k;
	double tmp1, tmp2;
	assert((fp->input_lexpr==NULL) && (fp->input_uexpr==NULL));
	size_t dims = expr->size;
	double res_sup = expr->sup_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_sup;
	}
	for(i=0; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}		
		if(layerno==-1){
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],-fp->input_val[k],fp->input_val[k]);
		}
		else{
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],-fp->layers[layerno]->neurons[k]->conVal,fp->layers[layerno]->neurons[k]->conVal);
		}
		res_sup = res_sup + tmp2;
			
	}
	return res_sup;	
}

expr_t * expr_replace_with_summary_over_input(fppoly_internal_t * pr, expr_t * expr, neuron_t ** neurons, bool is_lower){
	if(expr->size==0){
		return copy_cst_expr(expr);
	}	
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL ){
		return alloc_expr();
	}
	size_t num_neurons = expr->size;
	size_t i,k;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
	neuron_t * neuron_k = neurons[k];
	if(is_lower){
		if(expr->sup_coeff[0] < 0){
			mul_expr = neuron_k->summary_uexpr;
		}
		else if(expr->inf_coeff[0]<0){
			mul_expr = neuron_k->summary_lexpr;
		}
	}
	else{
		if(expr->sup_coeff[0] < 0){
			mul_expr = neuron_k->summary_lexpr;
		}
		else if(expr->inf_coeff[0]<0){
			mul_expr = neuron_k->summary_uexpr;
		}
	}	
	if(mul_expr==NULL){
		double tmp1=0.0, tmp2=0.0;
		if(expr->inf_coeff[0]!=0 || expr->sup_coeff[0]!=0){
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		double coeff[1];
		size_t dim[1];
		coeff[0] = 0;
		dim[0] = 0;
		if(is_lower){
			res = create_sparse_expr(coeff,-tmp1,dim,1);
		}
		else{
			res = create_sparse_expr(coeff,tmp2,dim,1);
		}
	}
	else if(mul_expr->size==0){	
		res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);	
	}
	else{		
		res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
	}	
	for(i=1; i < num_neurons; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		neuron_k = neurons[k];
		mul_expr = NULL;
		if(is_lower){
			if(expr->sup_coeff[i] < 0){
				mul_expr = neuron_k->summary_uexpr;
			}
			else if(expr->inf_coeff[i]<0){
				mul_expr = neuron_k->summary_lexpr;
			}
		}
		else{
			if(expr->sup_coeff[i] < 0){
				mul_expr = neuron_k->summary_lexpr;
			}
			else if(expr->inf_coeff[i]<0){
				mul_expr = neuron_k->summary_uexpr;
			}
		}
		if(expr->sup_coeff[i]==0 && expr->inf_coeff[i]==0){
			continue;
		}
		expr_t * tmp_mul_expr = NULL;
		if(expr->sup_coeff[i] < 0 || expr->inf_coeff[i]<0){
			if(mul_expr->size==0){
				tmp_mul_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,tmp_mul_expr);
				free_expr(tmp_mul_expr);
			}
			else{
				tmp_mul_expr = multiply_expr(pr, mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);	
				add_expr(pr,res,tmp_mul_expr);
				free_expr(tmp_mul_expr);		
			}
		}
		else{
			double tmp1, tmp2;
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
			if(is_lower){
				res->inf_cst = res->inf_cst + tmp1;
				res->sup_cst = res->sup_cst - tmp1;
			}
			else{
				res->inf_cst = res->inf_cst - tmp2;
				res->sup_cst = res->sup_cst + tmp2;
			}
		}
	}
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst;
	return res;
}

double get_lb_using_summary_over_input(fppoly_internal_t * pr,fppoly_t *fp, expr_t **lexpr_ptr, int k){
	expr_t * tmp_l;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *lexpr = *lexpr_ptr;
	double res = INFINITY;
	res = compute_lb_from_expr(pr,lexpr,fp,k);
	tmp_l = lexpr;
	*lexpr_ptr = expr_replace_with_summary_over_input(pr, lexpr, aux_neurons, true);
	free_expr(tmp_l);
	return res;
}

double get_ub_using_summary_over_input(fppoly_internal_t * pr,fppoly_t *fp, expr_t **uexpr_ptr, int k){
	expr_t * tmp_u;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *uexpr = *uexpr_ptr;
	double res = INFINITY;
	tmp_u = uexpr;
	res = compute_ub_from_expr(pr,uexpr,fp,k);
	*uexpr_ptr = expr_replace_with_summary_over_input(pr,uexpr,aux_neurons, false);
	free_expr(tmp_u);
	return res;
}

void print_expr(expr_t * source){
	size_t num_neurons; 
	num_neurons = source->size;
	size_t count;
        printf("Print this expr: ");
	for (count=0; count< num_neurons; count++){
		printf("%.6f ",source->sup_coeff[count]);
	}
	printf(";;;; ");
	printf("Bias: %.6f    End of print this expr\n",source->sup_cst);
	fflush(stdout);
}

expr_t * copy_empty_expr(expr_t *src){
	expr_t *dst = (expr_t *)malloc(sizeof(expr_t));
	dst->inf_coeff = (double *)malloc(src->size*sizeof(double));
	dst->sup_coeff = (double *)malloc(src->size*sizeof(double));
	size_t i;
	dst->inf_cst = 0.0;
	dst->sup_cst = 0.0; 
	dst->type = src->type;
	for(i=0; i < src->size; i++){
		dst->inf_coeff[i] = 0.0;
		dst->sup_coeff[i] = 0.0;
	}
	if(src->type==SPARSE){
		dst->dim = (size_t *)malloc(src->size*sizeof(size_t));
		for(i=0; i < src->size; i++){
			dst->dim[i] = src->dim[i];
		}
	}
	dst->size = src->size; 
	return dst;
}

double get_lb_using_prev_layer(elina_manager_t *man, fppoly_t *fp, expr_t **expr, int k, bool is_blk_segmentation)
{
	size_t i;
	expr_t *lexpr = copy_expr(*expr);
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	double res = INFINITY;
	double lb = 0;
	if (k >= 0)
	{
		if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
			lb = get_lb_using_summary_over_input(pr,fp, &lexpr, k);
		}
		else{
			lb = get_lb_using_predecessor_layer(pr, fp, &lexpr, k, 0, NULL, true, false);
		}
		res = fmin(res, lb);
	}
	else
	{
		lb = compute_lb_from_expr(pr, lexpr, fp, -1);
		res = fmin(res, lb);
	}
	free_expr(*expr);
	*expr = lexpr;
	return res;
}

double get_ub_using_prev_layer(elina_manager_t *man, fppoly_t *fp, expr_t **expr, int k, bool is_blk_segmentation)
{
	size_t i;
	expr_t *uexpr = copy_expr(*expr);
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	double res = INFINITY;
	if (k >= 0)
	{
		if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
			res = fmin(res, get_ub_using_summary_over_input(pr,fp, &uexpr, k));
		}
		else{
			res = fmin(res, get_ub_using_predecessor_layer(pr, fp, &uexpr, k, 0, NULL, true, false));
		}
	}
	else
	{
		res = fmin(res, compute_ub_from_expr(pr, uexpr, fp, -1));
	}
	free_expr(*expr);
	*expr = uexpr;
	return res;
}

//end of my new functions

double get_lb_using_previous_layers(elina_manager_t *man, fppoly_t *fp, expr_t **expr, size_t layerno, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	size_t i, count1, count2, counter, thre_count;
	int k, cur_blk, pre_blk;
	expr_t * lexpr = copy_expr(*expr);
	expr_t * blk_lsum_defined_over_start_layer = NULL;
    fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	if(fp->numlayers==layerno){
		//If this is the last layer of the network? Then the predecssor layer would be the second last layer
		k = layerno-1;
	}
	else if((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors==2)){
		//For concatenation and residual layer)
		k = layerno;
	}
	else{
		k = fp->layers[layerno]->predecessors[0]-1;
	}	
	double res = INFINITY;
	double pre_res;
	thre_count = 0;
	if(is_early_terminate && (early_termi_thre==0)){
		neuron_t ** aux_neurons = fp->layers[k]->neurons;
		double res = compute_lb_from_expr(pr,lexpr,fp,k);
		if(*expr){
			free_expr(*expr);
			*expr = NULL;
		}
		*expr = lexpr;
		return res;
	}
	while(k >=0){
		if(fp->layers[k]->num_predecessors==2 && !is_blk_segmentation){
			thre_count = thre_count + 1;
			expr_t * lexpr_copy = copy_expr(lexpr);
			lexpr_copy->inf_cst = 0;
			lexpr_copy->sup_cst = 0;
			size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(k,sizeof(char));
			int iter = fp->layers[predecessor1]->predecessors[0]-1;
			while(iter>=0){
				predecessor_map[iter] = 1;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  fp->layers[predecessor2]->predecessors[0]-1;
			int common_predecessor = 0;
			while(iter>=0){
				if(predecessor_map[iter] == 1){
					common_predecessor = iter;
					break;
				}
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter = predecessor1;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_lb_using_predecessor_layer(pr,fp, &lexpr,  iter, layerno, &blk_lsum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_lb_using_predecessor_layer(pr,fp, &lexpr_copy,  iter, layerno, &blk_lsum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			free(predecessor_map);
			add_expr(pr,lexpr,lexpr_copy);	
			free_expr(lexpr_copy);
			k = common_predecessor;	
		}
		else if(fp->layers[k]->num_predecessors==2 && (layerno==k) && is_blk_segmentation){
			//need to generate block summarization for this layerno
			expr_t * lexpr_copy = copy_expr(lexpr);
			lexpr_copy->inf_cst = 0;
			lexpr_copy->sup_cst = 0;
			size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(k,sizeof(char));
			// Assume no nested residual layers
			int iter = fp->layers[predecessor1]->predecessors[0]-1;
			while(iter>=0){
				predecessor_map[iter] = 1;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  fp->layers[predecessor2]->predecessors[0]-1;
			int common_predecessor = 0;
			while(iter>=0){
				if(predecessor_map[iter] == 1){
					common_predecessor = iter;
					break;
				}
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter = predecessor1;
			while(iter!=common_predecessor){
				// set the flag to be true because during this process, blk_sum is not going to be generated
				thre_count = thre_count + 1;
				get_lb_using_predecessor_layer(pr,fp, &lexpr,  iter, layerno, &blk_lsum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_lb_using_predecessor_layer(pr,fp, &lexpr_copy,  iter, layerno, &blk_lsum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			free(predecessor_map);
			add_expr(pr,lexpr,lexpr_copy);
			if(!is_sum_def_over_input && (common_predecessor==fp->layers[layerno]->start_idx_in_same_blk)){
				if(blk_lsum_defined_over_start_layer)
					free_expr(blk_lsum_defined_over_start_layer);
				blk_lsum_defined_over_start_layer = copy_expr(lexpr);
			}
			free_expr(lexpr_copy);
			k = common_predecessor;
		}
		else if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
			res= fmin(res,get_lb_using_summary_over_input(pr,fp, &lexpr, k));
			thre_count = thre_count + 1;
			if(is_sum_def_over_input)
				k = -1; //directly jump to input layer
			else
				k = fp->layers[k]->start_idx_in_same_blk; //jump to the start layer of this block
		}
		else{
			res= fmin(res,get_lb_using_predecessor_layer(pr,fp, &lexpr, k, layerno, &blk_lsum_defined_over_start_layer, is_sum_def_over_input, (fp->numlayers==layerno)));
			k = fp->layers[k]->predecessors[0]-1;
			thre_count = thre_count + 1; //one step for normal back-substitution
		}
		if(is_early_terminate && thre_count >= early_termi_thre){
			if(is_blk_segmentation && (fp->numlayers!=layerno) && fp->layers[layerno]->is_end_layer_of_blk && is_sum_def_over_input){
				//do nothing and needs for the summary over input neuron to complete
			}
			else if(is_blk_segmentation && (fp->numlayers!=layerno) && fp->layers[layerno]->is_end_layer_of_blk && !is_sum_def_over_input && blk_lsum_defined_over_start_layer==NULL){
				//also do nothing and wait for the summary to complete
			}
			else{
				//Before early terminate, don't waste the newly generated lexpr
				res = fmin(res,compute_lb_from_expr(pr,lexpr,fp,k));
				if(*expr){
					free_expr(*expr);
					*expr = NULL;
				}
				if(is_blk_segmentation && (fp->numlayers!=layerno) && !is_sum_def_over_input && fp->layers[layerno]->start_idx_in_same_blk>=0){
					free_expr(lexpr);
					lexpr = NULL;
					*expr = blk_lsum_defined_over_start_layer;
				}
				else{
					*expr = lexpr;
					if(blk_lsum_defined_over_start_layer)
						free_expr(blk_lsum_defined_over_start_layer);
					blk_lsum_defined_over_start_layer = NULL;
				}
				//Force to return earlier in here
				return res;	
			}
		}	
	}	
	pre_res = compute_lb_from_expr(pr,lexpr,fp,-1);
	res = fmin(res,pre_res);
	if(*expr){
		free_expr(*expr);
		*expr = NULL;
	}
	if(is_blk_segmentation && !is_sum_def_over_input && (fp->numlayers!=layerno) && fp->layers[layerno]->start_idx_in_same_blk>=0){
		free_expr(lexpr);
		lexpr = NULL;
		*expr = blk_lsum_defined_over_start_layer;
	}
	else{
		*expr = lexpr;
		if(blk_lsum_defined_over_start_layer)
			free_expr(blk_lsum_defined_over_start_layer);
		blk_lsum_defined_over_start_layer = NULL;
	}
	return res;	
}   

double get_ub_using_previous_layers(elina_manager_t *man, fppoly_t *fp, expr_t **expr, size_t layerno, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	size_t i, count, count2, counter, thre_count;
	int k, cur_blk, pre_blk;
	expr_t * uexpr = copy_expr(*expr);
	expr_t * blk_usum_defined_over_start_layer = NULL;
    fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	if(fp->numlayers==layerno){
		k = layerno-1;
	}
	else if((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors==2)){
		k = layerno;
	}
	else{
		k = fp->layers[layerno]->predecessors[0]-1;
	}
	double res =INFINITY;
	thre_count = 0;
	if(is_early_terminate && (early_termi_thre==0)){
		neuron_t ** aux_neurons = fp->layers[k]->neurons;
		double res = compute_ub_from_expr(pr,uexpr,fp,k);
		if(*expr){
			free_expr(*expr);
			*expr = NULL;
		}
		*expr = uexpr;
		return res;
	}
	while(k >=0){
		if(fp->layers[k]->num_predecessors==2 && !is_blk_segmentation){
			expr_t * uexpr_copy = copy_expr(uexpr);
			uexpr_copy->inf_cst = 0;
			uexpr_copy->sup_cst = 0;
			size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(k,sizeof(char));
			int iter = fp->layers[predecessor1]->predecessors[0]-1;
			while(iter>=0){
				predecessor_map[iter] = 1;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  fp->layers[predecessor2]->predecessors[0]-1;
			int common_predecessor = 0;
			while(iter>=0){
				if(predecessor_map[iter] == 1){
					common_predecessor = iter;
					break;
				}
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter = predecessor1;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_ub_using_predecessor_layer(pr,fp, &uexpr,  iter, layerno, &blk_usum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_ub_using_predecessor_layer(pr,fp, &uexpr_copy,  iter, layerno, &blk_usum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			add_expr(pr,uexpr,uexpr_copy);
			free(predecessor_map);
			free_expr(uexpr_copy);
			k = common_predecessor;
        }	
		else if(fp->layers[k]->num_predecessors==2 && (layerno==k) && is_blk_segmentation){
			expr_t * uexpr_copy = copy_expr(uexpr);
			uexpr_copy->inf_cst = 0;
			uexpr_copy->sup_cst = 0;
			size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(k,sizeof(char));
			int iter = fp->layers[predecessor1]->predecessors[0]-1;
			while(iter>=0){
				predecessor_map[iter] = 1;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  fp->layers[predecessor2]->predecessors[0]-1;
			int common_predecessor = 0;
			while(iter>=0){
				if(predecessor_map[iter] == 1){
					common_predecessor = iter;
					break;
				}
				iter = fp->layers[iter]->predecessors[0]-1;
			}	
			iter = predecessor1;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_ub_using_predecessor_layer(pr,fp, &uexpr,  iter, layerno, &blk_usum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
				thre_count = thre_count + 1;
				get_ub_using_predecessor_layer(pr,fp, &uexpr_copy,  iter, layerno, &blk_usum_defined_over_start_layer, is_sum_def_over_input, true);
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			free(predecessor_map);
			add_expr(pr,uexpr,uexpr_copy);
			if(!is_sum_def_over_input && (common_predecessor==fp->layers[k]->start_idx_in_same_blk)){
				if(blk_usum_defined_over_start_layer)
					free_expr(blk_usum_defined_over_start_layer);
				blk_usum_defined_over_start_layer = copy_expr(uexpr);	
			}
			free_expr(uexpr_copy);
			k = common_predecessor;
		}
		else if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
			res= fmin(res,get_ub_using_summary_over_input(pr,fp, &uexpr, k));
			thre_count = thre_count + 1;
			if(is_sum_def_over_input)
				k = -1; //directly jump to input layer
			else
				k = fp->layers[k]->start_idx_in_same_blk; //jump to the start layer of this block
		}
		else{
			res= fmin(res,get_ub_using_predecessor_layer(pr,fp, &uexpr, k, layerno, &blk_usum_defined_over_start_layer, is_sum_def_over_input, (fp->numlayers==layerno)));
			k = fp->layers[k]->predecessors[0]-1;
			thre_count = thre_count + 1;
		}	
		if(is_early_terminate && thre_count >= early_termi_thre){
			if(is_blk_segmentation && (fp->numlayers!=layerno) && fp->layers[layerno]->is_end_layer_of_blk && is_sum_def_over_input){
				//do nothing and needs for the summary over input neuron to complete
			}
			else if(is_blk_segmentation && (fp->numlayers!=layerno) && fp->layers[layerno]->is_end_layer_of_blk && !is_sum_def_over_input && blk_usum_defined_over_start_layer==NULL){
				//also do nothing and wait for the block summary to complete
			}
			else{
				//Before early terminate, don't waste the newly generated lexpr
				res = fmin(res,compute_ub_from_expr(pr,uexpr,fp,k));
				if(*expr){
					free_expr(*expr);
					*expr = NULL;
				}
				if(is_blk_segmentation && !is_sum_def_over_input && (fp->numlayers!=layerno) && fp->layers[layerno]->start_idx_in_same_blk>=0){
					free_expr(uexpr);
					uexpr = NULL;
					*expr = blk_usum_defined_over_start_layer;
				}
				else{
					*expr = uexpr;
					if(blk_usum_defined_over_start_layer)
						free_expr(blk_usum_defined_over_start_layer);
					blk_usum_defined_over_start_layer = NULL;
				}
				//Force to return earlier in here
				return res;	
			}
		}	
	}	
	res = fmin(res,compute_ub_from_expr(pr,uexpr,fp,-1)); 
	if(*expr){
		free_expr(*expr);
		*expr = NULL;
	}
	if(!is_sum_def_over_input && is_blk_segmentation && (fp->numlayers!=layerno) && fp->layers[layerno]->start_idx_in_same_blk>=0){
		free_expr(uexpr);
		uexpr = NULL;
		*expr = blk_usum_defined_over_start_layer;
	}
	else{
		*expr = uexpr;
		if(blk_usum_defined_over_start_layer)
			free_expr(blk_usum_defined_over_start_layer);
		blk_usum_defined_over_start_layer = NULL;
	}
	return res;
}
