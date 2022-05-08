#include "relu_approx.h"

expr_t * create_relu_expr(neuron_t *out_neuron, neuron_t *in_neuron, size_t i, bool use_default_heuristics, bool is_lower, bool is_refinement){
	expr_t * res = alloc_expr();  
	res->inf_coeff = (double *)malloc(sizeof(double));
	res->sup_coeff = (double *)malloc(sizeof(double));
	res->inf_cst = 0.0;
	res->sup_cst = 0.0;
	res->type = SPARSE;
	res->size = 1;  
	res->dim = (size_t*)malloc(sizeof(size_t));
	res->dim[0] = i;
	double lb = in_neuron->lb;
	double ub = in_neuron->ub;
	double width = ub + lb;
	double lambda_inf = -ub/width;
	double lambda_sup = ub/width;
	
	if(ub<=0){
		res->inf_coeff[0] = 0.0;
		res->sup_coeff[0] = 0.0;
	}
	else if(lb<0){
		res->inf_coeff[0] = -1.0;
		res->sup_coeff[0] = 1.0;
	}
		
	else if(is_lower){
		double area1 = 0.5*ub*width;
		double area2 = 0.5*lb*width;
		if(use_default_heuristics){
			if(area1 < area2){
				res->inf_coeff[0] = 0.0;
				res->sup_coeff[0] = 0.0;
			}
			else{
				res->inf_coeff[0] = -1.0;
				res->sup_coeff[0] = 1.0;
			}
		}
		else{
				res->inf_coeff[0] = 0.0;
				res->sup_coeff[0] = 0.0;
		}
	}
	else{
		//Only upper bound contains bias cst
		double mu_inf = lambda_inf*lb;
		double mu_sup = lambda_sup*lb;
		res->inf_coeff[0] = lambda_inf;
		res->sup_coeff[0] = lambda_sup;
		res->inf_cst = mu_inf;
		res->sup_cst = mu_sup;
	}
	return res;
}

void handle_relu_layer(elina_manager_t *man, elina_abstract0_t* element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool use_default_heuristics, bool is_blk_segmentation, int blk_size, bool is_residual, bool is_refinement){
	assert(num_predecessors==1);
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	fppoly_add_new_layer(fp, num_neurons, predecessors, num_predecessors,true);
	neuron_t **out_neurons = fp->layers[numlayers]->neurons;
	int k = predecessors[0]-1;
	layer_t *predecessor_layer = fp->layers[k];
	//the session to mark start-of-block layer
	if(!is_residual){
		if(is_blk_segmentation && predecessor_layer->is_end_layer_of_blk){
			//if the preceding layer is end-layer, this will be start layer
			fp->layers[numlayers]->is_start_layer_of_blk = true;
		}
		if(is_blk_segmentation){
			if(predecessor_layer->is_end_layer_of_blk){
				fp->layers[numlayers]->start_idx_in_same_blk = numlayers;
			}else{
				fp->layers[numlayers]->start_idx_in_same_blk = predecessor_layer->start_idx_in_same_blk;
			}	
		}
	}
	else{
		if(fp->layers[numlayers]->is_start_layer_of_blk && !fp->layers[numlayers]->is_end_layer_of_blk)
			fp->layers[numlayers]->start_idx_in_same_blk = numlayers;
	}

	neuron_t **in_neurons = fp->layers[k]->neurons;
	size_t i;
	for(i=0; i < num_neurons; i++){
		out_neurons[i]->lb = -fmax(0.0, -in_neurons[i]->lb);
		out_neurons[i]->ub = fmax(0,in_neurons[i]->ub);
		out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, true, is_refinement);
		out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, use_default_heuristics, false, is_refinement);
		// printf("%.4f, %.4f\n",-out_neurons[i]->lb, out_neurons[i]->ub);
		// expr_print(out_neurons[i]->lexpr);
		// expr_print(out_neurons[i]->uexpr);
	}

}
