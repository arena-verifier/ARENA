#include "backsubstitute.h"
#include <time.h>
#include <math.h>

int max_integer(int num1, int num2)
{
    return (num1 > num2 ) ? num1 : num2;
}

int min_integer(int num1, int num2)
{
    return (num1 < num2 ) ? num1 : num2;
}

void * free_expr_for_prior_layers_in_same_block(fppoly_t *fp, fppoly_internal_t * pr, size_t layerno, bool is_residual, bool is_sum_def_over_input){
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return NULL;
	}
	layer_t * layer = fp->layers[layerno];
	size_t i;
	// Need to free according to two predecessor lines for residual network, otherwise will miss some predecessors.
	int predecessor1 = layer->predecessors[0]-1;
	int predecessor2 = -1;
	if(is_residual)
		predecessor2 = layer->predecessors[1]-1;
	if(is_sum_def_over_input){
		//All the layers in the group, including the previous end layer could all be released
		bool flag_free = true;
		while ((predecessor1 >= 0) && flag_free)
		{
			/* For those layers within the same block and are NOT input layer */
			// printf("Free in advance for layer %d\n", predecessor1);
			layer_t *predecessor_layer = fp->layers[predecessor1];
			if(predecessor_layer->is_end_layer_of_blk){
				flag_free = false;
			}
			size_t dims = predecessor_layer->dims;
			for(i=0; i < dims; i++){
				if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr!=predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->uexpr);
					predecessor_layer->neurons[i]->uexpr = NULL;
				}else if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr==predecessor_layer->neurons[i]->lexpr){
					predecessor_layer->neurons[i]->uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->lexpr);
					predecessor_layer->neurons[i]->lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_lexpr){
					free_expr(predecessor_layer->neurons[i]->summary_lexpr);
					predecessor_layer->neurons[i]->summary_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_uexpr){
					free_expr(predecessor_layer->neurons[i]->summary_uexpr);
					predecessor_layer->neurons[i]->summary_uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_lexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_lexpr);
					predecessor_layer->neurons[i]->backsubstituted_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_uexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_uexpr);
					predecessor_layer->neurons[i]->backsubstituted_uexpr = NULL;
				}
			}
			predecessor1 = predecessor_layer->predecessors[0]-1;		
		}
		flag_free = true;
		while ((predecessor2 >= 0) && flag_free)
		{
			// printf("Free in advance for layer %d\n", predecessor2);
			layer_t *predecessor_layer = fp->layers[predecessor2];
			if(predecessor_layer->is_end_layer_of_blk){
				flag_free = false;
			}
			size_t dims = predecessor_layer->dims;
			for(i=0; i < dims; i++){
				if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr!=predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->uexpr);
					predecessor_layer->neurons[i]->uexpr = NULL;
				}else if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr==predecessor_layer->neurons[i]->lexpr){
					predecessor_layer->neurons[i]->uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->lexpr);
					predecessor_layer->neurons[i]->lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_lexpr){
					free_expr(predecessor_layer->neurons[i]->summary_lexpr);
					predecessor_layer->neurons[i]->summary_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_uexpr){
					free_expr(predecessor_layer->neurons[i]->summary_uexpr);
					predecessor_layer->neurons[i]->summary_uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_lexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_lexpr);
					predecessor_layer->neurons[i]->backsubstituted_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_uexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_uexpr);
					predecessor_layer->neurons[i]->backsubstituted_uexpr = NULL;
				}
			}
			predecessor2 = predecessor_layer->predecessors[0]-1;		
		}
	}
	else{
		//All the start layer should also be keeped
		while ((predecessor1 >= 0) && !fp->layers[predecessor1]->is_start_layer_of_blk)
		{
			/* For those layers within the same block and are NOT input layer */
			// printf("Free in advance for layer %d\n", predecessor1);
			layer_t *predecessor_layer = fp->layers[predecessor1];
			size_t dims = predecessor_layer->dims;
			for(i=0; i < dims; i++){
				if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr!=predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->uexpr);
					predecessor_layer->neurons[i]->uexpr = NULL;
				}else if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr==predecessor_layer->neurons[i]->lexpr){
					predecessor_layer->neurons[i]->uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->lexpr);
					predecessor_layer->neurons[i]->lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_lexpr){
					free_expr(predecessor_layer->neurons[i]->summary_lexpr);
					predecessor_layer->neurons[i]->summary_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_uexpr){
					free_expr(predecessor_layer->neurons[i]->summary_uexpr);
					predecessor_layer->neurons[i]->summary_uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_lexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_lexpr);
					predecessor_layer->neurons[i]->backsubstituted_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_uexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_uexpr);
					predecessor_layer->neurons[i]->backsubstituted_uexpr = NULL;
				}
			}
			predecessor1 = predecessor_layer->predecessors[0]-1;		
		}
		while ((predecessor2 >= 0) && !fp->layers[predecessor2]->is_start_layer_of_blk)
		{
			// printf("Free in advance for layer %d\n", predecessor2);
			layer_t *predecessor_layer = fp->layers[predecessor2];
			size_t dims = predecessor_layer->dims;
			for(i=0; i < dims; i++){
				if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr!=predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->uexpr);
					predecessor_layer->neurons[i]->uexpr = NULL;
				}else if(predecessor_layer->neurons[i]->uexpr && predecessor_layer->neurons[i]->uexpr==predecessor_layer->neurons[i]->lexpr){
					predecessor_layer->neurons[i]->uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->lexpr){
					free_expr(predecessor_layer->neurons[i]->lexpr);
					predecessor_layer->neurons[i]->lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_lexpr){
					free_expr(predecessor_layer->neurons[i]->summary_lexpr);
					predecessor_layer->neurons[i]->summary_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->summary_uexpr){
					free_expr(predecessor_layer->neurons[i]->summary_uexpr);
					predecessor_layer->neurons[i]->summary_uexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_lexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_lexpr);
					predecessor_layer->neurons[i]->backsubstituted_lexpr = NULL;
				}
				if(predecessor_layer->neurons[i]->backsubstituted_uexpr){
					free_expr(predecessor_layer->neurons[i]->backsubstituted_uexpr);
					predecessor_layer->neurons[i]->backsubstituted_uexpr = NULL;
				}
			}
			predecessor2 = predecessor_layer->predecessors[0]-1;		
		}
	}
	return NULL;
}

void * update_state_using_previous_layers(void *args){
	//testing of connecting using vs
	nn_thread_t * data = (nn_thread_t *)args;
	elina_manager_t * man = data->man;
	fppoly_t *fp = data->fp;
	bool layer_by_layer = data->layer_by_layer;
	bool is_residual = data->is_residual;
	bool is_blk_segmentation = data->is_blk_segmentation;
	int blk_size = data->blk_size;
	bool is_early_terminate = data->is_early_terminate;
	int early_termi_thre = data->early_termi_thre;
	bool is_sum_def_over_input = data->is_sum_def_over_input;
	bool is_refinement = data->is_refinement;
	fppoly_internal_t * pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	int k;
	
	neuron_t ** out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	for(i=idx_start; i < idx_end; i++){
		bool already_computed= false;
		expr_t *lexpr = copy_expr(out_neurons[i]->lexpr);
		expr_t *uexpr = copy_expr(out_neurons[i]->uexpr);
		// expr_print(lexpr);
		out_neurons[i]->lb = get_lb_using_previous_layers(man, fp, &lexpr, layerno, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
		out_neurons[i]->ub = get_ub_using_previous_layers(man, fp, &uexpr, layerno, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
		if(!layer_by_layer && is_blk_segmentation && fp->layers[layerno]->is_end_layer_of_blk){
			out_neurons[i]->summary_lexpr = lexpr;
			out_neurons[i]->summary_uexpr = uexpr;
		}else{
			out_neurons[i]->summary_lexpr = NULL;
			out_neurons[i]->summary_uexpr = NULL;
			if(lexpr)
				free_expr(lexpr);
			if(uexpr)
				free_expr(uexpr);
		}
	}
	return NULL;
}

void update_state_using_previous_layers_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno, bool layer_by_layer,  bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
  	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	fppoly_internal_t * pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t num_out_neurons = fp->layers[layerno]->dims;
	if(!layer_by_layer && is_blk_segmentation){
		//the session to mark the end-of-block/start-of-block layer
		if(is_residual && (fp->layers[layerno]->num_predecessors==2)){
			//identify the start layer for residual block
			size_t predecessor1 = fp->layers[layerno]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[layerno]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(layerno,sizeof(char));
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
			free(predecessor_map);
			fp->layers[common_predecessor]->is_start_layer_of_blk = true;
			fp->layers[layerno]->start_idx_in_same_blk = common_predecessor;
			iter = predecessor1;
			while(iter!=common_predecessor){
				fp->layers[iter]->start_idx_in_same_blk = common_predecessor;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
				fp->layers[iter]->start_idx_in_same_blk = common_predecessor;
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			//set the end layer 
			fp->layers[layerno]->is_end_layer_of_blk = true;
		}
		else if(!is_residual && ((layerno+2)%(2*blk_size) == 0)){
			fp->layers[layerno]->is_end_layer_of_blk = true;
		}
	}
	//the session to set the start_idx_in_same_blk
	if(!layer_by_layer && is_blk_segmentation){
		if(!is_residual){
			if(layerno==0){
				fp->layers[layerno]->start_idx_in_same_blk = -1;
			}
			else{
				int predecessor = fp->layers[layerno]->predecessors[0]-1;
				layer_t *predecessor_layer = fp->layers[predecessor];
				if(predecessor_layer->is_end_layer_of_blk){
					fp->layers[layerno]->is_start_layer_of_blk = true;
					fp->layers[layerno]->start_idx_in_same_blk = layerno;
				}else{
					fp->layers[layerno]->start_idx_in_same_blk = predecessor_layer->start_idx_in_same_blk;
				}
			}
		}
	}
	// if(is_blk_segmentation && is_refinement){
	// 	// only apply modular for refinement process, not for original deeppoly execution
	// 	is_blk_segmentation = false;
	// }
	size_t i;
	int k = fp->layers[layerno]->predecessors[0] - 1;;
	//Set the global ReLU nodes in here
	if(num_out_neurons < NUM_THREADS){
		for (i = 0; i < num_out_neurons; i++){
			args[i].start = i; 
			args[i].end = i+1;   
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].linexpr0 = NULL;
			args[i].res = NULL;
	    	args[i].layer_by_layer = layer_by_layer;
	    	args[i].is_residual = is_residual;
	    	args[i].is_blk_segmentation = is_blk_segmentation;
	    	args[i].blk_size = blk_size;
			args[i].is_early_terminate = is_early_terminate;
	    	args[i].early_termi_thre = early_termi_thre;
	    	args[i].is_sum_def_over_input = is_sum_def_over_input;
			args[i].is_refinement = is_refinement;
			pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
			
	  	}
		for (i = 0; i < num_out_neurons; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	else{
		size_t idx_start = 0;
		size_t idx_n = num_out_neurons / NUM_THREADS;
		size_t idx_end = idx_start + idx_n;
	  	for (i = 0; i < NUM_THREADS; i++){
	    		args[i].start = idx_start; 
	    		args[i].end = idx_end;   
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].linexpr0 = NULL;
			args[i].res = NULL;
			args[i].layer_by_layer = layer_by_layer;
			args[i].is_residual = is_residual;
	    	args[i].is_blk_segmentation = is_blk_segmentation;
	    	args[i].blk_size = blk_size;
			args[i].is_early_terminate = is_early_terminate;
	    	args[i].early_termi_thre = early_termi_thre;
	    	args[i].is_sum_def_over_input = is_sum_def_over_input;
			args[i].is_refinement = is_refinement;
	    	pthread_create(&threads[i], NULL,update_state_using_previous_layers, (void*)&args[i]);
			idx_start = idx_end;
			idx_end = idx_start + idx_n;
	    		if(idx_end>num_out_neurons){
				idx_end = num_out_neurons;
			}
			if((i==NUM_THREADS-2)){
				idx_end = num_out_neurons;
				
			}
	  	}
		for (i = 0; i < NUM_THREADS; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	if(!layer_by_layer && is_blk_segmentation && fp->layers[layerno]->is_end_layer_of_blk)
		free_expr_for_prior_layers_in_same_block(fp, pr, layerno, is_residual, is_sum_def_over_input);
}

//layer-by-layer analysis coding
void * update_state_layer_by_layer_lb(void *args)
{
	nn_thread_t *data = (nn_thread_t *)args;
	elina_manager_t *man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	int k = data->k;
	bool is_blk_segmentation = data->is_blk_segmentation;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	neuron_t **out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	for (i = idx_start; i < idx_end; i++)
	{
		bool already_computed = false;
		//evaluate constraint defined over k, and further back-sub to k-1
		if(is_blk_segmentation && fp->layers[layerno]->is_end_layer_of_blk && (k==fp->layers[layerno]->start_idx_in_same_blk)){
			out_neurons[i]->summary_lexpr = copy_expr(out_neurons[i]->backsubstituted_lexpr);
		}
		out_neurons[i]->lb = fmin(out_neurons[i]->lb, get_lb_using_prev_layer(man, fp, &out_neurons[i]->backsubstituted_lexpr, k, is_blk_segmentation));
		// printf("lower bound is %.6f\n", out_neurons[i]->lb);
	}
	return NULL;
}

void *update_state_layer_by_layer_ub(void *args)
{
	nn_thread_t *data = (nn_thread_t *)args;
	elina_manager_t *man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	int k = data->k;
	bool is_blk_segmentation = data->is_blk_segmentation;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	neuron_t **out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	for (i = idx_start; i < idx_end; i++)
	{
		bool already_computed = false;
		if(is_blk_segmentation && fp->layers[layerno]->is_end_layer_of_blk && (k==fp->layers[layerno]->start_idx_in_same_blk)){
			out_neurons[i]->summary_uexpr = copy_expr(out_neurons[i]->backsubstituted_uexpr);
		}
		//evaluate constraint defined over k, and further back-sub to k-1
		out_neurons[i]->ub = fmin(out_neurons[i]->ub, get_ub_using_prev_layer(man, fp, &out_neurons[i]->backsubstituted_uexpr, k, is_blk_segmentation));
		// printf("upper bound is %.6f\n", out_neurons[i]->ub);
	}
	return NULL;
}

void update_state_layer_by_layer_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno, bool layer_by_layer,  bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement)
{
	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t num_out_neurons = fp->layers[layerno]->dims;
	if(is_blk_segmentation){
		//the session to mark the end-of-block/start-of-block layer
		if(is_residual && (fp->layers[layerno]->num_predecessors==2)){
			//identify the start layer for residual block
			size_t predecessor1 = fp->layers[layerno]->predecessors[0]-1;
			size_t predecessor2 = fp->layers[layerno]->predecessors[1]-1;
			char * predecessor_map = (char *)calloc(layerno,sizeof(char));
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
			free(predecessor_map);
			fp->layers[common_predecessor]->is_start_layer_of_blk = true;
			fp->layers[layerno]->start_idx_in_same_blk = common_predecessor;
			iter = predecessor1;
			while(iter!=common_predecessor){
				fp->layers[iter]->start_idx_in_same_blk = common_predecessor;
				iter = fp->layers[iter]->predecessors[0]-1;
			}
			iter =  predecessor2;
			while(iter!=common_predecessor){
				fp->layers[iter]->start_idx_in_same_blk = common_predecessor;
				iter = fp->layers[iter]->predecessors[0]-1;					
			}
			//set the end layer 
			fp->layers[layerno]->is_end_layer_of_blk = true;
		}
		else if(!is_residual && ((layerno+2)%(2*blk_size) == 0)){
			fp->layers[layerno]->is_end_layer_of_blk = true;
		}
	}
	//the session to set the start_idx_in_same_blk
	if(is_blk_segmentation){
		if(!is_residual){
			if(layerno==0){
				fp->layers[layerno]->start_idx_in_same_blk = -1;
			}
			else{
				int predecessor = fp->layers[layerno]->predecessors[0]-1;
				layer_t *predecessor_layer = fp->layers[predecessor];
				if(predecessor_layer->is_end_layer_of_blk){
					fp->layers[layerno]->is_start_layer_of_blk = true;
					fp->layers[layerno]->start_idx_in_same_blk = layerno;
				}else{
					fp->layers[layerno]->start_idx_in_same_blk = predecessor_layer->start_idx_in_same_blk;
				}
			}
		}
	}
	// printf("set up the block info\n");
	if(is_blk_segmentation && is_refinement){
		// only apply modular for refinement process, not for original deeppoly execution
		is_blk_segmentation = false;
	}
	size_t i;
	int k;
	if (fp->numlayers == layerno)
	{
		k = layerno - 1;
	}
	else if ((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors == 2))
	{
		k = layerno;
	}
	else
	{
		k = fp->layers[layerno]->predecessors[0] - 1;
	}
	while (k >= -1)
	{
		if (num_out_neurons < NUM_THREADS)
		{
			for (i = 0; i < num_out_neurons; i++)
			{
				args[i].start = i;
				args[i].end = i + 1;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_lb, (void *)&args[i]);
			}
			for (i = 0; i < num_out_neurons; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		else
		{
			size_t idx_start = 0;
			size_t idx_n = num_out_neurons / NUM_THREADS;
			size_t idx_end = idx_start + idx_n;
			for (i = 0; i < NUM_THREADS; i++)
			{
				args[i].start = idx_start;
				args[i].end = idx_end;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_lb, (void *)&args[i]);
				idx_start = idx_end;
				idx_end = idx_start + idx_n;
				if (idx_end > num_out_neurons)
				{
					idx_end = num_out_neurons;
				}
				if ((i == NUM_THREADS - 2))
				{
					idx_end = num_out_neurons;
				}
			}
			for (i = 0; i < NUM_THREADS; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		if (num_out_neurons < NUM_THREADS)
		{
			for (i = 0; i < num_out_neurons; i++)
			{
				args[i].start = i;
				args[i].end = i + 1;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_ub, (void *)&args[i]);
			}
			for (i = 0; i < num_out_neurons; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		else
		{
			size_t idx_start = 0;
			size_t idx_n = num_out_neurons / NUM_THREADS;
			size_t idx_end = idx_start + idx_n;
			for (i = 0; i < NUM_THREADS; i++)
			{
				args[i].start = idx_start;
				args[i].end = idx_end;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_ub, (void *)&args[i]);
				idx_start = idx_end;
				idx_end = idx_start + idx_n;
				if (idx_end > num_out_neurons)
				{
					idx_end = num_out_neurons;
				}
				if ((i == NUM_THREADS - 2))
				{
					idx_end = num_out_neurons;
				}
			}
			for (i = 0; i < NUM_THREADS; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		if (k < 0)
			break;
		if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
			k = fp->layers[k]->start_idx_in_same_blk;
		}
		else{
			k = fp->layers[k]->predecessors[0] - 1;
		}
		
	}
}

void update_state_layer_by_layer_parallel_until_certain_layer(elina_manager_t *man, fppoly_t *fp, size_t layerno, bool is_blk_segmentation, int block_start_layer)
{
	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t num_out_neurons = fp->layers[layerno]->dims;
	size_t i;
	int k;
	// printf("you do have the block execution right %d?\n", is_blk_segmentation);
	// The back-substitution stops at certain layer, which is block_start_layer
	if (fp->numlayers == layerno)
	{
		k = layerno - 1;
	}
	else if ((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors == 2))
	{
		k = layerno;
	}
	else
	{
		k = fp->layers[layerno]->predecessors[0] - 1;
	}
	while (k >= block_start_layer)
	{
		if (num_out_neurons < NUM_THREADS)
		{
			for (i = 0; i < num_out_neurons; i++)
			{
				args[i].start = i;
				args[i].end = i + 1;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_lb, (void *)&args[i]);
			}
			for (i = 0; i < num_out_neurons; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		else
		{
			size_t idx_start = 0;
			size_t idx_n = num_out_neurons / NUM_THREADS;
			size_t idx_end = idx_start + idx_n;
			for (i = 0; i < NUM_THREADS; i++)
			{
				args[i].start = idx_start;
				args[i].end = idx_end;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_lb, (void *)&args[i]);
				idx_start = idx_end;
				idx_end = idx_start + idx_n;
				if (idx_end > num_out_neurons)
				{
					idx_end = num_out_neurons;
				}
				if ((i == NUM_THREADS - 2))
				{
					idx_end = num_out_neurons;
				}
			}
			for (i = 0; i < NUM_THREADS; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		if (num_out_neurons < NUM_THREADS)
		{
			for (i = 0; i < num_out_neurons; i++)
			{
				args[i].start = i;
				args[i].end = i + 1;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_ub, (void *)&args[i]);
			}
			for (i = 0; i < num_out_neurons; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		else
		{
			size_t idx_start = 0;
			size_t idx_n = num_out_neurons / NUM_THREADS;
			size_t idx_end = idx_start + idx_n;
			for (i = 0; i < NUM_THREADS; i++)
			{
				args[i].start = idx_start;
				args[i].end = idx_end;
				args[i].man = man;
				args[i].fp = fp;
				args[i].layerno = layerno;
				args[i].is_blk_segmentation = is_blk_segmentation;
				args[i].k = k;
				args[i].linexpr0 = NULL;
				args[i].res = NULL;
				pthread_create(&threads[i], NULL, update_state_layer_by_layer_ub, (void *)&args[i]);
				idx_start = idx_end;
				idx_end = idx_start + idx_n;
				if (idx_end > num_out_neurons)
				{
					idx_end = num_out_neurons;
				}
				if ((i == NUM_THREADS - 2))
				{
					idx_end = num_out_neurons;
				}
			}
			for (i = 0; i < NUM_THREADS; i = i + 1)
			{
				pthread_join(threads[i], NULL);
			}
		}
		if (k < 0)
			break;
		if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
			k = fp->layers[k]->start_idx_in_same_blk;
		}
		else{
			k = fp->layers[k]->predecessors[0] - 1;
		}
	}
}

void * update_layer_for_concrete_img(void *args)
{
	nn_thread_t *data = (nn_thread_t *)args;
	elina_manager_t *man = data->man;
	fppoly_t *fp = data->fp;
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t layerno = data->layerno;
	int k = data->k;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	size_t i;
	neuron_t **out_neurons = fp->layers[layerno]->neurons;
	size_t num_out_neurons = fp->layers[layerno]->dims;
	for (i = idx_start; i < idx_end; i++)
	{
		bool already_computed = false;
		out_neurons[i]->conVal = fmin(out_neurons[i]->conVal, compute_concrete_value_from_expr(pr, out_neurons[i]->lexpr, fp, k));
		// printf("lower bound is %.6f\n", out_neurons[i]->lb);
	}
	return NULL;
}

void update_layer_for_concrete_img_parallel(elina_manager_t *man, fppoly_t *fp, size_t layerno, int defined_layer)
{
	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t num_out_neurons = fp->layers[layerno]->dims;
	size_t i;
	int k;
	// printf("you do have the block execution right %d?\n", is_blk_segmentation);
	// The back-substitution stops at certain layer, which is block_start_layer
	if (fp->numlayers == layerno)
	{
		k = layerno - 1;
	}
	else if ((fp->layers[layerno]->is_concat == true) || (fp->layers[layerno]->num_predecessors == 2))
	{
		k = layerno;
	}
	else
	{
		k = fp->layers[layerno]->predecessors[0] - 1;
	}
	if (num_out_neurons < NUM_THREADS)
	{
		for (i = 0; i < num_out_neurons; i++)
		{
			args[i].start = i;
			args[i].end = i + 1;
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].k = k;
			args[i].linexpr0 = NULL;
			args[i].res = NULL;
			pthread_create(&threads[i], NULL, update_layer_for_concrete_img, (void *)&args[i]);
		}
		for (i = 0; i < num_out_neurons; i = i + 1)
		{
			pthread_join(threads[i], NULL);
		}
	}
	else
	{
		size_t idx_start = 0;
		size_t idx_n = num_out_neurons / NUM_THREADS;
		size_t idx_end = idx_start + idx_n;
		for (i = 0; i < NUM_THREADS; i++)
		{
			args[i].start = idx_start;
			args[i].end = idx_end;
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].k = k;
			args[i].linexpr0 = NULL;
			args[i].res = NULL;
			pthread_create(&threads[i], NULL, update_layer_for_concrete_img, (void *)&args[i]);
			idx_start = idx_end;
			idx_end = idx_start + idx_n;
			if (idx_end > num_out_neurons)
			{
				idx_end = num_out_neurons;
			}
			if ((i == NUM_THREADS - 2))
			{
				idx_end = num_out_neurons;
			}
		}
		for (i = 0; i < NUM_THREADS; i = i + 1)
		{
			pthread_join(threads[i], NULL);
		}
	}
}


// end of layer-by-layer analysis coding

/* end of the code */