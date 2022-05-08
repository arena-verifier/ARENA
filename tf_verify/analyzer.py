'''
@author: Adrian Hoffmann
'''

from doctest import FAIL_FAST
from pickle import FALSE
from elina_abstract0 import *
from elina_manager import *
from deeppoly_nodes import *
from krelu import *
from functools import reduce
from ai_milp import milp_callback
import gc

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pool_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, cons) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(cons) for l,w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(cons) for l,w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradiant(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper


class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing = False, layer_by_layer = False, is_residual = False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, is_refinement = False, REFINE_MAX_ITER = 5):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono', 'refinezono' or 'deeppoly'
        """
        self.ir_list = ir_list
        self.is_greater = None
        self.man = None
        self.layer_by_layer = layer_by_layer
        self.is_residual = is_residual
        self.is_blk_segmentation = is_blk_segmentation
        self.blk_size = blk_size
        self.is_early_terminate = is_early_terminate
        self.early_termi_thre = early_termi_thre
        self.is_sum_def_over_input = is_sum_def_over_input
        self.MAX_ITER = REFINE_MAX_ITER
        self.is_refinement = is_refinement
        self.refine = False
        if domain == 'deeppoly' or domain == 'refinepoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
            self.label_deviation_lb = label_deviation_lb
        self.domain = domain
        self.nn = nn
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.output_constraints = output_constraints
        self.use_default_heuristic = use_default_heuristic
        self.testing = testing
        self.relu_groups = []
        self.label = label
        self.prop = prop
    
    def __del__(self):
        elina_manager_free(self.man)
        
    def get_abstract0(self):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        testing_nlb = []
        testing_nub = []
        # print("The len of deeppolyNodes is ", len(self.ir_list))
        # print(self.ir_list)
        for i in range(1, len(self.ir_list)):
            #print(self.is_early_terminate)
            element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub, self.relu_groups, 'refine' in self.domain, self.timeout_lp, self.timeout_milp, self.use_default_heuristic, self.testing, self.layer_by_layer, self.is_residual, self.is_blk_segmentation, self.blk_size, self.is_early_terminate, self.early_termi_thre, self.is_sum_def_over_input, self.is_refinement)
            # print("Transformer done for ",i)
            if self.testing and isinstance(element_test_bounds, tuple):
                element, test_lb, test_ub = element_test_bounds
                testing_nlb.append(test_lb)
                testing_nub.append(test_ub)
            else:
                element = element_test_bounds
        if self.domain in ["refinezono", "refinepoly"]:
            gc.collect()
        if self.testing:
            return element, testing_nlb, testing_nub
        return element, nlb, nub
    
    def analyze(self):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element, nlb, nub = self.get_abstract0()
        output_size = 0
        output_size = self.ir_list[-1].output_length #reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
        dominant_class = -1
        label_failed = []
        x = None
        if self.output_constraints is None:
            candidate_labels = []
            if self.label == -1:
                for i in range(output_size):
                    candidate_labels.append(i)
            else:
                candidate_labels.append(self.label)
            adv_labels = []
            if self.prop == -1:
                for i in range(output_size):
                    adv_labels.append(i)
            else:
                adv_labels.append(self.prop)
            # print("adv_labels",adv_labels)   
            for i in candidate_labels:
                flag = True
                label = i
                for j in adv_labels:
                    if label!=j and not self.is_greater(self.man, element, label, j, self.use_default_heuristic, self.layer_by_layer, self.is_residual, self.is_blk_segmentation, self.blk_size, self.is_early_terminate, self.early_termi_thre, self.is_sum_def_over_input, self.is_refinement):
                        # testing if label is always greater than j
                        flag = False
                        if self.label!=-1:
                            label_failed.append(j)
                        if config.complete == False:
                            break


                if flag:
                    dominant_class = i
                    break
        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub, label_failed, x

    def cascade1_label_prune(self, man, element, ground_truth_label, poten_cex, spurious_list, spurious_count, MAX_ITER=5, layer_by_layer=False, is_blk_segmentation=False, blk_size=0, is_sum_def_over_input=FALSE):
        itr_count = 0 
        clear_neurons_status(man, element)
        run_deeppoly(man, element)
        
        while(itr_count < MAX_ITER):
            # print(itr_count, MAX_ITER)
            itr_count = itr_count + 1
            num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, coeffs = self.wholenet_krelu_cons(man, element, self.nn, 'refinepoly')
            res = cascade1_is_spurious(man, element, ground_truth_label, poten_cex, spurious_list, spurious_count, itr_count, num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, coeffs)
            # print("refreshed relu is ", res.relu_refresh_count)
            if(res.status == 1):
                return 1, itr_count
            elif(res.status == -1):
                return -1, itr_count
            elif(res.relu_refresh_count == 0):
                return 0, 5
        # print(0, itr_count)
        return 0, itr_count

    def prune_with_abstract_cascade(self, ground_truth_label, multi_cex_count = 4):
        """
        analyses the network with the given ground truth label
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        assert ground_truth_label!=-1, "The ground truth label cannot be -1!!!!!!!!!!!!!"
        assert self.output_constraints is None, "The output constraints are supposed to be None"
        assert self.prop == -1, "The prop are supposed to be deactivated"
        element, nlb, nub = self.get_abstract0()
        # print(nlb[-2])
        # print(nub[-2])
        output_size = 0
        cex_flag = False
        output_size = self.ir_list[-1].output_length #reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
        dominant_class = -1
        label_failed = [] # we can use this to record the actual counterexample we find out using LP solution
        potential_adv_labels = {} 
        # potential_adv_labels is the dictionary where key is the adv label i and value is the deviation ground_truth_label-i
        
        adv_labels = []
        sorted_adv_labels = []
        for i in range(output_size):
            if ground_truth_label!=i:
                adv_labels.append(i)
        flag = True
        potential_adv_count = 0
        for j in adv_labels:
            lb = self.label_deviation_lb(self.man, element, ground_truth_label, j, self.use_default_heuristic, self.layer_by_layer, self.is_residual, self.is_blk_segmentation, self.blk_size, self.is_early_terminate, self.early_termi_thre, self.is_sum_def_over_input, self.is_refinement)
            if lb < 0:
                # testing if label is always greater than j
                flag = False
                potential_adv_labels[j] = lb
                potential_adv_count = potential_adv_count + 1
        if flag:
            # if we successfully mark the groud truth label as dominant label
            dominant_class = ground_truth_label
        elif self.is_refinement:
            # do the spurious region pruning refinement
            sorted_d = dict(sorted(potential_adv_labels.items(), key=lambda x: x[1],reverse=False))
            spurious_list = []
            last_solve_ite = 5
            for poten_cex in sorted_d:
                sorted_adv_labels.append(poten_cex)
            n = 0
            while(n < len(sorted_adv_labels)):
                if(last_solve_ite > 2):
                    execution_flag, last_solve_ite = self.cascade1_label_prune(self.man, element, ground_truth_label, sorted_adv_labels[n], spurious_list, len(spurious_list), self.MAX_ITER)
                    if(execution_flag == 1):
                        spurious_list.append(sorted_adv_labels[n])
                        potential_adv_count = potential_adv_count - 1
                        n = n + 1
                    elif(execution_flag == -1):
                        cex_flag = True
                        break    
                    else:
                        break
                else:
                # if(True):
                    if(n+multi_cex_count <= len(sorted_adv_labels)):
                        # self.prima_calling_test()
                        execution_flag = self.check_multi_adv_labels(element, ground_truth_label, sorted_adv_labels[n:n+multi_cex_count], len(nlb), spurious_list)
                        if(execution_flag == 1):
                            spurious_list.extend(sorted_adv_labels[n:n+multi_cex_count])
                            potential_adv_count = potential_adv_count - multi_cex_count
                            n = n + multi_cex_count
                        elif(execution_flag == -1):
                            cex_flag = True
                            break
                        else:
                            break
                    else:
                        execution_flag = self.check_multi_adv_labels(element, ground_truth_label, sorted_adv_labels[n:len(sorted_adv_labels)], len(nlb), spurious_list)
                        if(execution_flag == 1):
                            spurious_list.extend(sorted_adv_labels[n:len(sorted_adv_labels)])
                            potential_adv_count = potential_adv_count - (len(sorted_adv_labels) - n)
                            n = len(sorted_adv_labels)
                        elif(execution_flag == -1):
                            cex_flag = True
                            break    
                        else:
                            break

            if(potential_adv_count == 0):
                print("Successfully refine the result")
                dominant_class = ground_truth_label
            
        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub, label_failed, cex_flag
    
    def index_grouping(self, grouplen, K):
        sparsed_combs = []
        i = 0
        if(K==3):
            while(i+2 < grouplen):
                sparsed_combs.append([i, i+1, i+2])
                i = i + 2
        return sparsed_combs
    
    def relu_grouping(self, length, lb, ub, K=3, s=-2):
        assert length == len(lb) == len(ub)

        all_vars = [i for i in range(length) if lb[i] < 0 < ub[i]]
        areas = {var: -lb[var] * ub[var] for var in all_vars}

        assert len(all_vars) == len(areas)
        sparse_n = config.sparse_n
        cutoff = 0.05
        # Sort vars by descending area
        all_vars = sorted(all_vars, key=lambda var: -areas[var])

        vars_above_cutoff = [i for i in all_vars if areas[i] >= cutoff]
        n_vars_above_cutoff = len(vars_above_cutoff)

        kact_args = []
        if len(vars_above_cutoff) > 0 and config.sparse_n >= K:
            grouplen = min(sparse_n, len(vars_above_cutoff))
            # print(grouplen)
            group = vars_above_cutoff[:grouplen]
            vars_above_cutoff = vars_above_cutoff[grouplen:]
            if grouplen <= K:
                kact_args.append(group)
            elif K>2:
                # sparsed_combs = generate_sparse_cover(grouplen, K, s=s)
                sparsed_combs = self.index_grouping(grouplen, K)
                for comb in sparsed_combs:
                    kact_args.append(tuple([group[i] for i in comb]))
            elif K==2:
                raise RuntimeError("K=2 is not supported")

        # Also just apply 1-relu for every var.
        # for var in all_vars:
        #     kact_args.append([var])

        # print("krelu: n", config.sparse_n,
        #     "split_zero", len(all_vars),
        #     "after cutoff", n_vars_above_cutoff,
        #     "number of args", len(kact_args))
        # print("number of args", len(kact_args))
        return kact_args
      
    def wholenet_krelu_cons(self, man, element, nn, domain, K=3, s=-2, approx=True):
        groupNum_each_layer = []
        consNum_each_group = []
        varsid_one_dim = []
        conv_coeffs = []
        total_group_Num = 0
        
        relu_layer_indexes = [i for i,x in enumerate(nn.layertypes) if x=='ReLU']
        for i, relu_layer in enumerate(relu_layer_indexes):
            # get the predecessor layer
            layerno = relu_layer - 1 
            length = get_num_neurons_in_layer(man, element, layerno)
            bounds = box_for_layer(self.man, element, layerno)
            itv = [bounds[i] for i in range(length)]
            nlb = [x.contents.inf.contents.val.dbl for x in itv]
            nub = [x.contents.sup.contents.val.dbl for x in itv]
            elina_interval_array_free(bounds,length)
            lbi = np.asarray(nlb, dtype=np.double)
            ubi = np.asarray(nub, dtype=np.double)
            kact_args = self.relu_grouping(length, lbi, ubi, K=K, s=s)
            if(len(kact_args) >= 1):
                groupNum_each_layer.append(len(kact_args))
                tdim = ElinaDim(length)
                KAct.man = man
                KAct.element = element
                KAct.tdim = tdim
                KAct.length = length
                KAct.layerno = layerno
                KAct.offset = 0
                KAct.domain = domain
                KAct.type = "ReLU"
                total_size = 0    
                for varsid in kact_args:
                    total_group_Num = total_group_Num + 1
                    varsid_one_dim.extend(varsid)
                    size = 3**len(varsid) - 1
                    total_size = total_size + size
                # print(len(kact_args))
                linexpr0 = elina_linexpr0_array_alloc(total_size)
                i = 0
                for varsid in kact_args:
                    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                        if all(c == 0 for c in coeffs):
                            continue
                        linexpr0[i] = generate_linexpr0(0, varsid, coeffs)
                        i = i + 1
                upper_bound = get_upper_bound_for_linexpr0(man,element,linexpr0, total_size, layerno)
                i=0
                input_hrep_array = []
                for varsid in kact_args:
                    input_hrep = []
                    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                        if all(c == 0 for c in coeffs):
                            continue
                        input_hrep.append([upper_bound[i]] + [-c for c in coeffs])
                        i = i + 1
                    input_hrep_array.append(input_hrep)

                with multiprocessing.Pool(config.numproc) as pool:
                    kact_results = pool.starmap(make_kactivation_obj, zip(input_hrep_array, len(input_hrep_array) * [approx]))
                
                gid = 0
                total_row = 0
                for inst in kact_results:
                    varsid = kact_args[gid]
                    inst.varsid = varsid
                    gid = gid+1
                    rows = 0
                    cols = 2*len(varsid)+1
                    non_redun_cons = []
                    for row in inst.cons:
                        if non_redun_cons == []:
                            non_redun_cons.append(row)
                        elif all([any([(abs(element[i]-row[i]) >= 10**-8) for i in range(len(row))]) for element in non_redun_cons]):        
                            non_redun_cons.append(row)
                    for row in non_redun_cons:
                        total_row = total_row + 1
                        rows = rows + 1
                        for i in range(cols):
                            if(abs(row[i])<= 10**-7):
                                conv_coeffs.append(0.0)
                            elif(abs(row[i]-1)<= 10**-7):
                                conv_coeffs.append(1.0)
                            else:
                                conv_coeffs.append(row[i])
                    consNum_each_group.append(rows)
            else:
                groupNum_each_layer.append(len(kact_args))
            
        print(len(groupNum_each_layer), groupNum_each_layer, total_group_Num, len(consNum_each_group), len(varsid_one_dim), len(np.float64(conv_coeffs)))
        return len(groupNum_each_layer), groupNum_each_layer, total_group_Num, consNum_each_group, varsid_one_dim, np.float64(conv_coeffs)
         
    def check_multi_adv_labels(self, element, ground_truth_label, multi_list, total_layer_num, spurious_list):
        # if any stablized node, leave it to cdd to handle it
        itr_count = 0 
        clear_neurons_status(self.man, element)
        run_deeppoly(self.man, element)
        pre_solve_status = 0
        while(itr_count < self.MAX_ITER):
            # get the bound of input neuron to the output layer
            itr_count = itr_count + 1
            num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, krelu_coeffs = self.wholenet_krelu_cons(self.man, element, self.nn, 'refinepoly')
            if(pre_solve_status != -2):
                execution_flag = multi_cex_spurious_with_cdd(self.man, element, ground_truth_label, multi_list, len(multi_list), spurious_list, len(spurious_list), itr_count, num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, krelu_coeffs)
                # use cdd to handle the disjunction
                if(execution_flag == 1):
                    return 1
                elif(execution_flag == -1):
                    return -1
                elif(execution_flag == -2):
                    pre_solve_status = -2
                    break
                
        if(pre_solve_status == -2):
            # divide the multi_cex list and call two times, if we face numerical error when calling cddlib    
            half_len = int(len(multi_list)/2)
            clear_neurons_status(self.man, element)
            run_deeppoly(self.man, element)
            itr_count = 0 
            execution_flag1 = 0
            while(itr_count < self.MAX_ITER):
                itr_count = itr_count + 1
                num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, krelu_coeffs = self.wholenet_krelu_cons(self.man, element, self.nn, 'refinepoly')
                execution_flag1 = multi_cex_spurious_with_cdd(self.man, element, ground_truth_label, multi_list[:half_len], len(multi_list[:half_len]), spurious_list, len(spurious_list), itr_count, num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, krelu_coeffs)
                print(multi_list[:half_len], len(multi_list[:half_len]))
                # use cdd to handle the disjunction
                if(execution_flag1 == 1):
                    break
                elif(execution_flag1 == -1):
                    return -1
            if(execution_flag1 != 1):
                return 0
            clear_neurons_status(self.man, element)
            run_deeppoly(self.man, element)
            itr_count = 0 
            while(itr_count < self.MAX_ITER):
                itr_count = itr_count + 1
                num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, krelu_coeffs = self.wholenet_krelu_cons(self.man, element, self.nn, 'refinepoly')
                execution_flag2 = multi_cex_spurious_with_cdd(self.man, element, ground_truth_label, multi_list[half_len:], len(multi_list[half_len:]), spurious_list, len(spurious_list), itr_count, num_relu_layers, groupNum_each_layer, group_num, consNum_each_group, varsid_one_dim, krelu_coeffs)
                print(multi_list[half_len:], len(multi_list[half_len:]))
                # use cdd to handle the disjunction
                if(execution_flag2 == 1):
                    return 1
                elif(execution_flag2 == -1):
                    return -1  
        return 0  
        
    def prima_calling_test(self):
        KAct.type = "ReLU"
        input_hrep = []
        input_hrep.append([2.0, 1, 1])
        input_hrep.append([2.0, 1, 0])
        input_hrep.append([2.0, 1, -1])
        input_hrep.append([2.0, 0, 1])
        input_hrep.append([1.2, 0, -1])
        input_hrep.append([2.0, -1, 1])
        input_hrep.append([2.0, -1, 0])
        input_hrep.append([2.0, -1, -1])
        
        print(input_hrep)
        kact_results = make_kactivation_obj(input_hrep, True)
        rows = 0
        cols = 5
        convex_coeffs = []
        print("!!!!!!!!!!!!!!!!!!!!!")
        non_redun_cons = []
        for row in kact_results.cons:
            if non_redun_cons == []:
                non_redun_cons.append(row)
            elif all([any([element[i] != row[i] for i in range(len(row))]) for element in non_redun_cons]):        
                non_redun_cons.append(row)
        
        for row in non_redun_cons:
            print(row)
        print("!!!!!!!!!!!!!!!!!!!!!")
        for row in kact_results.cons:
            print(row)
            rows = rows + 1
            for i in range(cols):
                convex_coeffs.append(row[i])
        print("!!!!!!!!!!!!!!!!!!!!!")
        print(convex_coeffs)
        print(rows)
        return False
           
        