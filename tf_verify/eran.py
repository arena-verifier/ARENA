'''
@author: Adrian Hoffmann
'''
import gc
from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *


class ERAN:
    def __init__(self, model, session=None, is_onnx = False):
        """
        This constructor takes a reference to a TensorFlow Operation, TensorFlow Tensor, or Keras model. The two TensorFlow functions graph_util.convert_variables_to_constants and 
        graph_util.remove_training_nodes will be applied to the graph to cleanse it of any nodes that are linked to training.
        In the resulting graph there should only be tf.Operations left that have one of the following types [Const, MatMul, Add, BiasAdd, Conv2D, Reshape, MaxPool, Placeholder, Relu, Sigmoid, Tanh]
        If the input should be a Keras model we will ignore operations with type Pack, Shape, StridedSlice, and Prod such that the Flatten layer can be used.
        
        Arguments
        ---------
        model : tensorflow.Tensor or tensorflow.Operation or tensorflow.python.keras.engine.sequential.Sequential or keras.engine.sequential.Sequential
            if tensorflow.Tensor: model.op will be treated as the output node of the TensorFlow model. Make sure that the graph only contains supported operations after applying
                                  graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [model.op.name] as output_node_names
            if tensorflow.Operation: model will be treated as the output of the TensorFlow model. Make sure that the graph only contains supported operations after applying
                                  graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [model.op.name] as output_node_names
            if tensorflow.python.keras.engine.sequential.Sequential: x = model.layers[-1].output.op.inputs[0].op will be treated as the output node of the Keras model. Make sure that the graph only
                                  contains supported operations after applying graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [x.name] as
                                  output_node_names
            if keras.engine.sequential.Sequential: x = model.layers[-1].output.op.inputs[0].op will be treated as the output node of the Keras model. Make sure that the graph only
                                  contains supported operations after applying graph_util.convert_variables_to_constants and graph_util.remove_training_nodes with [x.name] as
                                  output_node_names
        session : tf.Session
            session which contains the information about the trained variables. If session is None the code will take the Session from tf.get_default_session(). If you pass a keras model you don't 
            have to provide a session, this function will automatically get it.
        """
        if is_onnx:
            self.translator = ONNXTranslator(model)
        else:
            self.translator = TFTranslator(model, session)
        operations, resources = self.translator.translate()
        #print("Operations",operations)
        self.optimizer  = Optimizer(operations, resources)
        #print('This network has ' + str(self.optimizer.get_neuron_count()) + ' neurons.')

    def analyze_in_stages(self, specLB, specUB,  domain, timeout_lp, timeout_milp, use_default_heuristic, stage_layer_num=None, stage_block_num=None, layer_by_layer=False,
                    is_residual=False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, is_refinement=False, output_constraints=None, lexpr_weights=None, lexpr_cst=None, lexpr_dim=None,
                    uexpr_weights=None, uexpr_cst=None, uexpr_dim=None, expr_size=0, testing=False, label=-1, prop=-1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        More importantly, this function will do the analysis using staging
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.

        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deeppoly'], "The domain for staging only include deeppoly"
        operations_list, resources_list = self.translator.staging_trans(stage_layer_num, stage_block_num)
        #print("The number of dp nodes is ", len(operations_list))
        for i in range(len(operations_list)):
            #print (operations_list[i])
            optimizer  = Optimizer(operations_list[i], resources_list[i])
            specLB = np.reshape(specLB, (-1,))
            specUB = np.reshape(specUB, (-1,))
            nn = layers()
            nn.specLB = specLB
            nn.specUB = specUB
            execute_list, output_info = optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst,
                                                                    lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim,
                                                                    expr_size)
            analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints,
                                use_default_heuristic, label, prop, testing, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement)
            _, nlb, nub = analyzer.get_abstract0()
            specLB = nlb[-1]
            specUB = nub[-1]
            del optimizer
            del execute_list 
            del output_info
            del analyzer
            gc.collect()
            #print ("End eran process for this network segment")
        return nlb, nub

    def analyze_box(self, specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic, layer_by_layer = False, is_residual = False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, is_refinement = False,output_constraints=None, lexpr_weights= None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None, uexpr_cst=None, uexpr_dim=None, expr_size=0, testing = False,label=-1, prop = -1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain isn't valid, must be 'deepzono' or 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size)
        analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement)
        dominant_class, nlb, nub, failed_labels, x = analyzer.analyze()
        if testing:
            return dominant_class, nn, nlb, nub, output_info
        else:
            return dominant_class, nn, nlb, nub, failed_labels, x

    def SMUPoly(self, specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic, layer_by_layer = False, is_residual = False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, is_refinement = False, REFINE_MAX_ITER = 5, output_constraints=None, lexpr_weights= None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None, uexpr_cst=None, uexpr_dim=None, expr_size=0, testing = False,label=-1, prop = -1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain isn't valid, must be 'deepzono' or 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size)
        analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement, REFINE_MAX_ITER)
        dominant_class, nlb, nub, failed_labels, x = analyzer.analyze_groud_truth_label(label)
        # dominant_class, nlb, nub, failed_labels, x = analyzer.analyze_groud_truth_label_reverse(label)
        if testing:
            return dominant_class, nn, nlb, nub, output_info
        else:
            return dominant_class, nn, nlb, nub, failed_labels, x

    def refine_cascade(self, specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic, layer_by_layer = False, is_residual = False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, is_refinement = False, REFINE_MAX_ITER = 5, multiadv = 4, output_constraints=None, lexpr_weights= None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None, uexpr_cst=None, uexpr_dim=None, expr_size=0, testing = False,label=-1, prop = -1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain isn't valid, must be 'deepzono' or 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size)
        analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement, REFINE_MAX_ITER)
        dominant_class, nlb, nub, failed_labels, x = analyzer.prune_with_abstract_cascade(label, multiadv)
        if testing:
            return dominant_class, nn, nlb, nub, output_info
        else:
            return dominant_class, nn, nlb, nub, failed_labels, x

    def analyze_with_ground_truth_label(self, specLB, specUB, domain, timeout_lp, timeout_milp, use_default_heuristic, layer_by_layer = False, is_residual = False, is_blk_segmentation=False, blk_size=0, is_early_terminate = False, early_termi_thre = 0, is_sum_def_over_input = True, is_refinement = False, REFINE_MAX_ITER = 5, output_constraints=None, lexpr_weights= None, lexpr_cst=None, lexpr_dim=None, uexpr_weights=None, uexpr_cst=None, uexpr_dim=None, expr_size=0, testing = False,label=-1, prop = -1):
        """
        This function runs the analysis with the provided model and session from the constructor, the box specified by specLB and specUB is used as input. Currently we have three domains, 'deepzono',      		'refinezono' and 'deeppoly'.
        
        Arguments
        ---------
        specLB : numpy.ndarray
            ndarray with the lower bound of the input box
        specUB : numpy.ndarray
            ndarray with the upper bound of the input box
        domain : str
            either 'deepzono', 'refinezono', 'deeppoly', or 'refinepoly', decides which set of abstract transformers is used.
            
        Return
        ------
        dominant_class : int
            if the analysis is succesfull (it could prove robustness for this box) then the index of the class that dominates is returned
            if the analysis couldn't prove robustness then -1 is returned
        """
        assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain isn't valid, must be 'deepzono' or 'deeppoly'"
        specLB = np.reshape(specLB, (-1,))
        specUB = np.reshape(specUB, (-1,))
        nn = layers()
        nn.specLB = specLB
        nn.specUB = specUB
        execute_list, output_info = self.optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size)
        analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label, prop, testing, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement, REFINE_MAX_ITER)
        # dominant_class, nlb, nub, failed_labels, x = analyzer.analyze_groud_truth_label(label)
        dominant_class, nlb, nub, failed_labels, x = analyzer.analyze_with_multi_cex_pruning_cdd(label)
        if testing:
            return dominant_class, nn, nlb, nub, output_info
        else:
            return dominant_class, nn, nlb, nub, failed_labels, x