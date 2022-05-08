"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


from deeppoly_nodes import *
from functools import reduce
import numpy as np


operations_for_neuron_count = ["Relu", "Sigmoid", "Tanh", "MaxPool"]


class Optimizer:
    def __init__(self, operations, resources):
        """
        Arguments
        ---------
        operations : list
            list of dicts, each dict contains a mapping from a domain (like deepzono, refinezono or deeppoly) to a tuple with resources (like matrices, biases ...)
        resources : list
            list of str, each one being a type of operation (like "MatMul", "Conv2D", "Add" ...)
        """
        self.operations = operations
        self.resources  = resources

    def get_neuron_count(self):
        total_neurons = 0
        for op, res in zip(self.operations, self.resources):
            if op in operations_for_neuron_count:
                if len(res['deepzono'][-1])==4:
                    total_neurons += np.prod(res['deepzono'][-1][1:len(res['deepzono'][-1])])
                else:
                    total_neurons += np.prod(res['deepzono'][-1][0:len(res['deepzono'][-1])])
        return total_neurons

    def get_abstract_element(self, nn, i, execute_list, output_info, domain):
        # Function called as self.get_abstract_element(nn, 1, execute_list, output_info, 'deeppoly')
        assert domain == "deepzono" or domain == "deeppoly", "ERAN does not support" + domain + " abstraction"
        nbr_op = len(self.operations)
        while i < nbr_op:
            if self.operations[i] == "MatMul":
                nn.layertypes.append('FC')
                if i < nbr_op-1 and self.operations[i+1] in ["Add", "BiasAdd"]:
                    matrix,  m_input_names, _, _           = self.resources[i][domain]
                    bias, _, output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    #self.resources[i][domain].append(refine)
                    matrix, m_input_names , output_name , b_output_shape  = self.resources[i][domain]
                    
                    bias_length = reduce((lambda x, y: x*y), b_output_shape)
                    bias = nn.zeros(bias_length)

                    i += 1
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyFCNode(matrix, bias, m_input_names, output_name, b_output_shape))
                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.numlayer+= 1
            elif self.operations[i] == "Gemm":
                matrix, bias, m_input_names, b_output_name, b_output_shape = self.resources[i][domain]

                nn.weights.append(matrix)
                nn.biases.append(bias)
                nn.layertypes.append('FC')
                nn.numlayer+= 1
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyFCNode(matrix, bias, m_input_names, b_output_name, b_output_shape))
                i += 1
            
            elif self.operations[i] == "Conv2D":
                if i < nbr_op-1 and self.operations[i+1] == "BiasAdd":
                    filters, image_shape, strides, pad_top, pad_left, c_input_names, _, _ = self.resources[i][domain]
                    bias, _, b_output_name, b_output_shape = self.resources[i+1][domain]
                    i += 2
                else:
                    filters, image_shape, strides, pad_top, pad_left, c_input_names, b_output_name, b_output_shape = self.resources[i][domain]
                    bias_length = reduce((lambda x, y: x*y), output_shape)
                    bias = nn.zeros(bias_length)
                    i += 1
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.padding.append([pad_top, pad_left])
                nn.out_shapes.append(b_output_shape)
                nn.filters.append(filters)
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyConv2dNode(filters, strides, pad_top, pad_left, bias, image_shape, c_input_names, b_output_name, b_output_shape))
                nn.numlayer+=1
            elif self.operations[i] == "Conv":
                filters, bias, image_shape, strides, pad_top, pad_left, c_input_names, output_name, b_output_shape = self.resources[i][domain]
                nn.numfilters.append(filters.shape[3])
                nn.filter_size.append([filters.shape[0], filters.shape[1]])
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(b_output_shape)
                nn.padding.append([pad_top, pad_left])
                nn.filters.append(filters)
                # print("Conv")
                nn.biases.append(bias)
                nn.layertypes.append('Conv')
                nn.numlayer+=1
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyConv2dNode(filters, strides, pad_top, pad_left, bias, image_shape, c_input_names, output_name, b_output_shape))
                i += 1
            elif self.operations[i] == "Resadd":
                #self.resources[i][domain].append(refine)
                # print("Enter Resadd session")
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyResidualNode(*self.resources[i][domain]))
                nn.layertypes.append('Resadd')
                nn.numlayer += 1
                i += 1
            #elif self.operations[i] == "Add":
                #self.resources[i][domain].append(refine)
           #     execute_list.append(DeepzonoAdd(*self.resources[i][domain]))
           #     nn.layertypes.append('Add')
           #     nn.numlayer += 1
           #     i += 1
            elif self.operations[i] == "Sub":
                #self.resources[i][domain].append(refine)
                if domain == 'deeppoly':
                    execute_list.append(DeeppolySubNode(*self.resources[i][domain]))
                nn.layertypes.append('Sub')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Mul":
                #self.resources[i][domain].append(refine)
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyMulNode(*self.resources[i][domain]))
                nn.layertypes.append('Mul')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "MaxPool" or self.operations[i] == "AveragePool" or self.operations[i] == "AvgPool":
                image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape = self.resources[i][domain]
                nn.pool_size.append(window_size)
                nn.input_shape.append([image_shape[0],image_shape[1],image_shape[2]])
                nn.strides.append([strides[0],strides[1]])
                nn.out_shapes.append(output_shape)
                nn.padding.append([pad_top, pad_left])
                nn.numlayer+=1
                is_maxpool = (self.operations[i]=="MaxPool")
                if is_maxpool:
                    nn.layertypes.append('Maxpool')
                else:
                    nn.layertypes.append('Avgpool')
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyPoolNode(image_shape, window_size, strides, pad_top, pad_left, input_names, output_name, output_shape, is_maxpool))
                i += 1
            elif self.operations[i] == "Relu":
                #self.resources[i][domain].append(refine)
                nn.layertypes.append('ReLU')
                if domain == 'deeppoly':
                    #print("deeppoly relu")
                    execute_list.append(DeeppolyReluNode(*self.resources[i][domain]))
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Sigmoid":
                if domain == 'deeppoly':
                    execute_list.append(DeeppolySigmoidNode(*self.resources[i][domain]))
                nn.layertypes.append('Sigmoid')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Tanh":
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyTanhNode(*self.resources[i][domain]))
                nn.layertypes.append('Tanh')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Gather":
                image_shape, indexes, axis,  input_names, output_name, output_shape = self.resources[i][domain]
                calculated_indexes = self.get_gather_indexes(image_shape, indexes, axis)
                if domain == 'deeppoly':
                    # print("Adding DeeppolyGather node into the execution list")
                    execute_list.append(DeeppolyGather(calculated_indexes, input_names, output_name, output_shape))
                nn.layertypes.append('Gather')
                nn.numlayer += 1
                i += 1
            elif self.operations[i] == "Reshape":
                indexes, input_names, output_name, output_shape = self.resources[i][domain]
                if domain == 'deeppoly':
                    execute_list.append(DeeppolyGather(indexes, [input_names[0]], output_name, output_shape))
                nn.layertypes.append('Gather')
                nn.numlayer += 1
                i += 1
            else:
                assert 0, "the optimizer for" + domain + " doesn't know of the operation type " + self.operations[i]
            # the output info for last layer, which is the input name & shape for this layer
            output_info.append(self.resources[i-1][domain][-2:])

    def get_deeppoly(self, nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints=None):
        """
        This function will go through self.operations and self.resources and create a list of Deeppoly-Nodes which then can be run by an Analyzer object.
        It is assumed that self.resources[i]['deeppoly'] holds the resources for an operation of type self.operations[i].
        self.operations should only contain a combination of the following 4 basic sequences:
            - Placholder         (only at the beginning)
                - MatMul -> Add -> Relu
                - Conv2D -> Add -> Relu    (not as last layer)
                - MaxPool/AveragePool         (only as intermediate layer)

        Arguments
        ---------
        specLB : numpy.ndarray
            1D array with the lower bound of the input spec
        specUB : numpy.ndarray
            1D array with the upper bound of the input spec

        Return
        ------
        execute_list : list
            list of Deeppoly-Nodes that can be run by an Analyzer object
        """
        execute_list = []
        output_info = []
        domain = 'deeppoly'
        assert self.operations[0] == "Placeholder", "the optimizer for Deeppoly cannot handle this network "
        input_names, output_name, output_shape = self.resources[0][domain]
        # The return of the above statement is "[] input [1, 14, 14, 5]"
        output_info.append(self.resources[0][domain][-2:])
        execute_list.append(DeeppolyInput(specLB, specUB, input_names, output_name, output_shape,
                                            lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints))
        # append the placeholder first, where the DeeppolyInput() object is only initilaized but not doing anything so far
        self.get_abstract_element(nn, 1, execute_list, output_info, 'deeppoly')
        self.set_predecessors(nn, execute_list)
        #print("List of predecessors: ", nn.predecessors)
        return execute_list, output_info

    def set_predecessors(self, nn, output):
        output_index_store = {}
        index_o = 0
        for node in output:
            output_index_store[node.output_name] = index_o
            index_o += 1
        #print("set predecessors:",output_index_store)
        for node in output:
            #print("output ", node, node.input_names)
            predecessors = (c_size_t * len(node.input_names))()
            i = 0
            for input_name in node.input_names:
                # if(len(node.input_names)==2):
                #     print("Residual predecessor ", i, "is ", output_index_store[input_name])
                # if(len(node.input_names)==1):
                #     print("Conv predecessor ", i, "is ", output_index_store[input_name])
                predecessors[i] = output_index_store[input_name]
                i += 1
            node.predecessors = predecessors
            #if not isinstance(node, DeepzonoRelu):
            nn.predecessors.append(predecessors)

    def get_gather_indexes(self, input_shape, indexes, axis):
        size = np.prod(input_shape)
        base_indexes = np.arange(size).reshape(input_shape)
        return np.take(base_indexes, indexes, axis=axis)
