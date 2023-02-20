
import numpy as np
import torch.nn as nn
from cell_module.ops import OPS as ops_dict

class EncoderCell(nn.Module):
    def __init__(self, chromosome, node_boundaries, prev_C, current_C):
        super(EncoderCell, self).__init__()

        self.chromosome = chromosome
        self.node_boundaries = node_boundaries
        self.prev_C = prev_C
        self.current_C = current_C
        self.stem_conv = nn.Conv2d(self.prev_C, self.current_C, kernel_size=1, padding='same')
        self.compile()

    def compile(self):
        self.ops_list = nn.ModuleList([self.stem_conv])

        ops = self.chromosome[self.chromosome != '0'][:-2]
        for op in ops:
            if op != 'o':
                self.ops_list.append(ops_dict[op](self.current_C, self.current_C))
            else:
                self.ops_list.append(None)

    
    def forward(self, inputs):
        outputs = [0] * (len(list(self.node_boundaries.keys())) + 1) # Store output of hidden nodes
        outputs[0] = self.ops_list[0](inputs) # Stem Convolution - Equalize channel count
        outputs[1] = self.ops_list[1](outputs[0])
        
        op_idx = 2
        for hidden_node_idx, boundaries in self.node_boundaries.items():
            if hidden_node_idx == 1: continue

            lb, ub = boundaries
            in_hidden_nodes = np.where(self.chromosome[lb : ub + 1] != '0')[0] # Hidden nodes that have in edges to current hidden node

            outputs[hidden_node_idx] = sum([self.ops_list[op_idx + i](outputs[in_hidden_nodes[i]]) 
                                                                    for i in range(len(in_hidden_nodes)) if self.ops_list[op_idx + i] is not None])
            op_idx = op_idx + len(in_hidden_nodes)
        
        # Add out edge to nodes that have no out edge to any hidden node
        no_out_edge_nodes = self.get_no_out_edge_nodes(self.chromosome, self.node_boundaries, len(list(self.node_boundaries.items())))
        if len(no_out_edge_nodes) > 0:
            outputs[-1] = outputs[-1] + sum([outputs[hidden_node_idx] for hidden_node_idx in no_out_edge_nodes])
        
        return outputs[-1]
    
    def get_no_out_edge_nodes(self, chromosome, node_boundaries, nbr_hidden_nodes):

        out_edges = np.array([0 for i in range(nbr_hidden_nodes)]) # Input, Node 1, 2, 3, 4
        for node_id, boundaries in node_boundaries.items():
            lb, ub = boundaries
            out_edge_nodes = np.where(chromosome[lb: ub + 1] != '0') # index of nodes that have out edge
            out_edges[out_edge_nodes] += 1

        return np.where(out_edges == 0)[0]