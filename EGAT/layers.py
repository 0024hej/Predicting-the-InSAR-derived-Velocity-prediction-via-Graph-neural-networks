import time
import torch as th
import dgl
import math
import torch.nn as nn
import torch.nn.functional
from torch.nn import init
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
import torch.optim as optim
import argparse
import numpy as np
import random
import numpy as np
import scipy.sparse as sp
import torch
class MultiInputMLPUpdate(nn.Module):
    #input_including (self._node_feats, self.we, self._edge_feats, self._edge_feats, self._edge_feats)
    def __init__(self, in_feats1, in_feats2, in_feats3,hidden_feats, out_feats):
        super(MultiInputMLPUpdate, self).__init__()

        self.input_layer1 = nn.Linear(in_feats1, hidden_feats)
        self.input_layer2 = nn.Linear(in_feats2, hidden_feats)
        self.input_layer3 = nn.Linear(in_feats3, hidden_feats)

        self.hidden_layer = nn.Linear(hidden_feats, hidden_feats)
        # define output layer
        self.output_layer = nn.Linear(hidden_feats, out_feats)
    def update_edge(self, edges):
        input1 = edges.src['h_i']
        input2 = edges.dst['h_j']
        input3 = edges.src['e_i']
        input4 = edges.dst['e_j']
        input5 = edges.data['e_ij']
        # Transforms inputs through a linear layer
        h1 = self.input_layer1(input1)
        h2 = self.input_layer1(input2)
        h3 = self.input_layer2(input3)
        h4 = self.input_layer2(input4)
        h5 = self.input_layer3(input5)
        # Add or concatenate the five inputs
        #h = th.cat([h1,h2,h3,h4,h5], dim=-1)
        h = h1 + h2 + h3 + h4 + h5
        print(h.shape)
        # through hidden_layer
        h = nn.functional.elu(self.hidden_layer(h))
        # output layer
        h = self.output_layer(h)
        
        # Save the calculated result in nodes
        #print("000000000000000")
        return {'e_out':h}

    def forward(self, graph):
        with graph.local_scope():
            #print(graph)
            #device = torch.device('cuda:0')
            #graph = graph.to(device)
            graph.apply_edges(self.update_edge)
            #print("ppppppppppppppppppppppppppppppppppppppppp")
            # return graph.edata['e_out']
            return graph.edata['e_out']

class EGATLayer(nn.Module):

    def __init__(self,
                node_feats,
                 edge_feats,
                 lamda,
                 num_heads,bias=True):

        super().__init__()
        self._num_heads = num_heads
        self._node_feats = node_feats
        self._edge_feats = edge_feats
        self.wh = math.floor(node_feats*lamda)
        self.we = math.ceil(node_feats*(1-lamda))
        
        self.fh_n = nn.Linear(node_feats, math.floor(node_feats*lamda)*num_heads, bias=False)
        self.fe_n = nn.Linear(edge_feats, math.ceil(node_feats*(1-lamda))*num_heads, bias=False)
        self.fh_e = nn.Linear(node_feats, math.floor(node_feats*lamda)*num_heads, bias=False)
        self.fe_e = nn.Linear(edge_feats, math.ceil(node_feats*(1-lamda))*num_heads, bias=False)
        self.a_h_node = nn.Parameter(th.FloatTensor(size=(1, num_heads, math.floor(node_feats*lamda))))
        self.a_e_node = nn.Parameter(th.FloatTensor(size=(1, num_heads, math.ceil(node_feats*(1-lamda)))))
        self.a_h_edge = nn.Parameter(th.FloatTensor(size=(1, num_heads, math.floor(node_feats*lamda))))
        self.a_e_edge = nn.Parameter(th.FloatTensor(size=(1, num_heads, math.ceil(node_feats*(1-lamda)))))
        if bias:
            self.bias1 = nn.Parameter(th.FloatTensor(size=(node_feats,)))
            self.bias2 = nn.Parameter(th.FloatTensor(size=(edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fh_n.weight, gain=gain)
        init.xavier_normal_(self.fe_n.weight, gain=gain)
        init.xavier_normal_(self.fh_e.weight, gain=gain)
        init.xavier_normal_(self.fe_e.weight, gain=gain)
        init.xavier_normal_(self.a_h_node, gain=gain)
        init.xavier_normal_(self.a_e_node, gain=gain)
        init.xavier_normal_(self.a_h_edge, gain=gain)
        init.xavier_normal_(self.a_e_edge, gain=gain)
        if self.bias1 is not None:
            nn.init.constant_(self.bias1, 0)
        if self.bias2 is not None:
            nn.init.constant_(self.bias2, 0)


    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue.')
            ###Node Module
            #print(self.fh_n(nfeats).shape)
            #print("------------------second layer")
            nfeat_wn = self.fh_n(nfeats).view(-1, self._num_heads, self.wh)
            efeat_wn = self.fe_n(efeats).view(-1, self._num_heads, self.we)
            Fh_n = (nfeat_wn * self.a_h_node).sum(dim=-1).unsqueeze(-1)
            Fe_n = (efeat_wn * self.a_e_node).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'h_node': nfeat_wn, 'f_i_n': Fh_n})
            graph.dstdata.update({'f_j_n': Fh_n})
            graph.edata['f_ij_n'] = Fe_n
            graph.edata['e_ij_n'] = efeat_wn
            # graph.srcdata.update({'f_ni': f_ni})
            # graph.dstdata.update({'f_nj': f_nj})
            # add ni, nj factors  
            graph.apply_edges(fn.u_add_v('f_i_n', 'f_j_n', 'f_tmp_n'))
            # add fij to node factor
            w_ij_n = graph.edata.pop('f_tmp_n') + graph.edata['f_ij_n']
            # if self.bias is not None:
            #     f_out = f_out + self.bias
            e_n =nn.functional.leaky_relu(w_ij_n)
            # compute attention factor
            graph.edata['a_n'] = edge_softmax(graph, e_n)
            # calc weighted sum
            def cat_n(edges):
              edges.data['m'] = th.cat([edges.src['h_node'],edges.data['e_ij_n']],dim=-1)
              # print(edges.src['h_node'].shape)
              # print(edges.data['e_ij_n'].shape)
              edges.data['m'] = edges.data['m'] * edges.data['a_n']
              return
            graph.update_all(cat_n,fn.sum('m', 'h_node'))
            h_node = nn.functional.elu(graph.ndata['h_node'].view(-1, self._num_heads, self._node_feats))
            #print(h_node.shape)
            h_node = h_node.mean(dim=-2)
            if self.bias1 is not None:
                h_node = h_node + self.bias1
            #print("--------------------------")
            #print(h_node.shape)

            
            ###Edge Module
            #print(self.fh_e(h_node).shape)
            #print(self.wh)
            nfeat_we = self.fh_e(h_node).view(-1, self._num_heads, self.wh)
            efeat_we = self.fe_e(efeats).view(-1, self._num_heads, self.we)
            #print(nfeat_we.shape)
            Fh_e = (nfeat_we * self.a_h_edge).sum(dim=-1).unsqueeze(-1)
            Fe_e = (efeat_we * self.a_e_edge).sum(dim=-1).unsqueeze(-1)
            #print(Fh_e.shape)
            graph.srcdata.update({'f_i_e': Fh_e})
            graph.dstdata.update({'f_j_e': Fh_e})
            graph.edata['f_ij_e'] = efeat_we
            graph.apply_edges(fn.u_add_v('f_i_e', 'f_j_e', 'f_tmp_e'))
            # add fij to node factor
            w_ij_e = graph.edata.pop('f_tmp_e') + graph.edata['f_ij_e']
            # if self.bias is not None:
            #     f_out = f_out + self.bias
            e_e =nn.functional.leaky_relu(w_ij_e)
            # compute attention factor
            graph.edata['a_e'] = edge_softmax(graph, e_e)
            def add_e(edges):
              #print(edges.data['f_ij_e'].shape,edges.data['a_e'].shape)
              #print("1111111111111111")
              edges.data['m2'] = edges.data['f_ij_e'] * edges.data['a_e']
              #print(edges.data['m2'].shape)
              return
            graph.update_all(add_e,fn.sum('m2', 'e_node'))
            #print("--------------------------------------------------")
            #print(graph.ndata['e_node'].shape)
            #print(graph.edata['f_ij_e'].shape)
            e_node = graph.ndata['e_node'].view(-1, self._num_heads, self.we)
            e_node = e_node.mean(dim=-2)
            
            graph.srcdata.update({'h_i':h_node,'e_i':e_node})
            graph.dstdata.update({'h_j':h_node,'e_j':e_node})
            graph.edata.update({'e_ij':efeats})

            multi_input_mlp_edge_update = MultiInputMLPUpdate(self._node_feats, self.we, self._edge_feats, self._edge_feats, self._edge_feats)

            graph.edata['e_out'] = multi_input_mlp_edge_update(graph)
            e_out = graph.edata['e_out'].view(-1,self._edge_feats)
            if self.bias2 is not None:
                e_out = e_out + self.bias2
            
            return h_node, e_out   

class BottleneckLayer(nn.Module):
    def __init__(self, in_node, in_edge, out_node, out_edge):
        super(BottleneckLayer, self).__init__()
        self.node = nn.Linear(in_node, out_node)
        self.edge = nn.Linear(in_edge, out_edge)
    
    def forward(self, h, e):
        h_out = self.node(h)
        h_out = nn.functional.elu(h_out)
        e_out = self.edge(e)
        e_out = nn.functional.elu(e_out)
        return h_out,e_out

#MLP Predictor
class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.elu(x)
        x = self.fc2(x)
        #x = nn.functional.softmax(x,dim=1)
        return x

#MergeLayer
class MergeLayer(nn.Module):
    def __init__(self, node_feats,edge_feats,l_num):
        super(MergeLayer, self).__init__()
        self.fc1 = nn.Linear(l_num * node_feats, node_feats)
        self.fc2 = nn.Linear(l_num * edge_feats, edge_feats)

    def forward(self, h_final, e_final):
        h_final = self.fc1(h_final)
        h_final = nn.functional.elu(h_final)
        e_final = self.fc2(e_final)
        e_final = nn.functional.elu(e_final)
        return h_final, e_final

