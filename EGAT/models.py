import torch.nn as nn
import torch.nn.functional as F
import torch as th
from layers import BottleneckLayer,EGATLayer,MLPPredictor,MergeLayer
class EGAT(nn.Module):
    def __init__(self, node_feats, edge_feats, f_h,f_e,lamda, num_heads, dropout,pred_hid,l_num):
        #node_feats=node_features.shape[1],edge_feats=edge_features.shape[1],
        #f_h = 128,f_e = 128,lamda = args.lamda,num_heads = args.num_heads,dropout=args.dropout,pred_hid = 128, l_num = args.l_nums, 
        super(EGAT, self).__init__()
        self.l = l_num
        self.bottleneck = BottleneckLayer(node_feats, edge_feats,f_h,f_e)  #first, load node and edge data and preliminary non-linear transformations
        self.egat = EGATLayer(f_h, f_e, lamda, num_heads)  ## Node and edge information update
        self.pred = MLPPredictor(f_e,pred_hid,1)  ## MLP output
        self.merge = MergeLayer(f_h,f_e,l_num)  #last, merging the results of multiple attention calculations
        self.dropout = dropout
        self.embedding_1 = nn.Embedding(10,2)  #equal to calsses of land cover
        self.embedding_2 = nn.Embedding(13,2)  #equal to calsses of lithology


    def forward(self, graph, h_in,e_in):
        #graph, node_features, edge_features
        #embedded_1 = self.embedding_1(h_in[:,-2].long())
        #embedded_2 = self.embedding_2(h_in[:,-1].long())
        #combined = th.cat((h_in[:,:-2], embedded_1.view(embedded_1.size(0), -1), embedded_2.view(embedded_2.size(0), -1)), dim=1)

        
        h_out,e_out = self.bottleneck(h_in,e_in)  #first layer
        #print("----------------------first layer")
        #print(h_out.shape,e_out.shape)
        #print(h_in.shape,e_in.shape)
        h_final = h_out
        e_final = e_out
        for i in range(self.l):
          h_out,e_out = self.egat(graph,h_out,e_out)  #second layer, with multiple EGAT layers
          if i != 0:#Merge Layer
            h_final = th.cat([h_final,h_out], dim=-1)
            e_final = th.cat([e_final,e_out], dim=-1)
        #print('--------------------seconde layer')
        #print(h_out.shape,e_out.shape)
        
        h_final, e_final = self.merge(h_final, e_final)   # last layer
        #print('----------------------last layer')
        #print("h_final")
        #print(h_final.shape)
        return self.pred(h_final)     
