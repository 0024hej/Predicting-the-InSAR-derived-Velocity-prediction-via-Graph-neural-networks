import dgl
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import r2_score
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def r2_fun(output,labels):
    out = output.detach().numpy()
    lab = labels.detach().numpy()
    
    r2 = r2_score(out,lab)
    return r2


def load_data():

    num_nodes, num_edges = 8, 30
    # generate a graph
    graph = dgl.rand_graph(num_nodes,num_edges)
    
    node_features = torch.rand((num_nodes, 20))
    edge_features = torch.rand((num_edges, 20))
    labels = encode_onehot([1,0,1,1,0,1,1,0,0,0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,1,0,1])
    print(labels)
    
    node_features = normalize(node_features)
    edge_features = normalize(edge_features)

    idx_train = range(10)
    idx_val = range(10, 20)
    idx_test = range(20, 30)

    node_features = torch.FloatTensor(np.array(node_features))
    edge_features = torch.FloatTensor(np.array(edge_features))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print("1111111111111111111111111")

    print(labels)
    return graph, node_features, edge_features,labels, idx_train, idx_val, idx_test
